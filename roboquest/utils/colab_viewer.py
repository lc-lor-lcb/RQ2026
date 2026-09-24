"""
colab_viewer.py — Google Colab インライン インタラクティブビューアー

アーキテクチャ:
  Python (MuJoCo 物理) ←→ Colab kernel ←→ JavaScript (HTML5 Canvas 2D 描画)

JS が ~100ms ごとに Python コールバック `rq_poll` を呼び出し、
Python は 1 制御ステップ（5 物理ステップ）進めて JSON 状態を返す。
JS がその状態を 2D 俯瞰マップとして Canvas に描画する。

使い方:
    from roboquest.utils.colab_viewer import ColabViewer
    viewer = ColabViewer()
    viewer.display()  # ← これを Colab セルで実行
"""
from __future__ import annotations

import json
import os
from typing import Optional

import mujoco
import numpy as np

# ── パス ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_ARENA_XML = os.path.join(_HERE, "..", "..", "models", "go2", "arena.xml")

# ── 制御定数 ──────────────────────────────────────────────────────────────────
STANDING_POS = np.array([
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
    0.0,  0.9, -1.8,
], dtype=np.float64)

ACTION_SCALE = 0.3
KP = 20.0
KD = 0.5

MAX_TIME_S  = 60.0
TAG_DIST    = 0.65
ARENA_HALF  = 2.35
ONI_START   = (1.5, 1.5)

FOOT_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


class ColabViewer:
    """Colab インライン ビューアー。

    Parameters
    ----------
    xml_path  : arena XML パス（デフォルト = arena.xml）
    oni_speed : 鬼の移動速度 (m/step)
    policy    : SB3 PPO 高レベルポリシー（省略時 = 手動操作）
    cell_height : Colab セルの高さ (px)
    """

    def __init__(
        self,
        xml_path: Optional[str] = None,
        oni_speed: float = 0.025,
        policy=None,
        cell_height: int = 680,
    ):
        self.oni_speed   = oni_speed
        self.policy      = policy
        self.cell_height = cell_height

        xml = xml_path or _ARENA_XML
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data  = mujoco.MjData(self.model)

        # 足ボディ ID
        self._foot_ids = []
        for name in FOOT_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._foot_ids.append(bid)

        self._vel_cmd    = np.zeros(3)
        self._last_action = np.zeros(12)
        self._oni_x: float = ONI_START[0]
        self._oni_y: float = ONI_START[1]
        self._sim_time   = 0.0
        self._running    = False
        self._done       = False
        self._done_reason = ""

        self._reset()
        self._register_callbacks()

    # ── Colab コールバック ────────────────────────────────────────────────────

    def _register_callbacks(self) -> None:
        try:
            from google.colab import output
            output.register_callback("rq_poll",  self._cb_poll)
            output.register_callback("rq_reset", self._cb_reset)
        except ImportError:
            pass  # Colab 外でも import できるようにする

    def _cb_poll(self, cmd_json: str) -> str:
        """JS から呼ばれる: コマンド受信 → 1 ステップ進める → 状態 JSON を返す。"""
        try:
            cmd = json.loads(cmd_json)
        except json.JSONDecodeError:
            cmd = {}

        self._running = bool(cmd.get("running", self._running))
        if not self._done and self._running:
            vx    = float(cmd.get("vx",    0.0))
            vy    = float(cmd.get("vy",    0.0))
            omega = float(cmd.get("omega", 0.0))
            self._vel_cmd = np.array([vx, vy, omega])
            self._step()

        return json.dumps(self._get_state())

    def _cb_reset(self, _unused: str = "") -> str:
        self._reset()
        self._running = True
        return json.dumps(self._get_state())

    # ── シミュレーション ──────────────────────────────────────────────────────

    def _reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = 0.0

        self._oni_x, self._oni_y = ONI_START
        self.data.qpos[19] = self._oni_x
        self.data.qpos[20] = self._oni_y
        mujoco.mj_forward(self.model, self.data)

        self._vel_cmd     = np.zeros(3)
        self._last_action = np.zeros(12)
        self._sim_time    = 0.0
        self._done        = False
        self._done_reason = ""

    def _step(self) -> None:
        # 高レベルポリシー（あれば）
        if self.policy is not None:
            obs     = self._build_obs_for_policy()
            action, _ = self.policy.predict(obs, deterministic=True)
            self._vel_cmd = np.clip(action.astype(np.float64), -1, 1)

        # 低レベル PD 制御
        q_target = STANDING_POS + self._last_action * ACTION_SCALE
        q_cur    = self.data.qpos[7:19]
        dq_cur   = self.data.qvel[6:18]
        tau      = KP * (q_target - q_cur) - KD * dq_cur
        lim      = self.model.actuator_ctrlrange
        self.data.ctrl[:] = np.clip(tau, lim[:, 0], lim[:, 1])

        # 物理サブステップ
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # 鬼を追わせる
        rx, ry = float(self.data.qpos[0]), float(self.data.qpos[1])
        dx, dy = rx - self._oni_x, ry - self._oni_y
        dist   = float(np.sqrt(dx*dx + dy*dy))
        if dist > 0.01:
            self._oni_x = float(np.clip(
                self._oni_x + (dx / dist) * self.oni_speed, -ARENA_HALF, ARENA_HALF))
            self._oni_y = float(np.clip(
                self._oni_y + (dy / dist) * self.oni_speed, -ARENA_HALF, ARENA_HALF))
        self.data.qpos[19] = self._oni_x
        self.data.qpos[20] = self._oni_y
        mujoco.mj_forward(self.model, self.data)

        self._sim_time += 1 / 50.0

        # 終了判定
        rz = float(self.data.qpos[2])
        if dist < TAG_DIST:
            self._done, self._done_reason = True, "tagged"
        elif rz < 0.15:
            self._done, self._done_reason = True, "fallen"
        elif self._sim_time >= MAX_TIME_S:
            self._done, self._done_reason = True, "escaped"

    def _build_obs_for_policy(self) -> np.ndarray:
        """高レベルポリシー用 10 次元観測。"""
        rx, ry = self.data.qpos[0], self.data.qpos[1]
        dx = self._oni_x - rx
        dy = self._oni_y - ry
        dist = float(np.sqrt(dx*dx + dy*dy))
        ang_vel = self.data.qvel[3:6].copy()
        proj_grav = self._proj_gravity()
        time_left = (MAX_TIME_S - self._sim_time) / MAX_TIME_S
        return np.concatenate([[dx, dy, dist], ang_vel, proj_grav, [time_left]]).astype(np.float32)

    def _proj_gravity(self) -> np.ndarray:
        w, qx, qy, qz = self.data.qpos[3:7]
        R = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy+w*qz),    2*(qx*qz-w*qy)],
            [2*(qx*qy-w*qz),    1-2*(qx**2+qz**2),  2*(qy*qz+w*qx)],
            [2*(qx*qz+w*qy),    2*(qy*qz-w*qx),     1-2*(qx**2+qy**2)],
        ])
        return (R.T @ np.array([0, 0, -1])).astype(np.float64)

    def _get_state(self) -> dict:
        """JS Canvas 描画用コンパクト状態。"""
        qpos = self.data.qpos
        w, qx, qy, qz = qpos[3], qpos[4], qpos[5], qpos[6]
        yaw = float(np.arctan2(2*(w*qz + qx*qy), 1 - 2*(qy**2 + qz**2)))

        # 足の XY 位置
        feet_xy = []
        for bid in self._foot_ids:
            pos = self.data.xpos[bid]
            feet_xy.append([float(pos[0]), float(pos[1])])

        rx, ry = float(qpos[0]), float(qpos[1])
        dist = float(np.sqrt((rx - self._oni_x)**2 + (ry - self._oni_y)**2))

        return {
            "rx": rx, "ry": ry, "rz": float(qpos[2]),
            "yaw": yaw,
            "ox": self._oni_x, "oy": self._oni_y,
            "feet": feet_xy,
            "time": self._sim_time,
            "dist": dist,
            "done": self._done,
            "reason": self._done_reason,
            "running": self._running,
        }

    # ── 表示 ──────────────────────────────────────────────────────────────────

    def display(self) -> None:
        """Colab セル内にインタラクティブビューアーを表示する。"""
        from IPython.display import HTML, display as ipy_display
        ipy_display(HTML(self._build_html()))

    def _build_html(self) -> str:
        h = self.cell_height
        return f"""
<style>
#rq-wrap {{
  display: flex; gap: 12px; background: #0d0d14; padding: 12px;
  border-radius: 10px; font-family: 'Segoe UI', system-ui, sans-serif;
  color: #e0e0f0; user-select: none; width: fit-content;
}}
#rq-canvas {{
  border: 2px solid #2a3a2a; border-radius: 6px; display: block;
}}
#rq-panel {{
  width: 200px; display: flex; flex-direction: column; gap: 8px;
}}
.rq-section {{
  background: #13131f; border: 1px solid #1e1e30;
  border-radius: 6px; padding: 8px;
}}
.rq-section h3 {{
  margin: 0 0 6px 0; font-size: 0.68rem; text-transform: uppercase;
  letter-spacing: 0.08em; color: #4466aa;
}}
.rq-btn {{
  padding: 6px 0; border: 1px solid #2a3a5a; border-radius: 5px;
  background: #1a2440; color: #a0c0ff; cursor: pointer;
  font-size: 0.82rem; width: 100%; margin-bottom: 4px; transition: background .15s;
}}
.rq-btn:hover {{ background: #22305a; }}
.rq-btn.primary {{ background: #1e4080; border-color: #4080ff; font-weight: 600; }}
.rq-btn.primary:hover {{ background: #2a52a0; }}
.rq-btn-row {{ display: flex; gap: 4px; }}
.rq-btn-row .rq-btn {{ flex: 1; }}
.rq-stat {{ display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 4px; }}
.rq-stat span:first-child {{ color: #778; }}
.rq-stat span:last-child {{ color: #c0d8ff; font-weight: 600; font-variant-numeric: tabular-nums; }}
.rq-pbar {{ background: #1e1e30; border-radius: 3px; height: 5px; margin-bottom: 6px; overflow: hidden; }}
#rq-timebar {{ height: 100%; background: linear-gradient(90deg,#4080ff,#40c0ff); transition: width .15s; }}
.rq-dpad {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 3px; }}
.rq-dpad .rq-btn {{ margin-bottom: 0; padding: 7px 0; }}
.rq-slider label {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: #889; margin-bottom: 2px; }}
.rq-slider label span {{ color: #aac; font-weight: 600; }}
input[type=range] {{ width: 100%; accent-color: #4080ff; }}
</style>

<div id="rq-wrap">
  <canvas id="rq-canvas" width="500" height="500"></canvas>

  <div id="rq-panel">
    <!-- ステータス -->
    <div class="rq-section">
      <h3>📊 ステータス</h3>
      <div class="rq-stat"><span>生存時間</span><span id="rq-time">0.0 s</span></div>
      <div class="rq-pbar"><div id="rq-timebar" style="width:0%"></div></div>
      <div class="rq-stat"><span>鬼との距離</span><span id="rq-dist">-- m</span></div>
      <div class="rq-stat"><span>高さ</span><span id="rq-height">-- m</span></div>
      <div class="rq-stat"><span>状態</span><span id="rq-state">待機中</span></div>
    </div>

    <!-- シミュレーション -->
    <div class="rq-section">
      <h3>⏯ シミュレーション</h3>
      <div class="rq-btn-row">
        <button class="rq-btn primary" id="rq-playbtn">▶ 開始</button>
        <button class="rq-btn" id="rq-resetbtn">↺</button>
      </div>
      <div class="rq-slider">
        <label>速度倍率 <span id="rq-speedval">1×</span></label>
        <input type="range" id="rq-speed" min="1" max="8" step="1" value="1">
      </div>
    </div>

    <!-- 手動操作 -->
    <div class="rq-section" id="rq-manual-section">
      <h3>🕹 手動操作（WASD）</h3>
      <div class="rq-dpad">
        <div></div>
        <button class="rq-btn" data-vx="1" data-vy="0" data-om="0">↑</button>
        <div></div>
        <button class="rq-btn" data-vx="0" data-vy="0" data-om="1">←</button>
        <button class="rq-btn" data-vx="0" data-vy="0" data-om="0" id="rq-stopbtn">■</button>
        <button class="rq-btn" data-vx="0" data-vy="0" data-om="-1">→</button>
        <div></div>
        <button class="rq-btn" data-vx="-1" data-vy="0" data-om="0">↓</button>
        <div></div>
      </div>
    </div>

    <!-- 鬼の速度 -->
    <div class="rq-section">
      <h3>👹 鬼の速度</h3>
      <div class="rq-slider">
        <label>速度 <span id="rq-onispeedval">0.025</span></label>
        <input type="range" id="rq-onispeed" min="0.005" max="0.08" step="0.005" value="0.025">
      </div>
    </div>

    <!-- AI モード表示 -->
    <div class="rq-section" id="rq-ai-info" style="display:none">
      <h3>🤖 AI モード</h3>
      <div style="font-size:0.75rem; color:#60c060;">ポリシーが自動操縦中</div>
    </div>
  </div>
</div>

<script>
(function() {{
  // ── 状態 ──────────────────────────────────────────────────────────────────
  let running = false;
  let speed   = 1;
  let oniSpeed = 0.025;
  let velCmd  = {{vx: 0, vy: 0, omega: 0}};
  let inFlight = false;
  const HAS_POLICY = {'true' if self.policy else 'false'};
  const MAX_TIME = {MAX_TIME_S};
  const TAG_DIST = {TAG_DIST};
  const ARENA    = 5.0; // m
  const PX_PER_M = 500 / ARENA; // 100 px/m
  const CANVAS_H = 500;

  // ── Canvas ────────────────────────────────────────────────────────────────
  const canvas = document.getElementById('rq-canvas');
  const ctx    = canvas.getContext('2d');

  // ── ループ ────────────────────────────────────────────────────────────────
  let pollInterval = null;

  function startLoop() {{
    if (pollInterval) clearInterval(pollInterval);
    // 100ms ≒ 10 FPS (Python 側で 1 制御ステップ = 20ms 分進める)
    pollInterval = setInterval(poll, Math.max(100, Math.round(100 / speed)));
  }}

  async function poll() {{
    if (inFlight) return;
    inFlight = true;
    try {{
      const cmd = JSON.stringify({{
        vx: velCmd.vx, vy: velCmd.vy, omega: velCmd.omega,
        running: running, oniSpeed: oniSpeed,
      }});
      const result = await google.colab.kernel.invokeFunction('rq_poll', [cmd], {{}});
      const raw = result.data['text/plain'] || '{{}}';
      // Python が json.dumps() した文字列 → repr では 'json...' の形で届く
      const jsonStr = (raw[0] === "'" || raw[0] === '"') ? raw.slice(1, -1) : raw;
      const state = JSON.parse(jsonStr);
      render(state);
      updateStats(state);
    }} catch(e) {{ console.warn('rq poll error', e); }}
    inFlight = false;
  }}

  // ── 描画 ──────────────────────────────────────────────────────────────────
  function toCanvas(mx, my) {{
    // MuJoCo (x fwd, y left) → Canvas (x right, y down)
    return [
      CANVAS_H / 2 + mx * PX_PER_M,
      CANVAS_H / 2 - my * PX_PER_M,
    ];
  }}

  function render(state) {{
    ctx.clearRect(0, 0, 500, 500);

    // --- 床 ---
    ctx.fillStyle = '#1e3a1e';
    ctx.fillRect(0, 0, 500, 500);

    // --- グリッド ---
    ctx.strokeStyle = '#2a4a2a';
    ctx.lineWidth = 0.5;
    for (let i = -2; i <= 2; i++) {{
      const [x0, y0] = toCanvas(i, -2.5);
      const [x1, y1] = toCanvas(i,  2.5);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
      const [x2, y2] = toCanvas(-2.5, i);
      const [x3, y3] = toCanvas( 2.5, i);
      ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x3, y3); ctx.stroke();
    }}

    // --- 壁 ---
    ctx.strokeStyle = '#556677';
    ctx.lineWidth = 6;
    ctx.strokeRect(
      CANVAS_H/2 - 2.5*PX_PER_M, CANVAS_H/2 - 2.5*PX_PER_M,
      5*PX_PER_M, 5*PX_PER_M
    );

    // --- 足 (黒点) ---
    if (state.feet) {{
      state.feet.forEach(([fx, fy]) => {{
        const [px, py] = toCanvas(fx, fy);
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, 2*Math.PI);
        ctx.fillStyle = '#111';
        ctx.fill();
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 1;
        ctx.stroke();
      }});
    }}

    // --- ロボット胴体 ---
    const [rx, ry] = toCanvas(state.rx, state.ry);
    const yaw = state.yaw || 0; // MuJoCo yaw → canvas rotation
    ctx.save();
    ctx.translate(rx, ry);
    ctx.rotate(-yaw); // MuJoCo Y-left → canvas Y-down
    // 胴体 (0.376 × 0.094 m)
    ctx.fillStyle = '#e8c040';
    ctx.beginPath();
    ctx.roundRect(
      -0.188*PX_PER_M, -0.047*PX_PER_M,
       0.376*PX_PER_M,  0.094*PX_PER_M, 4
    );
    ctx.fill();
    ctx.strokeStyle = '#c09020';
    ctx.lineWidth = 1;
    ctx.stroke();
    // 前方インジケータ（青い三角）
    ctx.fillStyle = '#40a0ff';
    ctx.beginPath();
    ctx.moveTo( 0.188*PX_PER_M,  0);
    ctx.lineTo( 0.130*PX_PER_M, -7);
    ctx.lineTo( 0.130*PX_PER_M,  7);
    ctx.closePath(); ctx.fill();
    ctx.restore();

    // --- 鬼タグ範囲（点線円） ---
    const [ox, oy] = toCanvas(state.ox, state.oy);
    ctx.save();
    ctx.beginPath();
    ctx.arc(ox, oy, TAG_DIST * PX_PER_M, 0, 2*Math.PI);
    ctx.strokeStyle = 'rgba(255,60,60,0.4)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // --- 鬼 ---
    ctx.beginPath();
    ctx.arc(ox, oy, 0.28 * PX_PER_M, 0, 2*Math.PI);
    ctx.fillStyle = '#dd2222';
    ctx.fill();
    ctx.strokeStyle = '#ff6060';
    ctx.lineWidth = 2;
    ctx.stroke();
    // 目（白点）
    ctx.beginPath();
    ctx.arc(ox + 10, oy - 6, 5, 0, 2*Math.PI);
    ctx.fillStyle = 'white';
    ctx.fill();

    // --- タイマーバー ---
    const pct = Math.min(1, (state.time || 0) / MAX_TIME);
    ctx.fillStyle = '#0a0a12';
    ctx.fillRect(10, 8, 480, 8);
    const grad = ctx.createLinearGradient(10, 0, 490, 0);
    grad.addColorStop(0, '#4080ff');
    grad.addColorStop(1, '#40c0ff');
    ctx.fillStyle = grad;
    ctx.fillRect(10, 8, 480 * pct, 8);
    ctx.strokeStyle = '#2a3a5a';
    ctx.lineWidth = 1;
    ctx.strokeRect(10, 8, 480, 8);

    // --- 結果バナー ---
    if (state.done) {{
      const success = state.reason === 'escaped';
      ctx.fillStyle = 'rgba(10,10,20,0.80)';
      ctx.beginPath();
      ctx.roundRect(80, 190, 340, 120, 12);
      ctx.fill();
      ctx.strokeStyle = success ? '#40c060' : '#ff4040';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = success ? '#80ff80' : '#ff8080';
      ctx.font = 'bold 28px sans-serif';
      ctx.textAlign = 'center';
      const msg = success ? '🎉 逃げ切り成功！' :
                  state.reason === 'tagged' ? '😱 タグされた！' : '😵 転倒！';
      ctx.fillText(msg, 250, 245);
      ctx.fillStyle = '#8899bb';
      ctx.font = '16px sans-serif';
      ctx.fillText('生存時間: ' + (state.time||0).toFixed(1) + ' 秒', 250, 280);
      ctx.fillStyle = '#5566aa';
      ctx.font = '13px sans-serif';
      ctx.fillText('← リセットボタンで再挑戦', 250, 308);
      ctx.textAlign = 'left';
    }}
  }}

  // ── UI 更新 ───────────────────────────────────────────────────────────────
  function updateStats(state) {{
    document.getElementById('rq-time').textContent = (state.time||0).toFixed(1) + ' s';
    document.getElementById('rq-timebar').style.width = Math.min(100, (state.time||0)/MAX_TIME*100) + '%';
    document.getElementById('rq-dist').textContent   = (state.dist||0).toFixed(2) + ' m';
    document.getElementById('rq-height').textContent = (state.rz||0).toFixed(3) + ' m';

    const stEl = document.getElementById('rq-state');
    if (state.done) {{
      stEl.textContent = state.reason === 'escaped' ? '🎉 成功' :
                         state.reason === 'tagged'  ? '😱 タグ' : '😵 転倒';
    }} else if (!running) {{
      stEl.textContent = '停止中';
    }} else if ((state.dist||99) < 1.0) {{
      stEl.textContent = '🔴 危険！';
    }} else {{
      stEl.textContent = '🟢 逃走中';
    }}
  }}

  // ── ボタン ────────────────────────────────────────────────────────────────
  // Play / Pause
  const playBtn = document.getElementById('rq-playbtn');
  playBtn.addEventListener('click', () => {{
    running = !running;
    playBtn.textContent = running ? '⏸ 停止' : '▶ 開始';
  }});

  // Reset
  document.getElementById('rq-resetbtn').addEventListener('click', async () => {{
    running = false;
    playBtn.textContent = '▶ 開始';
    try {{
      const result = await google.colab.kernel.invokeFunction('rq_reset', [''], {{}});
      const raw = result.data['text/plain'] || '{{}}';
      const jsonStr = (raw[0] === "'" || raw[0] === '"') ? raw.slice(1, -1) : raw;
      render(JSON.parse(jsonStr));
    }} catch(e) {{ console.warn(e); }}
  }});

  // Speed slider
  document.getElementById('rq-speed').addEventListener('input', e => {{
    speed = parseInt(e.target.value);
    document.getElementById('rq-speedval').textContent = speed + '×';
    startLoop();
  }});

  // Oni speed slider
  document.getElementById('rq-onispeed').addEventListener('input', e => {{
    oniSpeed = parseFloat(e.target.value);
    document.getElementById('rq-onispeedval').textContent = oniSpeed.toFixed(3);
  }});

  // 方向ボタン
  document.querySelectorAll('.rq-dpad .rq-btn[data-vx]').forEach(btn => {{
    const setCmd = pressed => {{
      if (pressed) {{
        velCmd.vx    = parseFloat(btn.dataset.vx) * 0.8;
        velCmd.vy    = parseFloat(btn.dataset.vy) * 0.8;
        velCmd.omega = parseFloat(btn.dataset.om) * 0.8;
      }} else {{
        velCmd = {{vx: 0, vy: 0, omega: 0}};
      }}
    }};
    btn.addEventListener('pointerdown', () => setCmd(true));
    btn.addEventListener('pointerup',   () => setCmd(false));
    btn.addEventListener('pointerleave',() => setCmd(false));
  }});

  // WASD キーボード
  const keyMap = {{w:'vx+', s:'vx-', a:'om+', d:'om-', q:'vy+', e:'vy-'}};
  const pressed = new Set();
  document.addEventListener('keydown', e => {{
    if (HAS_POLICY) return;  // AI モード時はキー無効
    pressed.add(e.key.toLowerCase());
    let vx=0, vy=0, om=0;
    const s = 0.8;
    if (pressed.has('w')) vx += s;
    if (pressed.has('s')) vx -= s;
    if (pressed.has('a')) om += s;
    if (pressed.has('d')) om -= s;
    if (pressed.has('q')) vy += s * 0.5;
    if (pressed.has('e')) vy -= s * 0.5;
    velCmd = {{vx, vy, omega: om}};
  }});
  document.addEventListener('keyup', e => {{
    pressed.delete(e.key.toLowerCase());
    if (pressed.size === 0) velCmd = {{vx: 0, vy: 0, omega: 0}};
  }});

  // AI モード: 手動操作 UI を非表示
  if (HAS_POLICY) {{
    document.getElementById('rq-manual-section').style.display = 'none';
    document.getElementById('rq-ai-info').style.display = 'block';
  }}

  // ── 初期化 ────────────────────────────────────────────────────────────────
  // 初回ポーリングで状態を取得して描画
  (async () => {{
    try {{
      const result = await google.colab.kernel.invokeFunction('rq_poll', [JSON.stringify({{running: false}})], {{}});
      const raw = result.data['text/plain'] || '{{}}';
      const jsonStr = (raw[0] === "'" || raw[0] === '"') ? raw.slice(1, -1) : raw;
      render(JSON.parse(jsonStr));
    }} catch(e) {{ console.warn('init poll error', e); }}
  }})();

  startLoop();
}})();
</script>
"""
