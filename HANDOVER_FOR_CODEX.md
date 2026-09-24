# RoboQuest 2026 — Handover Document for Codex

**Project:** RoboQuest 2026  
**Repo:** https://github.com/SingularityBattleQuest/RoboQuest2026  
**Purpose:** RL teaching environment for Japanese high school students  
**Date:** 2026-05-24  
**Prepared by:** Claude Sonnet 4.6

---

## 1. Project Overview

RoboQuest 2026 is a reinforcement learning competition platform where Japanese high school students train a Unitree Go2 quadruped robot to **escape from an "oni" (demon) for 60 seconds** inside a 5×5m walled arena. Everything runs serverless in Google Colab — no local setup required.

### Learning Flow (notebook: `notebooks/quickstart_ja.ipynb`)
1. **Phase 1 (Walk):** Student tunes reward weights → PPO trains `Go2WalkEnv` → walk policy saved
2. **Phase 2 (Flee):** Student tunes reward weights → PPO trains `Go2TagHierarchicalEnv` → flee policy saved
3. **Phase 3 (View/Evaluate):** mjswan displays the walk policy with manual velocity commands; `evaluate_flee` runs the saved hierarchy with frozen normalization and fixed evaluation seeds.

`notebooks/guide_and_experiments_ja.ipynb` uses the same workflow with explanations and experiments. Shared training, persistence, and evaluation live in `scripts/notebook_workflow.py`. Both notebooks save to `MyDrive/RoboQuest2026/{team_name}/{experiment_name}`.

---

## 2. Repository Structure

```
RoboQuest2026/
├── models/go2/
│   ├── go2.xml              # Full Go2 model (uses OBJ meshes, for local use)
│   ├── go2_simple.xml       # Collision-only Go2 (no meshes, for browser/webapp)
│   ├── walk_scene.xml       # Walk env: Go2 + floor
│   ├── arena.xml            # Flee env: Go2 + floor + 4 walls + oni body
│   └── arena_web.xml        # Browser viewer: go2_simple + arena (no OBJ)
│
├── roboquest/
│   ├── envs/
│   │   ├── go2_walk_env.py          # Phase 1: velocity-tracking walk env
│   │   ├── go2_tag_env.py           # Tag game base env
│   │   └── go2_tag_hierarchical_env.py  # Phase 2: hierarchical flee env
│   └── utils/
│       ├── flask_viewer.py          # ★ MAIN VIEWER — Flask + JPEG polling
│       ├── colab_viewer.py          # Alt viewer: kernel.invokeFunction + Canvas2D
│       ├── reward_utils.py          # WalkRewardConfig, FleeRewardConfig dataclasses
│       └── visualization.py        # (legacy)
│
├── notebooks/
│   ├── quickstart_ja.ipynb        # ★ MAIN NOTEBOOK — student-facing, JP
│   └── guide_and_experiments_ja.ipynb      # Explained workflow and experiments
│
├── scripts/
│   ├── download_models.py           # Downloads Go2 MJCF + mesh assets
│   ├── export_for_web.py            # Exports PPO → ONNX + VecNorm stats JSON
│   ├── train_walk.py                # CLI walk training
│   └── train_flee.py                # CLI flee training
│
├── webapp/                          # Browser viewer (mujoco_wasm + Three.js)
│   ├── index.html
│   └── setup_wasm.sh               # Downloads mujoco_wasm binaries
│
└── competition/
    └── run_match.py                 # Head-to-head match runner
```

---

## 3. Key Technical Details

### 3.1 RL Architecture

**Phase 1 — Walk Policy**
- **Env:** `Go2WalkEnv` (`roboquest/envs/go2_walk_env.py`)
- **Obs (45-dim):** vel_cmd(3) + ang_vel(3) + projected_gravity(3) + joint_pos(12) + joint_vel(12) + last_action(12)
- **Action (12-dim):** Normalized joint angle offsets [-1, 1]
- **Control:** PD controller: `tau = KP*(STANDING_POS + action*0.3 - q) - KD*dq`
- **Constants:** `KP=20.0, KD=0.5, ACTION_SCALE=0.3`
- **Saved to:** `/tmp/walk_model.zip` + `/tmp/walk_model_vecnorm.pkl`

**Phase 2 — Flee Policy (Hierarchical)**
- **Env:** `Go2TagHierarchicalEnv` (`roboquest/envs/go2_tag_hierarchical_env.py`)
- **High-level obs (10-dim):** oni_dx(1) + oni_dy(1) + dist(1) + ang_vel(3) + projected_gravity(3) + time_left(1)
- **High-level action (3-dim):** [vx, vy, omega] velocity command → fed into frozen walk policy
- **Hierarchy:** High-level runs at 5Hz, Low-level (walk) at 50Hz → `N_LOW_STEPS = 10`
- **Termination:** tagged (dist < 0.65m), fallen (z < 0.15m), escaped (60s survived)
- **Saved to:** `/tmp/flee_model.zip` + `/tmp/flee_model_vecnorm.pkl`

### 3.2 MuJoCo XML Files

| File | nq | Use |
|------|-----|-----|
| `walk_scene.xml` | 19 | Walk training (7 free + 12 leg joints) |
| `arena.xml` | 21 | Flee training (+ 2 oni slide joints) |
| `go2_simple.xml` | — | Collision-only, no OBJ meshes (for webapp) |
| `arena_web.xml` | 21 | Browser viewer (includes go2_simple) |

**Joint order in `data.qpos`:**
- `[0:3]` = base xyz
- `[3:7]` = base quaternion (w, x, y, z)
- `[7:19]` = leg joints: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
- `[19]` = oni x (arena.xml only)
- `[20]` = oni y (arena.xml only)

### 3.3 FlaskViewer (`roboquest/utils/flask_viewer.py`)

The core real-time viewer. Architecture:
```
Thread 1: _sim_loop()    — 50Hz physics simulation, acquires self._lock
Thread 2: _render_loop() — 25fps JPEG rendering, acquires self._lock
Thread 3: Flask server   — serves /frame /status /cmd
Main:     display()      — renders HTML directly into Colab cell output
```

**Critical design decision (as of 2026-05-24):**  
`display()` uses `IPython.display.HTML()` to render the full viewer UI **directly in the Colab cell output**, NOT inside an IFrame. All Flask API calls use absolute `proxy_url` paths:
```python
proxy_url = eval_js(f'google.colab.kernel.proxyPort({actual_port})')
ipy_display(HTML(self._notebook_html(proxy_url, actual_port)))
```
This bypasses Colab's IFrame auth interception and X-Frame-Options blocking that caused blank white pages.

**Flask endpoints:**
- `GET /` — Standalone HTML page (for opening in new tab)
- `GET /frame` — Latest JPEG frame (no-store cache)
- `GET /status` — JSON: `{time, rz, dist, done, reason, running}`
- `POST /cmd` — JSON: `{action: "play"|"pause"|"reset"|"vel"|"oni_speed", ...}`

**MUJOCO_GL handling:**
```python
# At top of flask_viewer.py (before import mujoco):
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"  # fallback
```
Setup cell sets `egl` (GPU) or `osmesa` (CPU) before any mujoco import.

---

## 4. Colab Setup Cell (cell[1] in notebook)

This cell must run first and sets critical env vars before any mujoco import:

```python
_has_gpu = subprocess.run('nvidia-smi', ...).returncode == 0
if _has_gpu:
    # Fix NVIDIA EGL ICD (required in Colab GPU runtime)
    open('/usr/share/glvnd/egl_vendor.d/10_nvidia.json', 'w').write(
        '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n')
    os.environ['MUJOCO_GL'] = 'egl'
else:
    subprocess.run('apt-get install -y -q libosmesa6', ...)
    os.environ['MUJOCO_GL'] = 'osmesa'

pip install: mujoco gymnasium stable-baselines3[extra] flask Pillow mediapy tqdm
git clone https://github.com/SingularityBattleQuest/RoboQuest2026.git /content/RoboQuest2026
python scripts/download_models.py
```

### 4.1 mjswan on Colab's Python 3.13

Colab now runs **Python 3.13**, and *every* published `mjswan` release (0.0.9 … 0.9.3)
declares `Requires-Python: <3.13,>=3.10`. pip therefore refuses to install it and the
whole setup cell used to die with `CalledProcessError` — students could not even train,
not just view.

The version cap is metadata only: mjswan 0.8.2 is pure Python plus a prebuilt frontend
bundle, and its dependencies (`mujoco==3.8.1`, `onnx`, `wandb`, `typer`, `rich`,
`nodeenv`) all ship cp313 wheels. Verified on Python 3.13: install, `import mjswan`, and
`build_walk()` / `build_flee()` (including the vite frontend build) all succeed.

So mjswan is installed **separately from the other libraries**, with the cap ignored on
3.13:

```python
flags = ['--ignore-requires-python'] if sys.version_info >= (3, 13) else []
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *flags, 'mjswan==0.8.2'])
```

Rules to keep:

* Keep the mjswan install in its **own** pip invocation — `--ignore-requires-python`
  applies to the whole resolution, so it must not leak into the other packages.
* Never make that install `check=True` at setup time. A mjswan failure must degrade to
  "viewer cells unavailable", never break training. The viewer cells re-install it
  themselves (`importlib.util.find_spec('mjswan') is None`) so they still work after a
  runtime restart.
* `setup.py` / `requirements.txt` carry the marker `mjswan==0.8.2; python_version < "3.13"`
  so `pip install -e .` and `pip install -r requirements.txt` do not fail on 3.13.
* Still pinned to 0.8.2: 0.9.x moved the observation functions into `mjlab`, whose
  `mujoco` pin conflicts with mjswan's.

---

## 5. Viewer Cells (cell[7] and cell[12])

Both viewer cells must clear the Python module cache after git pull:

```python
subprocess.run(['git', '-C', '/content/RoboQuest2026', 'pull', 'origin', 'main'], ...)

# CRITICAL: clear cached modules so fresh code is loaded
for _k in [k for k in sys.modules if k.startswith('roboquest')]:
    del sys.modules[_k]

from roboquest.utils.flask_viewer import FlaskViewer, WALK_XML
```

Without the `sys.modules` clearing, `git pull` updates disk but Python still runs old cached code.

---

## 6. Current Status & Known Issues

### ✅ Working
- Walk training (`Go2WalkEnv`) — verified
- Flee training (`Go2TagHierarchicalEnv`) — verified
- `FlaskViewer._render_loop` — MuJoCo Renderer initializes with `✅ MuJoCo Renderer 初期化成功` 
- Placeholder JPEG on startup (no empty-bytes /frame response)
- Auto port detection (socket scan across 20 ports)
- Module cache clearing in viewer cells

### 🔴 Unverified / Pending
- **FlaskViewer inline display** — The IFrame → `HTML()` rewrite was just pushed (commit `40b5cdf`). Has NOT been confirmed working by user yet. This is the #1 priority to verify.
- The `_notebook_html()` method generates a self-contained UI with absolute proxy URLs. If it fails, check:
  1. Does `proxy_url` end with `/`? (code strips trailing slash: `proxy_url.rstrip('/')`)
  2. Is the JavaScript inside `_notebook_html()` executing? (check browser DevTools console)
  3. Is `/frame` returning valid JPEG? (open `proxy_url/frame` directly in browser)

### ⚠️ Edge Cases
- **Multiple viewer cells running simultaneously** — each gets a different port; uid (port number) is used as HTML element ID prefix to avoid conflicts
- **Colab runtime restart** — after restart, `/tmp/walk_model.zip` is lost; users must re-run training cells
- **Port exhaustion** — if 20 ports (7860–7879) are all in use, display() fails with error message

---

## 7. File Paths on Colab Runtime

```
/content/RoboQuest2026/          ← git clone destination (os.chdir here)
/content/RoboQuest2026/models/go2/  ← MJCF XML files
/tmp/walk_model.zip              ← saved walk policy
/tmp/walk_model_vecnorm.pkl      ← VecNormalize stats for walk
/tmp/flee_model.zip              ← saved flee policy  
/tmp/flee_model_vecnorm.pkl      ← VecNormalize stats for flee
/content/drive/MyDrive/RoboQuest2026/{team_name}/  ← Google Drive backup
```

---

## 8. Reward Configs (student-tunable)

```python
# roboquest/utils/reward_utils.py
@dataclass
class WalkRewardConfig:
    lin_vel_weight:    float = 1.0    # velocity tracking (linear)
    ang_vel_weight:    float = 0.5    # velocity tracking (angular)
    orientation_weight: float = -1.0  # penalize tilting
    torques_weight:    float = 0.0    # penalize energy use
    action_rate_weight: float = -0.05 # penalize jerky motion
    # ... more fields

@dataclass
class FleeRewardConfig:
    survival_weight: float = 0.5    # reward per step survived
    distance_weight: float = 1.0    # reward proportional to oni distance
    tag_penalty:     float = 50.0   # penalty on capture
```

---

## 9. What Codex Should Work On Next

**Priority 1 — Confirm FlaskViewer works end-to-end**  
The last commit (`40b5cdf`) rewrote `display()` to use `HTML()` instead of `IFrame()`. The user has not yet confirmed this works. Verify that:
1. Running the viewer cell shows the robot simulation inline in Colab
2. `▶ 開始` button starts the simulation
3. WASD keys control the robot

**Priority 2 — If still broken, add `/ping` debug endpoint**  
Add `GET /ping` to Flask that returns `text/plain` "pong". Tell user to open `proxy_url/ping` in browser to confirm Flask is reachable before debugging the UI.

**Priority 3 — `_notebook_html()` polish**  
- Height is currently hardcoded via `min-height:{h}px` on the video div. If user passes `height=` to `display()`, it should resize the whole viewer.
- The `#main` section needs `height: {height}px` to match.

**Priority 4 — Walk policy not used in viewer**  
`FlaskViewer` for walk (`with_oni=False, policy=None`) does not load the walk model — it's purely manual control via WASD. If the user wants to see the AI walk without touching WASD, they should be able to pass the trained walk policy as `policy=walk_model`. The `_step_sim()` method already supports this.

**Priority 5 — Competition runner**  
`competition/run_match.py` likely needs updating to match the current `Go2TagHierarchicalEnv` API.

---

## 10. Dependencies

```
mujoco>=3.0            # mjswan 0.8.2 pins it to ==3.8.1 in practice
gymnasium>=0.29
stable-baselines3[extra]>=2.0
onnx>=1.20.0
onnxruntime>=1.17.0
mediapy, tqdm, pandas, matplotlib
mjswan==0.8.2          # browser viewer; see §4.1 for the Python 3.13 install rule
numpy
```

On Colab GPU: also needs `libosmesa6` (if CPU fallback) and NVIDIA EGL ICD fix.

---

## 11. Git History (recent)

```
40b5cdf  Rewrite display(): render UI directly in notebook, drop IFrame entirely
fd18436  Fix white page: CORS headers, serve_kernel_port_as_iframe, explicit content-type
605ec02  Fix module cache issue: clear sys.modules before import in viewer cells
e39d8bb  Fix flask_viewer: MUJOCO_GL fallback, placeholder JPEG, render error logging
7ac1d54  Restore setup cell: GPU detection, mujoco install, repo clone, flask+Pillow
25a23cb  Fix FlaskViewer: auto port detection + Pillow install
d66ca10  Fix viewer cells: #@title first, git pull silent, fix flee_model_path
b88dd64  Replace fake video cells with Flask real-time MuJoCo viewer
```

---

*Document generated 2026-05-24 — Claude Sonnet 4.6*
