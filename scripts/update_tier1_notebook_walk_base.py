"""Update the Tier1 Colab notebook to use the PC-trained walk base."""
from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path("notebooks/tier1_colab_ga_ja.ipynb")


SETUP_PATCH = r'''

# Tier1/Tier2共通の強い歩行ベースを使うためのColabパッチ。
# 1. Tag環境も歩行学習と同じ go2_posctrl.xml を使う。
# 2. 高レベル逃げ命令は、低レベル歩行が得意な前進+軽い旋回へ写像する。
def patch_colab_tier1_environment(repo):
    repo = Path(repo)
    arena_posctrl = repo / 'models/go2/arena_posctrl.xml'
    arena_posctrl.write_text("""<mujoco model="go2_arena_posctrl">
  <include file="go2_posctrl.xml"/>
  <statistic center="0 0 0.3" extent="4"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.1 0.15 0.2 1"/>
    <global azimuth="120" elevation="-30"/>
  </visual>
  <asset>
    <texture type="2d" name="floor_tex" builtin="checker" rgb1="0.35 0.55 0.35" rgb2="0.28 0.45 0.28" width="256" height="256"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="5 5" reflectance="0.1"/>
    <material name="wall_mat" rgba="0.55 0.55 0.65 1"/>
    <material name="oni_mat"  rgba="0.95 0.12 0.12 1"/>
  </asset>
  <worldbody>
    <light name="main_light" pos="0 0 5" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" specular="0.1 0.1 0.1"/>
    <light name="fill_light" pos="0 0 3" dir="0 1 -1" directional="true" diffuse="0.3 0.3 0.35" specular="0 0 0"/>
    <geom name="floor" type="plane" size="2.5 2.5 0.1" material="floor_mat" contype="1" conaffinity="1" friction="0.8 0.1 0.1"/>
    <geom name="wall_n" type="box" pos="0  2.55 0.5" size="2.6 0.05 0.5" material="wall_mat" contype="1" conaffinity="1"/>
    <geom name="wall_s" type="box" pos="0 -2.55 0.5" size="2.6 0.05 0.5" material="wall_mat" contype="1" conaffinity="1"/>
    <geom name="wall_e" type="box" pos=" 2.55 0 0.5" size="0.05 2.6 0.5" material="wall_mat" contype="1" conaffinity="1"/>
    <geom name="wall_w" type="box" pos="-2.55 0 0.5" size="0.05 2.6 0.5" material="wall_mat" contype="1" conaffinity="1"/>
    <camera name="top_view" pos="0 0 9" euler="0 0 0" fovy="60"/>
    <camera name="side_view" pos="6 -4 4" euler="55 0 45" fovy="50"/>
    <body name="oni" pos="0 0 0.3">
      <joint name="oni_x" type="slide" axis="1 0 0" limited="true" range="-2.4 2.4" damping="100" frictionloss="0"/>
      <joint name="oni_y" type="slide" axis="0 1 0" limited="true" range="-2.4 2.4" damping="100" frictionloss="0"/>
      <geom name="oni_body" type="sphere" size="0.28" material="oni_mat" contype="0" conaffinity="0" mass="0.001"/>
      <geom name="oni_eye" type="sphere" size="0.08" pos="0.25 0 0.1" rgba="1 1 1 1" contype="0" conaffinity="0" mass="0"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="arena_home" qpos="0 0 0.27 1 0 0 0 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 1.6 1.6" ctrl="0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8 0 0.9 -1.8"/>
  </keyframe>
</mujoco>
""", encoding='utf-8')

    tag_env_path = repo / 'roboquest/envs/go2_tag_env.py'
    tag_text = tag_env_path.read_text(encoding='utf-8')
    tag_text = tag_text.replace('ARENA_XML = os.path.join(_MODEL_DIR, "arena.xml")',
                                'ARENA_XML = os.path.join(_MODEL_DIR, "arena_posctrl.xml")')
    tag_env_path.write_text(tag_text, encoding='utf-8')

    hier_path = repo / 'roboquest/envs/go2_tag_hierarchical_env.py'
    hier_text = hier_path.read_text(encoding='utf-8')
    changed = False
    if 'high_level_command_mode: str = "direct"' in hier_text:
        hier_text = hier_text.replace('high_level_command_mode: str = "direct"',
                                      'high_level_command_mode: str = "safe_forward"')
        changed = True
    old_ranges = '''                "vx": (0.25, 0.55),
                "vy": (-0.05, 0.05),
                "omega": (-0.20, 0.20),'''
    new_ranges = '''                "vx": (0.75, 1.55),
                "vy": (-0.02, 0.02),
                "omega": (-0.95, 0.95),'''
    if old_ranges in hier_text:
        hier_text = hier_text.replace(old_ranges, new_ranges)
        changed = True
    if changed:
        hier_path.write_text(hier_text, encoding='utf-8')
        print('Tier1 environment patch: arena_posctrl + race-speed safe_forward defaults applied')
    else:
        print('Tier1 environment patch: arena_posctrl applied. race-speed defaults were already applied or not found.')

patch_colab_tier1_environment(REPO)
'''


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    cell1 = "".join(nb["cells"][1]["source"])
    if "patch_colab_tier1_environment" not in cell1:
        cell1 = cell1.replace("print('setup done')", "print('setup done')" + SETUP_PATCH)
        nb["cells"][1]["source"] = cell1.splitlines(keepends=True)

    cell2 = "".join(nb["cells"][2]["source"])
    insert = """

# 歩行ベースの選択。
# drive_zip: PCで作った強い歩行ベースzipをDriveから読む。推奨。
# repo_pretrained: GitHub同梱のsmooth_walkを使う。比較・緊急用。
walk_source_mode = 'drive_zip' #@param ['drive_zip', 'repo_pretrained']
walk_bundle_zip = '/content/drive/MyDrive/RoboQuest2026_GA/walk_bases/tier1_walk_speed_race_001_bundle.zip' #@param {type:'string'}
"""
    if "walk_source_mode" not in cell2:
        marker = "show_ppo_progress = True #@param {type:'boolean'}\n"
        cell2 = cell2.replace(marker, marker + insert)
        nb["cells"][2]["source"] = cell2.splitlines(keepends=True)

    cell3 = "".join(nb["cells"][3]["source"])
    old = """def copy_walk_files(dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    src = REPO / 'models/pretrained/smooth_walk'
    # 公式quickstartの retrain_walk=False と同じく同梱歩行モデルを使用します。
    for name in ['walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json']:
        shutil.copy2(src / name, dst / name)
"""
    new = """def prepare_walk_source():
    source_dir = BASE_DIR / 'walk_source'
    source_dir.mkdir(parents=True, exist_ok=True)
    required = ['walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json']
    if walk_source_mode == 'drive_zip':
        bundle = Path(walk_bundle_zip)
        if not bundle.exists():
            raise FileNotFoundError(f'歩行ベースzipが見つかりません: {bundle}')
        shutil.unpack_archive(str(bundle), str(source_dir))
        print('Drive zipから歩行ベースを読み込みました:', bundle)
    elif walk_source_mode == 'repo_pretrained':
        source_dir = REPO / 'models/pretrained/smooth_walk'
        print('repo同梱のsmooth_walkを使います:', source_dir)
    else:
        raise ValueError(walk_source_mode)
    missing = [name for name in required if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f'歩行ベースに必要ファイルがありません: {missing} in {source_dir}')
    (BASE_DIR / 'walk_source_manifest.json').write_text(json.dumps({
        'walk_source_mode': walk_source_mode,
        'walk_bundle_zip': walk_bundle_zip,
        'resolved_source_dir': str(source_dir),
        'required': required,
        'prepared_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return Path(source_dir)

WALK_SOURCE_DIR = prepare_walk_source()


def copy_walk_files(dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ['walk_model.zip', 'walk_model_vecnorm.pkl', 'walk_params.json']:
        shutil.copy2(WALK_SOURCE_DIR / name, dst / name)
"""
    if old in cell3:
        cell3 = cell3.replace(old, new)
        nb["cells"][3]["source"] = cell3.splitlines(keepends=True)
    elif "def prepare_walk_source" not in cell3:
        raise RuntimeError("copy_walk_files block not found")

    cell5 = "".join(nb["cells"][5]["source"])
    cell5 = cell5.replace(
        "walk_check_model = REPO / 'models/pretrained/smooth_walk/walk_model'\n"
        "walk_check_vecnorm = REPO / 'models/pretrained/smooth_walk/walk_model_vecnorm.pkl'",
        "walk_check_model = WALK_SOURCE_DIR / 'walk_model'\n"
        "walk_check_vecnorm = WALK_SOURCE_DIR / 'walk_model_vecnorm.pkl'",
    )
    cell5 = cell5.replace(
        "def check_walk_command(action, label, seed_value=100, max_high_steps=150):",
        "def check_walk_command(action, label, seed_value=100, max_high_steps=150, required_seconds=None):",
    )
    cell5 = cell5.replace(
        "        return survived, height, reason",
        "        ok = survived >= (required_seconds if required_seconds is not None else max_high_steps * 0.2)\n"
        "        return survived, height, reason, ok",
    )
    cell5 = cell5.replace(
        "    check_walk_command([0.0, 0.0, 0.0], 'stand'),\n"
        "    check_walk_command([1.0, 0.0, 0.0], 'forward_1.0'),\n"
        "    check_walk_command([1.4, 0.0, 0.0], 'forward_1.4'),",
        "    check_walk_command([0.0, 0.0, 0.0], 'stand', max_high_steps=150, required_seconds=20.0),\n"
        "    # 高速直進を長く続けると5mアリーナの壁に当たるため、短時間の安定性だけ確認します。\n"
        "    check_walk_command([1.0, 0.0, 0.0], 'forward_1.0', max_high_steps=15, required_seconds=3.0),\n"
        "    check_walk_command([1.4, 0.0, 0.0], 'forward_1.4', max_high_steps=15, required_seconds=3.0),",
    )
    cell5 = cell5.replace(
        "if all(survived >= 10.0 for survived, _, _ in checks):",
        "if all(ok for _, _, _, ok in checks):",
    )
    nb["cells"][5]["source"] = cell5.splitlines(keepends=True)

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
