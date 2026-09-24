"""
unitree_mujoco から Go2 MuJoCo モデルファイルを取得するスクリプト
"""
import os
import urllib.request
import sys

BASE_URL = "https://raw.githubusercontent.com/unitreerobotics/unitree_mujoco/main/unitree_robots/go2"
DEST_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "go2")

ROOT_FILES = [
    "go2.xml",
    "scene.xml",
    "scene_terrain.xml",
]

ASSET_FILES = [
    "base_0.obj",
    "base_1.obj",
    "base_2.obj",
    "base_3.obj",
    "base_4.obj",
    "calf_0.obj",
    "calf_1.obj",
    "calf_mirror_0.obj",
    "calf_mirror_1.obj",
    "foot.obj",
    "hip_0.obj",
    "hip_1.obj",
    "thigh_0.obj",
    "thigh_1.obj",
    "thigh_mirror_0.obj",
    "thigh_mirror_1.obj",
]


def download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  スキップ (既存): {os.path.basename(dest)}")
        return
    print(f"  ダウンロード中: {os.path.basename(dest)}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f"  警告: {os.path.basename(dest)} の取得に失敗しました: {e}")


def main() -> None:
    print("=== Go2 MuJoCo モデルのダウンロード ===")
    print(f"保存先: {os.path.abspath(DEST_DIR)}\n")

    print("[1/2] ルートファイルを取得...")
    for fname in ROOT_FILES:
        url = f"{BASE_URL}/{fname}"
        dest = os.path.join(DEST_DIR, fname)
        download_file(url, dest)

    print("\n[2/2] アセット(.obj)を取得...")
    for fname in ASSET_FILES:
        url = f"{BASE_URL}/assets/{fname}"
        dest = os.path.join(DEST_DIR, "assets", fname)
        download_file(url, dest)

    # 取得結果の確認
    xml_path = os.path.join(DEST_DIR, "go2.xml")
    if os.path.exists(xml_path):
        print("\n✅ 完了: go2.xml が正常に取得されました")
    else:
        print("\n❌ エラー: go2.xml が見つかりません。ネットワーク接続を確認してください")
        sys.exit(1)


if __name__ == "__main__":
    main()
