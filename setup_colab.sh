#!/bin/bash
# Google Colab 用セットアップスクリプト
# Colab のセルで実行: !bash setup_colab.sh

set -e

echo "=== RoboQuest2026 セットアップ ==="

# Run from this checkout, irrespective of the caller's working directory.
cd "$(cd "$(dirname "$0")" && pwd)"

echo "[1/3] ローカルと共通のライブラリをインストール中..."
python3 -m pip install -q -r requirements.txt

# mjswan（ブラウザビューアー）は別途インストールする。
# - 0.8.2 固定：0.9 系は API 破壊（観測関数が mjlab へ移動）＋ mjlab との mujoco pin 衝突。
# - mjswan の wheel は Requires-Python が <3.13 だが、中身は純 Python + ビルド済み
#   フロントエンドで、依存（mujoco==3.8.1 / onnx）にも cp313 wheel があるため
#   Python 3.13（現在の Colab）でも動作する。3.13 では pip が候補から外すので
#   --ignore-requires-python を付ける。
# - mujoco は mjswan の pin（==3.8.1）に従うのでここでは指定しない。
MJSWAN_FLAGS=""
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)'; then
  MJSWAN_FLAGS="--ignore-requires-python"
fi
if ! python3 -m pip install -q $MJSWAN_FLAGS -c requirements-training.txt "mjswan==0.8.2"; then
  echo "⚠ mjswan のインストールに失敗しました。ブラウザビューアーのセルのみ使えません。"
  echo "  学習・動画のセルはそのまま実行できます。"
fi

# 3. Go2 モデルファイルのダウンロード
echo "[2/3] Go2 モデルをダウンロード中..."
python3 scripts/download_models.py

# 4. パッケージのインストール
echo "[3/3] roboquest パッケージをインストール中..."
python3 -m pip install -q -e . -c requirements-training.txt

# ブラウザビューアーは mjswan がノートブック内でビルドする（vendor assets 不要）。
# mjswan 0.8.2 の wheel はビルド済み dist を同梱しているので Node は不要。

echo ""
echo "✅ セットアップ完了！ノートブックを実行できます。"
