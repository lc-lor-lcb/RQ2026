"""Export a selected saved Walk checkpoint without retraining it."""
from pathlib import Path
import tempfile


def preview_saved_walk(model_path, vecnorm_path=None):
    from stable_baselines3 import PPO
    from scripts.export_for_web import export_normalized_policy_onnx

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    model = PPO.load(str(model_path), device="cpu")
    if model.observation_space.shape != (45,) or model.action_space.shape != (12,):
        raise ValueError(
            f"このモデルは現在のWalkビューアーに対応しません。"
            f"観測={model.observation_space.shape}, 行動={model.action_space.shape}。"
            "必要な形は観測45・行動12です。"
        )
    # Separate each export so a previous team's ONNX/stats can never be reused.
    preview_dir = Path(tempfile.mkdtemp(prefix="rq_saved_walk_"))
    if vecnorm_path:
        stats = Path(vecnorm_path)
        if not stats.is_file():
            raise FileNotFoundError(f"指定した正規化データがありません: {stats}")
    else:
        candidates = [
            model_path.with_name(model_path.stem + "_vecnorm.pkl"),
            model_path.with_name("walk_vecnorm.pkl"),
        ]
        stats = next((p for p in candidates if p.is_file()), preview_dir / "missing.pkl")
    has_stats = stats.is_file()
    if has_stats:
        print(f"✅ 正規化データ: {stats.name}")
    else:
        print("⚠ 正規化データがありません。ZIPのみでプレビューします。")
        print("学習時と動きが異なったり、転倒したりする可能性があります。")
        print("この表示だけで学習の成功・失敗を判定しないでください。")
    onnx_path = preview_dir / "walk_policy_normalized.onnx"
    export_normalized_policy_onnx(model_path, stats, onnx_path)
    return onnx_path, has_stats
