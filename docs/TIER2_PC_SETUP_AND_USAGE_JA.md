# Tier2 PC 実行環境と使用方法

推奨環境は WSL2 Ubuntu + Python 3.12 です。Windowsネイティブでも動く可能性はありますが、Colabに近いLinux環境での再現性を優先します。

## セットアップ

```bash
cd ~/RoboQuest2026-main
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt -c requirements-training.txt
```

## 1. ベンチマーク

まずCPU/GPUと並列環境数を比較します。

```bash
python -m scripts.benchmark_tier2 --config configs/tier2_benchmark.yaml
```

結果は `runs/tier2_benchmark/benchmark_results.json` に保存されます。`steps_per_sec` が高く、安定している設定を `configs/tier2_compat_ga.yaml` の `default_training` に反映してください。

## 2. Tier2 GA 学習

```bash
python -m scripts.train_tier2_ga --config configs/tier2_compat_ga.yaml
```

再開する場合:

```bash
python -m scripts.train_tier2_ga --config configs/tier2_compat_ga.yaml --resume runs/tier2/実験フォルダ名
```

## 3. 評価

```bash
python -m scripts.evaluate_tier2 --run runs/tier2/latest --candidate best --seeds final
```

`--seeds coarse`, `--seeds medium`, `--seeds final` が使えます。個別指定もできます。

```bash
python -m scripts.evaluate_tier2 --run runs/tier2/latest --candidate best --seeds "100 101 102"
```

## 4. 動画生成

```bash
python -m scripts.record_tier2_video --run runs/tier2/latest --candidate best --seeds "100 101 102"
```

動画は `runs/tier2/.../videos/` に保存されます。

## 5. 提出パッケージ

```bash
python -m scripts.package_submission --run runs/tier2/latest --candidate best
```

出力先は `runs/tier2/.../submission_package/` です。

## 6. 変則ルート

幾何学教師データを集める場合:

```bash
python -m scripts.collect_teacher_data --episodes 100 --output runs/tier2_teacher/teacher_data.npz
python -m scripts.imitate_teacher --data runs/tier2_teacher/teacher_data.npz
```

このルートは、MPCや幾何学コントローラの判断を方策に焼き込むための入口です。提出互換PPOへの統合は次段階の作業です。
