# RoboQuest2026 学習量の増やし方・進捗確認・再開手順

このメモは Tier1 Colab と Tier2 PC の共通運用手順です。目的は、いきなり重い学習を走らせて落とすのではなく、動作確認から段階的に学習量を増やして、最後に長時間実行へ移ることです。

## 1. Tier1 Colab

### 1.1 新しく始める

1. `notebooks/tier1_colab_ga_ja.ipynb` を Google Colab にアップロードする。
2. `run_name` を新しい名前にする。
3. `tier1_profile` を選ぶ。
4. 上から順に実行する。

例:

```python
run_name = "tier1_small_001"
tier1_profile = "small"
```

同じ `run_name` を使うと、Drive上の `state.json` を読み込んで再開しようとします。設定を変えて新しく始めたい場合は、必ず `run_name` を変えてください。

### 1.2 再開する

1. Colabを開く。
2. 前回と同じ `run_name` を指定する。
3. セットアップ、実験設定、共通関数のセルを実行する。
4. `GA実行` セルを実行する。

Driveに以下があれば再開できます。

```text
MyDrive/RoboQuest2026_GA/tier1/{run_name}/state.json
```

### 1.3 進捗確認

ノートブック内の「進捗状況の確認」セルを実行します。以下が表示されます。

- 保存済み世代
- 評価済み個体数
- 現在のベストスコア
- 平均生存秒数
- 最悪seed生存秒数
- escaped / tagged / fell
- 上位候補表
- 世代別サマリー

GA実行セルには個体単位の進捗バーも表示されます。

### 1.4 動画確認

「ベストモデルの動画確認」セルを実行します。

出力先:

```text
MyDrive/RoboQuest2026_GA/tier1/{run_name}/videos/
```

最初は `video_seeds = [100]` だけで確認し、問題なければ `[100, 101, 102]` に増やします。

### 1.5 Tier1 プロファイル

| profile | 用途 | population | generations | timesteps | envs |
|---|---:|---:|---:|---:|---:|
| `quick_test` | 動作確認 | 2 | 1 | 10,000 | 1 |
| `small` | 軽い探索 | 4 | 2 | 25,000 | 1 |
| `medium` | 標準探索 | 6 | 3 | 50,000 | 1 |
| `long` | 長めの探索 | 8 | 4 | 50,000 / 75,000 | 1 |
| `overnight` | 放置実行 | 8 | 6 | 75,000 / 100,000 | 1 |

Colab無料枠では `flee_num_envs=1` を基本にします。2以上にすると低レベル歩行モデルを複数読み込むため、クラッシュしやすくなります。

### 1.6 Tier1 の増やし方

順番:

```text
quick_test
small
medium
long
overnight
```

判断基準:

- `quick_test` が通る: パイプラインOK。
- `small` が通る: GA再開・世代交代OK。
- `medium` が通る: 通常運用OK。
- `long` 以上: 長時間放置用。

クラッシュしたら1段階戻してください。

## 2. Tier2 PC

### 2.1 新しく始める

PowerShellでプロジェクトフォルダへ移動し、仮想環境を有効化します。

```powershell
cd C:\Users\tsumu\Downloads\RoboQuest2026-main\RoboQuest2026-main
.\.venv\Scripts\Activate.ps1
```

まず動作確認:

```powershell
python -m scripts.benchmark_tier2 --config configs\tier2_benchmark.yaml
```

次にGA:

```powershell
python -m scripts.train_tier2_ga --config configs\tier2_compat_ga.yaml
```

新しいrunは自動で作成されます。

```text
runs\tier2\YYYYMMDD_HHMMSS_compat_ga\
```

### 2.2 再開する

途中でPowerShellを閉じた、PCを再起動した、または学習を止めた場合:

```powershell
cd C:\Users\tsumu\Downloads\RoboQuest2026-main\RoboQuest2026-main
.\.venv\Scripts\Activate.ps1
python -m scripts.train_tier2_ga --config configs\tier2_compat_ga.yaml --resume runs\tier2\YYYYMMDD_HHMMSS_compat_ga
```

最新runを確認:

```powershell
type runs\tier2\latest_run.txt
```

### 2.3 進捗確認

`train_tier2_ga.py` の実行中は、PowerShell上に候補単位の進捗バーが表示されます。

```text
Gen 2/5 Ind 3/8: ... candidate=abcd1234 best=42.5
```

進捗はrunフォルダにも保存されます。

```text
runs\tier2\{run}\progress_state.json
```

別のPowerShellで確認する場合:

```powershell
type runs\tier2\latest_run.txt
type runs\tier2\{run}\progress_state.json
```

ベスト候補の詳細:

```powershell
type runs\tier2\{run}\best\best_record.json
```

ランキングを作り直す場合:

```powershell
python -m scripts.rank_candidates --run runs\tier2\latest
```

### 2.4 評価

軽い評価:

```powershell
python -m scripts.evaluate_tier2 --run runs\tier2\latest --candidate best --seeds coarse
```

最終寄り評価:

```powershell
python -m scripts.evaluate_tier2 --run runs\tier2\latest --candidate best --seeds final
```

### 2.5 動画確認

```powershell
python -m scripts.record_tier2_video --run runs\tier2\latest --candidate best --seeds "100"
```

複数seed:

```powershell
python -m scripts.record_tier2_video --run runs\tier2\latest --candidate best --seeds "100 101 102"
```

出力先:

```text
runs\tier2\{run}\videos\
```

### 2.6 提出候補作成

```powershell
python -m scripts.package_submission --run runs\tier2\latest --candidate best
```

出力先:

```text
runs\tier2\{run}\submission_package\
```

以下6ファイルが揃っていることを確認します。

```text
walk_model.zip
walk_model_vecnorm.pkl
walk_params.json
flee_model.zip
flee_model_vecnorm.pkl
flee_params.json
```

### 2.7 Tier2 の増やし方

`configs/tier2_compat_ga.yaml` を段階的に変えます。

#### Stage 0: 動作確認

```json
"population_size": 2,
"generations": 1,
"default_training": {
  "timesteps": 2000,
  "n_envs": 1,
  "device": "cpu",
  "checkpoint_steps": 1000
},
"flee_timesteps": ["choice", [2000]],
"flee_num_envs": ["choice", [1]]
```

#### Stage 1: 軽い学習

```json
"population_size": 2,
"generations": 1,
"default_training": {
  "timesteps": 10000,
  "n_envs": 1,
  "device": "cpu",
  "checkpoint_steps": 5000
},
"flee_timesteps": ["choice", [10000]],
"flee_num_envs": ["choice", [1]]
```

#### Stage 2: 小規模探索

```json
"population_size": 4,
"generations": 2,
"default_training": {
  "timesteps": 25000,
  "n_envs": 1,
  "device": "cpu",
  "checkpoint_steps": 10000
},
"flee_timesteps": ["choice", [25000]],
"flee_num_envs": ["choice", [1]]
```

#### Stage 3: 標準探索

```json
"population_size": 6,
"generations": 3,
"default_training": {
  "timesteps": 50000,
  "n_envs": 1,
  "device": "cpu",
  "checkpoint_steps": 25000
},
"flee_timesteps": ["choice", [50000]],
"flee_num_envs": ["choice", [1]]
```

#### Stage 4: 長時間探索

```json
"population_size": 8,
"generations": 5,
"default_training": {
  "timesteps": 75000,
  "n_envs": 1,
  "device": "cpu",
  "checkpoint_steps": 25000
},
"flee_timesteps": ["choice", [50000, 75000, 100000]],
"flee_num_envs": ["choice", [1]]
```

#### Stage 5: PC本命

ベンチマークで安定していれば `n_envs=2` を試します。

```json
"population_size": 8,
"generations": 8,
"default_training": {
  "timesteps": 100000,
  "n_envs": 2,
  "device": "cpu",
  "checkpoint_steps": 25000
},
"flee_timesteps": ["choice", [75000, 100000, 150000]],
"flee_num_envs": ["choice", [1, 2]]
```

`n_envs=4` は速くなる可能性がありますが、メモリと安定性の面で後回しにしてください。

## 3. 共通の判断基準

見る順番:

1. `escaped_count`
2. `fell_count`
3. `mean_survived_seconds`
4. `worst_survived_seconds`
5. `score`
6. `mean_distance`

距離だけ高いモデルは壁際で詰む可能性があります。逃げ切り数と転倒数を優先してください。

## 4. 失敗時の戻し方

- クラッシュした: 1段階軽くする。
- 動画が一瞬で終わる: 転倒の可能性が高い。学習量を少し増やして再評価する。
- タグされ続ける: `flee_timesteps` と世代数を増やす。
- 転倒が多い: `vy` / `omega` に頼りすぎている可能性がある。Tier2では探索結果の報酬係数を確認する。

