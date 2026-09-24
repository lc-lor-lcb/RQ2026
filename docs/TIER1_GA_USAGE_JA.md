# Tier1 Colab GA 使用方法

`notebooks/tier1_colab_ga_ja.ipynb` を Google Colab で開いて、上から順に実行します。

## 目的

- 環境本体や競技ルールは変更しない。
- PCで作った強い歩行ベースを使う。
- 逃げ報酬とPPO設定を遺伝的アルゴリズムで探索する。
- Google Drive に途中保存し、切断後も再開できるようにする。

## 現在の推奨構成

Tier2側で作った安定歩行モデルをTier1でも使います。

ローカルに作成済みのzip:

```text
C:\Users\tsumu\Downloads\RoboQuest2026-main\RoboQuest2026-main\artifacts\tier1_walk_speed_race_001_bundle.zip
```

このzipをGoogle Driveの次の場所へアップロードしてください。

```text
MyDrive/RoboQuest2026_GA/walk_bases/tier1_walk_speed_race_001_bundle.zip
```

ノートブック側の初期値もこの場所を読む設定になっています。

```python
walk_source_mode = 'drive_zip'
walk_bundle_zip = '/content/drive/MyDrive/RoboQuest2026_GA/walk_bases/tier1_walk_speed_race_001_bundle.zip'
```

緊急時や比較用にGitHub同梱モデルへ戻す場合は、Colab上で次に変更します。

```python
walk_source_mode = 'repo_pretrained'
```

ただし、今回の検証では同梱 `smooth_walk` はTag環境で不安定だったため、通常は `drive_zip` を使ってください。

## 保存先

```text
MyDrive/RoboQuest2026_GA/tier1/{run_name}/
```

## 出力

```text
state.json
results.csv
results.jsonl
best_candidates.json
individuals/
best_submission/
```

## 使い方

1. `artifacts/tier1_walk_speed_race_001_bundle.zip` をGoogle Driveへアップロードする。
2. 更新済みの `notebooks/tier1_colab_ga_ja.ipynb` をColabで開く。
3. セットアップセルを実行する。
   - Colab上のrepoに `arena_posctrl.xml` を追加します。
   - Tag環境のXMLを位置制御版へ切り替えます。
   - 高レベル命令を `safe_forward` にします。
4. 実験設定セルで `run_name` を変える。
5. 歩行モデル単体確認セルを実行する。
   - `stand` が20秒以上、`forward_1.0`, `forward_1.4` が3秒以上ならOK。
   - 高速直進を長く続けると5mアリーナの壁に当たるため、`forward_1.0/1.4` は短時間チェックで十分です。
   - race歩行では低速 `forward_0.4` が逆走気味になることがあるため、Tier1逃げ学習も高速域 `vx=0.75〜1.55` を使う設定にしています。
6. 最初は `official_smoke` か `official_full` でGA実行セルを回す。
7. `best_submission/` の動画を確認する。
8. 問題なければ `search_full` または `overnight` へ増やす。

## 推奨プロファイル

最初:

```python
tier1_profile = 'official_smoke'
run_name = 'tier1_seed2_smoke_001'
```

歩行チェックと逃げ学習が正常なら:

```python
tier1_profile = 'official_full'
run_name = 'tier1_seed2_full_001'
```

Colabを複数アカウントで回すなら、`run_name` と `seed` を変えて並列にします。

例:

```text
tier1_seed2_full_a_seed0
tier1_seed2_full_b_seed10
tier1_seed2_full_c_seed20
```

## 注意

このTier1ノートブックは「提出物のファイル形式」は公式互換を保ちつつ、学習時の歩行土台を今回作った強いモデルへ差し替えます。
Colabランタイムは毎回初期化されるため、セットアップセルの自動パッチを必ず実行してください。

Colab無料枠はGPUや継続時間が保証されないため、Tier1は保険・広範囲探索として使い、本命はPCのTier2側で深掘りしてください。
