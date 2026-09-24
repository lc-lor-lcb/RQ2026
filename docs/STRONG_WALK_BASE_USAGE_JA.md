# Tier1/Tier2 共通 強力歩行ベースの作り方

目的は、逃げAIを学習する前に、Tier1/Tier2の両方で使える安定した `walk_model` を作ることです。

今回の症状では `stand` でも 0.2 秒で倒れているため、逃げGAより先に歩行モデルを作り直します。

## 1. 前提

PowerShellでリポジトリに移動します。

```powershell
cd C:\Users\tsumu\Downloads\RoboQuest2026-main\RoboQuest2026-main
.\.venv\Scripts\Activate.ps1
```

すでに以下が完了していれば再インストールは不要です。

```powershell
python -m pip install -r requirements.txt -c requirements-training.txt
```

## 2. まず短い動作確認

最初は必ず `smoke` で、コードが最後まで動くかだけ確認します。

```powershell
python -m scripts.train_strong_walk_base --profile smoke --run-name walk_smoke_001 --record-videos
```

出力先:

```text
runs\walk_base\walk_smoke_001
```

重要ファイル:

```text
runs\walk_base\walk_smoke_001\ranking.json
runs\walk_base\walk_smoke_001\summary.csv
runs\walk_base\walk_smoke_001\best\walk_model.zip
runs\walk_base\walk_smoke_001\best\walk_model_vecnorm.pkl
runs\walk_base\walk_smoke_001\best\walk_params.json
```

`ranking.json` の `passed` が `true` なら歩行土台として合格です。
`false` でも `smoke` は短い確認なので、動画と `fall_count` を見て判断します。

## 3. 本命の標準学習

`smoke` が動いたら標準学習に進みます。

```powershell
python -m scripts.train_strong_walk_base --profile standard --run-name walk_standard_001 --record-videos
```

標準設定は複数seedで歩行モデルを作り、次を確認します。

- `Go2WalkEnv` で停止・前進 0.25・前進 0.4
- `Go2TagHierarchicalEnv` 内で停止・前進 0.25・前進 0.4

後者に通ることが重要です。逃げAIは `Go2TagHierarchicalEnv` の中で歩行モデルを使うため、単体歩行だけ通っても不十分です。

## 4. 強めの長時間学習

標準学習で合格候補が出たら、勝率狙いの強め設定を回します。

```powershell
python -m scripts.train_strong_walk_base --profile strong --run-name walk_strong_001 --record-videos
```

時間がかかります。PCを閉じず、スリープしない設定で実行してください。

## 5. 段階的な増やし方

おすすめ順:

```text
smoke
standard
strong
strong を seed 追加で再実行
```

seedを増やす場合:

```powershell
python -m scripts.train_strong_walk_base --profile strong --run-name walk_strong_002 --seeds 3,4,5 --record-videos
```

さらに評価seedも増やす場合:

```powershell
python -m scripts.train_strong_walk_base --profile strong --run-name walk_strong_003 --seeds 6,7,8 --eval-seeds 100,101,102,200,201,202,300,301 --record-videos
```

## 6. 合格判定

まず見る場所:

```text
runs\walk_base\<run_name>\summary.csv
```

見る項目:

- `passed` が `True`
- `walk_falls` が `0`
- `tag_falls` が `0`
- `tagged_count` が `0`
- `walk_mean_seconds` が設定秒数に近い
- `tag_mean_seconds` が設定秒数に近い
- `walk_min_height` が 0.15 より十分高い
- `tag_min_height` が 0.15 より十分高い

最重要は `tag_falls = 0` です。Tier1/Tier2の逃げ学習ではTag環境内で安定する必要があります。
歩行ベース確認では鬼を固定コマンドの進路から外しているため、`tagged_count > 0` の場合も採用前に原因確認してください。

## 7. 動画確認

`--record-videos` を付けた場合、動画はここに出ます。

```text
runs\walk_base\<run_name>\videos
```

動画では次を見ます。

- 立てているか
- 前進時に膝や胴体を床に擦っていないか
- 小刻みすぎる足踏みになっていないか
- 数秒後に崩れていないか

数値が合格でも、動画で明らかに不自然なら採用しない方がいいです。

## 8. Tier1/Tier2へ使うファイル

採用するのはこの3つです。

```text
runs\walk_base\<run_name>\best\walk_model.zip
runs\walk_base\<run_name>\best\walk_model_vecnorm.pkl
runs\walk_base\<run_name>\best\walk_params.json
```

Tier1 Colabへ持っていく場合は、この3ファイルをDriveまたはColabにアップロードし、逃げ学習の候補フォルダに入れます。

Tier2 PCで使う場合は、逃げ学習用candidateフォルダの既存 `walk_*` をこの3ファイルで置き換えます。

## 9. 再開・再評価

同じrunを再評価したい場合:

```powershell
python -m scripts.train_strong_walk_base --profile standard --run-name walk_standard_001 --skip-train --record-videos
```

注意: 既存候補を上書き学習しません。新しい学習は別の `--run-name` を付けてください。

### Tag環境だけ落ちた場合

`walk_falls = 0` なのに `tag_falls > 0` の場合、歩行モデルそのものより、Tag環境への接続が怪しいです。
このリポジトリでは歩行学習が `go2_posctrl.xml` の位置制御ロボットを使うため、Tag環境も `arena_posctrl.xml` を使うようにしています。

修正後は、まず再学習せずに再評価します。

```powershell
python -m scripts.train_strong_walk_base --profile standard --run-name walk_standard_001 --skip-train --record-videos
```

これで `tag_falls = 0` かつ `tagged_count = 0` になれば、既存の歩行候補をそのまま採用候補にできます。

## 10. 判断方針

勝率優先では、逃げAIより先に歩行を合格させます。

合格ライン:

```text
stand が落ちない
forward_0.25 が落ちない
forward_0.4 が落ちない
Tag環境内でも同じく落ちない
動画で見て破綻していない
```

これを満たすまでは、逃げGAやTier2改造へ進まない方がいいです。
