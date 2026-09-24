# RoboQuest2026 実験方針

この追加コードは、勝利確率を上げるために次の役割分担を想定しています。

## 方針

- Tier1: Google Colab で動かす保険・広範囲探索。
- Tier2: PC/WSL2 で動かす本命探索。
- 最初は既存の `models/pretrained/smooth_walk` を固定し、逃げAIを重点的に探索する。
- 最終評価は標準互換環境で行う。
- 提出候補は既存ノートブック互換のファイル名を維持する。

## 評価優先順位

1. 60秒逃げ切り数
2. 転倒しないこと
3. 平均生存秒数
4. 最悪seedの生存秒数
5. 平均距離

平均距離だけが高いモデルは、壁際で詰む可能性があるため補助指標として扱います。

## 提出候補ファイル

`submission_package/` または `best_submission/` に以下が揃っていることを確認します。

```text
walk_model.zip
walk_model_vecnorm.pkl
walk_params.json
flee_model.zip
flee_model_vecnorm.pkl
flee_params.json
```

## 注意

大会レギュレーションが更新された場合は、提出形式と改造可能範囲を優先して見直してください。
