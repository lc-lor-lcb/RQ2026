# RoboQuest2026

## 学習ノートブック

- [クイックスタート編](https://colab.research.google.com/github/SingularityBattleQuest/RoboQuest2026/blob/main/notebooks/quickstart_ja.ipynb)：上から実行して動かす。主要なパラメーターをスライダーで調整できます。
- [解説・実験編](https://colab.research.google.com/github/SingularityBattleQuest/RoboQuest2026/blob/main/notebooks/guide_and_experiments_ja.ipynb)：同じ学習フローを、観測・行動・報酬・PPOの解説と実験例付きで学びます。

Google Colabで実行してください。両教材とも「歩行 → 逃げ」の階層型PPOを使い、モデル・正規化データ・学習設定の保存形式は共通です。実験ごとに保存先を分けて比較できます。

ブラウザビューアーは歩行モデルの手動操作に対応しています。逃げAIはノートブックの数値評価セルで確認できます。

### ローカルでの歩行学習とColabへの持ち込み

環境の修正内容、学習の再現手順、評価結果は [ローカル歩行学習](docs/walk_training/README.md) にまとめています。現在の保存モデルは停止・前進の確認用です。後退・横移動・旋回を含む全方向の安定歩行は未達です。

`retrain_walk=False` のまま実行すると、同梱した停止・前進用の検証済みモデルを読み込みます。ZIPのアップロードは不要です。
