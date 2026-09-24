# ローカル学習とColabの再現

最新モデルは `models/teams/walk_verified_smooth/` です。低速の見本を覚えてからPPOで微調整する方式に更新しました。詳しくは末尾の「足踏みを減らす調整」を参照してください。

ローカル環境は `.venv`、数値計算ライブラリの固定版は `requirements-training.txt` にあります。
MacとColabではCPU・OSが異なるため、同じ設定・seedでも学習後の重みの完全一致は保証しません。

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m scripts.tune_walk --curriculum --steps 750000
.venv/bin/python -m scripts.tune_walk --eval-only
.venv/bin/python -m scripts.tune_walk --resume --steps 1000000
```

モデル、観測の正規化データ、設定、評価結果は `models/teams/local_walk/` に保存します。
`--steps` は追加で集めるデータ数です。PPOのロールアウト単位で切り上げられます。
継続学習は同じ環境・報酬設定のモデルを使います。異なる実験は `--folder` で分けます。
学習とノートブックは `scripts/notebook_workflow.py` の同じ処理を呼び出します。

## 修正した環境

- 物理刻み0.002秒×10回＝制御周期0.02秒（50Hz）。従来は説明と異なる100Hzだった。
- 世界座標から胴体座標への回転をMuJoCoの四元数変換に統一。
- 次の観測に、たった今適用した行動を入れる。行動変化の報酬は更新前の行動との差を使う。
- 足の滑りは、脚の胴体側の位置ではなく足のgeom中心の速度を使う。
- 初期化時に位置制御の目標もホーム姿勢に設定。

旧環境で学習したモデルは観測数が同じでも条件が違うため、修正後の学習結果として扱いません。

## 判定

報酬だけでは判定しません。停止・前進・後退・左右移動・左右旋回をそれぞれ3つのseedで20秒ずつ評価します。
転倒の有無、指示に対する平均速度の誤差、速度のRMSE、胴体の傾きを `evaluation.json` に保存します。
最初の1秒は姿勢の落ち着きを待つ時間として速度集計から除外します。転倒はこの時間内も不合格です。
合格基準は学習結果を見る前に `scripts/tune_walk.py` に定義しています。

## Colabへ同じコード・モデルを渡す

```sh
.venv/bin/python -m scripts.package_walk_colab models/teams/local_walk
```

出力された `artifacts/walk_colab_bundle.zip` をColab左側のファイル欄にアップロードしてから、
`quickstart_ja.ipynb` または `guide_and_experiments_ja.ipynb` のセットアップを実行します。
ZIPがある場合は、GitHubの公開版よりもZIP内のソースを優先します。
モデルは `/content/RoboQuest2026/models/teams/bundled_walk/` に展開されます。
コードとモデルのSHA-256は `bundle_manifest.json` に保存されます。
公開リポジトリへのpushやColab上の実行は、このローカル検証とは別です。

## 今回の結果（2026-09-18）

採用モデル: `models/teams/walk_verified_forward/`。通常のPPOで学習したモデルです。
動作例を模倣させる方法も比較しましたが、採用モデルには使っていません。

| 検証 | 結果 |
|---|---|
| 停止と前進、初期姿勢seed 100〜102、各20秒 | 6ケースとも基準内・転倒なし |
| 追加の初期姿勢seed 200〜209、停止と前進、各20秒 | 20ケースとも基準内・転倒なし |
| 前進指示0.4m/sに対する平均前進速度（追加確認） | 約0.37m/s |
| 後退・左右移動・左右旋回 | 未達。後退では転倒も発生 |

ここでの「基準内」は前述の速度・傾きの基準です。胴体の高さは前進時約0.20mで、
姿勢を高く保つ歩行や完全な直進を保証するものではありません。
初期姿勢のseedを変えた評価であり、複数の学習seed、段差、不整地、実機での評価ではありません。

全7方向の結果は `selected_evaluation.json`、追加確認は `heldout_evaluation.json`、
採用設定は `selected_params.json` にあります。全方向の合格を意味する `passed` は **false** のままです。

### 採用モデルの学習手順

1. 可動幅0.6rad、前進指示0.2〜0.5m/sで学習し、75万ステップの途中保存を採用。
2. 前進しながら旋回指示も加えて50万ステップ追加。
3. 前進指示0.4m/s・旋回0の条件で、学習率と探索のばらつきを下げて5万ステップ追加。
4. 同じ条件で20万ステップ追加。

PPOの収集単位への切り上げ後は累計1,520,048ステップです。
詳細は `selected_curriculum.json`。同じ処理は `train_walk_curriculum` と両方の教材で使います。
新規学習する場合:

```python
from scripts.notebook_workflow import train_walk_curriculum
train_walk_curriculum('models/teams/my_forward_run')
```

単発の条件比較・全方向の実験は `scripts.tune_walk` で行います。
最終保存が最良とは限らないため、モデルと正規化データを対にした途中保存も評価してください。

### 実施した確認

- 学習用XMLと表示用XMLに同じ行動を100回与え、関節・胴体の状態が一致。
- 座標変換、次の観測に入る行動、50Hz周期、保存・再開・階層制御の可動幅をテスト。
- ONNX出力をPPOと比較。途中モデルで最大差約3.6e-7。
- ローカルのブラウザでモデル読込、前進スライダー、描画を確認。
- Colab用ZIPの中身とSHA-256を確認。**Colabの実ランタイムでの学習・再生は未実行**。

`artifacts/walk_colab_bundle.zip` は採用モデル・正規化データ・設定・コード・ロボットXMLを同梱します。
GitHubにはpushしていないため、今回の変更をColabで使うときはZIP経由で読み込んでください。

## 胴体を高く保つ追加調整（2026-09-18）

この段階の採用モデルは `models/teams/walk_verified_upright/`。以前の `walk_verified_forward/` は比較用に保存しています。
胴体の目標高さを0.26 mにし、足先以外の床接触と膝の低さを減点しました。物理モデル・関節の制御方式・45次元の観測は変更していません。
短いエピソードで動き出しを練習した後、停止と前進の速度追従を調整しています。

| 停止・前進 各10初期姿勢、20秒 | 改善版 |
|---|---|
| 停止時の平均胴体高さ | 0.255 m（以前は約0.157 m） |
| 前進時の平均胴体高さ | 0.263 m（以前は約0.198 m） |
| 指示0.4 m/sに対する平均前進速度 | 0.384 m/s |
| 後ろ膝の最低高さ、開始直後を含む全ケース | 0.084 m |
| 足先以外の床接触・転倒 | 全20ケースでなし |

速度・平均高さなどは最初の1秒を除いた集計です。接地と膝の最低高さは `whole_episode_*` で最初の1秒も別に確認しています。
接地は50 Hzの制御時点で判定し、膝の高さは後ろ脚の膝関節中心から床までです。
停止時にも小さな移動・足踏みが残ります。後退・横移動・旋回、不整地、実機はこの合格範囲に含めません。
評価は `upright_heldout_evaluation.json`、設定は `upright_params.json`、段階ごとの履歴は `upright_curriculum.json`。

再学習は元の前進モデルから次の共通処理で行えます。

```sh
.venv/bin/python -m scripts.refine_walk_posture models/teams/walk_verified_forward models/teams/my_upright_run
```

前進モデルから約180万ステップを追加し、採用モデルは累計3,326,960ステップです。
教材2冊もこの追加処理を呼びます。`retrain_walk=False` ではZIP内の検証済みモデルを読み込みます。
更新した `artifacts/walk_colab_bundle.zip` に同じコード、モデル、正規化データ、評価と設定を同梱しています。
Colabの実ランタイムでの実行は未確認です。

## 足踏みを減らす調整（2026-09-18・現在の採用版）

採用モデルは `models/teams/walk_verified_smooth/`、比較対象は直前の `walk_verified_upright/` です。
以前のモデルは前進中だけでなく停止中も約5.5 Hzの細かい足踏みで体を支えていました。
関節の支持力を上げ、力の上限を設定し、1.5 Hzのゆっくりした対角脚の動きと停止・再発進を見本として学習させました。
その後、正規化の統計を固定し、小さい学習率でPPOの微調整をしています。

| 停止・前進の確認 | 以前 | 改善版 |
|---|---:|---:|
| 前進時の関節角の主な周期 | 約5.4 Hz | 約1.5 Hz |
| 前進時の関節速度RMS | 約7.1 rad/s | 約1.7 rad/s（約76%減） |
| 停止時の関節速度RMS | 約6.6 rad/s | 約0.0004 rad/s |
| 前進時の平均胴体高さ | 約26 cm | 約30 cm |
| 前進指示0.4 m/sに対する平均速度 | 約0.38 m/s | 約0.43 m/s |

周期は関節角のスペクトルのピークであり、着地回数ではありません。静止時には周期を報告しません。
停止と前進について初期姿勢seed 200〜209、各20秒で20ケースを確認し、速度・安定後の姿勢は全件基準内、開始直後を含めて足先以外の接地と転倒はありませんでした。
開始直後の後ろ膝の最低高さは約5.9 cmです。以前の「最初から6 cm以上」という補助基準には一部がわずかに届かず、`startup_passed` は **false** のまま残しています。
停止→前進→停止→前進の連続切り替えも、3初期姿勢・各区間20秒で確認しました（`smooth_transition_evaluation.json`）。
後退・横移動・旋回の追従は未達です。全方向評価の `passed` は **false** です。

### 学習方法と再現

以前のPPOモデルへ報酬の変更だけで継続する方法は、足踏みが残るか、前進しなくなったため採用しませんでした。
新しいモデルは教師あり学習＋PPOです。見本30,000ステップ、教師あり更新4,000回、PPO 12,288ステップ（要求10,000を収集単位で切り上げ）で学習しました。
見本の時計はデータ生成だけに使います。再生時は45次元の観測から12関節の行動を出すニューラルネットで動作します。
学習率は1e-6、入力と報酬の正規化統計は微調整中に固定しています。

```sh
.venv/bin/python -m scripts.bootstrap_smooth_walk models/teams/my_smooth_run
.venv/bin/python -m scripts.tune_walk --folder models/teams/my_smooth_run --eval-only
.venv/bin/python -m scripts.evaluate_walk_transitions models/teams/my_smooth_run
```

関節の比例ゲインは60、減衰は2、アクチュエータ力の上限は股関節・大腿23.7 Nm、膝45.43 Nmです。
これらをモデルの設定、ONNXの契約ファイル、ブラウザ用の実際のMuJoCoシーンへ保存します。
旧モデルは従来の既定値20・0.5のまま読み込めます。
生成したブラウザシーンのゲイン・力の上限と、学習用／表示用の100ステップの物理挙動の一致を確認しました。
ONNX出力とPPO出力の最大差は単体比較で約2.4e-7、実歩行の500ステップを使った正規化込みの比較で約5.4e-7でした。関連テスト17件も通過しています。

両方の教材は同じ `train_smooth_walk` を呼び、報酬とPPOパラメータの変更を渡します。
現在の `artifacts/walk_colab_bundle.zip` はこのモデルと一致するコードを同梱しています。
Colab実ランタイムでの実行は未確認です。

比較動画: `artifacts/walk_smooth_comparison.mp4`（左: 以前、右: 改善版）。
評価・設定・履歴は、このフォルダの `smooth_*.json` に保存しています。
