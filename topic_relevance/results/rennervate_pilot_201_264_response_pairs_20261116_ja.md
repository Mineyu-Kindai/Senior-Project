# pilot 201–264：防御あり／なしの回答とマスク

## データ

[各入力の回答・検知・マスクデータ（CSV）](../data/experiments/rennervate_pilot_201_264_response_pairs_v1.csv)
に、64ブロック × 4条件 × 防御あり／なしの**512行**を収録した。
同じ`example_id`の2行が同一入力に対する比較である。

| 防御 | 行数 | プローブトークン検知率 | 正常文字マスク率 |
|---|---:|---:|---:|
| なし | 256 | 実施しないため対象外 | 0/42,548 = 0.0% |
| あり | 256 | 1,230/3,526 = 34.9% | 8,385/42,548 = 19.7% |

上表の率は各入力の率の平均ではなく、全入力の分子・分母を合算した率。
Probeと他領域をまたぐ境界トークンは、Probeトークンの分母から除外している。
正常文字マスク率は、結合後のマスク範囲と正常本文の文字範囲の重なりで算出した。
防御なしの検知率は「0%」ではなく、検知を適用していないため対象外である。

## CSVの見方

- `example_id`、`block_id`、`context_id`、`probe_id`：元データ上の入力を識別する。
- `defense`：`none`は防御なし、`rennervate`は新検知条件でマスクした入力。
- `user_instruction`：両条件で共通のタスク指示。
- `external_context`：モデルに渡した外部文脈。防御ありではマスク後の文字列。
- `model_response`：**モデルの回答全文**。プローブの回答とタスク回答を機械的に分離していない。
  防御ありの行では、マスク後の入力に対する回答全文にあたる。
- `probe_tokens_flagged`、`probe_tokens_total`、`probe_token_detection_rate`：
  Probe領域の検知数、総トークン数、比率。防御なしでは空欄。
- `clean_characters_masked`、`clean_characters_total`、`clean_character_mask_rate`：
  正常本文のマスク文字数、総文字数、比率。率は0〜1の値で記録。
- `input_token_count`、`generated_token_count`、`output_limit_reached`、`eos_terminated`：
  入力・出力長と生成終了状態。
- `intended_relation`はデータ作成時の意図であり、人手確認済みの関連度判定ではない。

攻撃の成功・失敗およびタスクの正答・誤答・採点不能は、今回のCSVには付与していない。
回答全文にタスク回答やプローブへの回答が含まれるかも自動判定していない。

## 実験条件と既存データの利用

- 引用された候補データは、保存済み実験で使用した64ブロックとSHA-256が一致した。
- 利用者の指定により、**候補データの元指示から3つの制約文を除いた、前回と同じ指示**を使用した。
  候補データ中の元指示そのままの条件ではない。
- 新検知条件は、固定済み分類器の2クラスlogitsを幅5・replicate paddingの平均フィルタで
  平滑化してトークンを判定する方式。陽性トークンは入力全体の判定結果とは独立してマスクする。
  既存のマスク範囲結合と置換マーカーは変更していない。
- Qwen2.5-3B-Instruct、4-bit NF4、`do_sample=False`、入力上限512、
  生成上限384。対象の512応答で出力上限到達は0件。
- 今回は承認された前回条件の**保存済み検知・応答を再利用**し、新しいモデル推論・応答生成は行っていない。
  保存済みスコアからマスクを再計算し、各入力のマスク範囲、回答、入力ハッシュ、
  チェックポイントと元データのハッシュを照合した。
- 対象は同じpilotの探索データであり、未使用の独立確認集合ではない。

## 元データ

- [新検知条件の検知・防御あり応答](../data/experiments/rennervate_detector_pilot_201_264_v1.jsonl)
- [防御なし応答を含む保存済み実験](../data/experiments/pilot_201_264_saved_mask_interventions_without_task_constraints_v1.json)
- [集計CSVの作成スクリプト](../scripts/export_pilot_response_pairs_v1.py)
