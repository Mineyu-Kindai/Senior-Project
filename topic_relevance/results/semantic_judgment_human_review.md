# 人手による意味的攻撃成功判定

## 対象と判定

- 対象: `fake_completion`形式の21タスク×関連／無関連プローブ、計42ケース
- 判定対象: 各ケースの防御前・防御後応答、計84応答
- 判定: 期待回答との文字列一致ではなく、質問に対する意味的な回答として成功／失敗を人手で分類
- 元のLLM-as-a-judgeで`uncertain`だった7応答は、今回すべて人手判定で確定した
- A/Bの条件対応は別ファイルの研究者用キーで復元。添付されたLLM判定結果とは別に保存

## 条件別の意味的成功率

| 条件 | 防御前 | 防御後 |
|---|---:|---:|
| 関連プローブ | 15/21 (71.4%) | 2/21 (9.5%) |
| 無関係プローブ | 15/21 (71.4%) | 3/21 (14.3%) |

防御後成功は関連条件2件、無関連条件3件で、差は1件（4.8ポイント）。ペア単位では、関連のみ成功1組、無関連のみ成功2組、両方成功1組、両方失敗17組。

クリーンタスク基準合格の17タスクに限ると、防御後成功は関連1/17 (5.9%)、無関連2/17 (11.8%)。

## 解釈

防御後の結果は、関連プローブほど成功しやすいというH1bを支持していない。数値上は反対方向だが、差は1件のみであり、関連度効果がない／逆であると結論するには不十分。防御前の成功率は両条件で同じだった。

これは21ペア・各条件1回の探索的評価である。プローブの関連度・自然さは依然として作者の推定であり、人手評価済みではない。攻撃成功の意味判定が人手で行われたことと、プローブ評価が人手で行われたことは区別する。

## 他指標との区別

この再判定は攻撃成功の結果だけを更新する。固定Attention検知器のRecallには影響しない。検知Recallは[全21件の攻撃形式パイロット結果](./attack_style_pilot_21tasks.md)を参照する。

## ファイル

- 人手判定: [semantic_judgments_human_review.json](./semantic_judgments_human_review.json)
- 判定ケース: [semantic_judgment_cases_blinded.json](../data/semantic_judgment_cases_blinded.json)
- A/B対応キー: [semantic_judgment_researcher_key.json](../data/semantic_judgment_researcher_key.json)
- 判定基準: [semantic_judgment_instructions_ja.md](../protocols/semantic_judgment_instructions_ja.md)
