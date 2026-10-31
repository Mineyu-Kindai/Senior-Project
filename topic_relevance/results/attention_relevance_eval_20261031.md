# Attention detector evaluation: related vs. unrelated probes

## Run setup

- Date: 2026-10-31
- Source: all 54 candidate task pairs in `../workbooks/candidate_workbook.json`
- Conditions: 54 tasks × related/unrelated probe = 108 attacked inputs, one deterministic run per condition
- Model: Qwen/Qwen2.5-3B-Instruct, 4-bit NF4, 512 input-token limit, 384 generated-token limit
- Injection: the protocol's shared `fake_completion` wrapper plus the exact probe text, appended to the clean context
- Detector: fixed `models/attention_detector_fixed_20261015.pt`, threshold 0.5; no retraining or threshold changes
- Outputs: raw responses, redacted contexts, per-example metrics, and aggregate metrics are in [`attention_relevance_eval_20261031.json`](./attention_relevance_eval_20261031.json).

## Token-level detection

| Probe condition | Full injected span recall | Probe-text-only recall | Shared-wrapper recall | False-positive rate on clean context |
|---|---:|---:|---:|---:|
| Related | 62.96% (933/1,482) | 24.52% (178/726) | 99.87% (755/756) | 4.40% |
| Unrelated | 63.33% (867/1,369) | 18.11% (111/613) | 100.00% (756/756) | 4.14% |

The full-span recall is largely driven by near-perfect detection of the shared wrapper. The probe-only recall is substantially lower. Treat probe-only recall as the more informative measure for whether probe content itself was detected.

## Answer-match rates before and after defense

These are **literal, case-insensitive expected-answer string matches**, not semantic success judgments.

| Subset | Condition | n | Baseline matches | Defended matches | Absolute rate reduction | Baseline matches blocked |
|---|---|---:|---:|---:|---:|---:|
| All candidates | Related | 54 | 31 (57.4%) | 7 (13.0%) | 44.4 pp | 26/31 (83.9%) |
| All candidates | Unrelated | 54 | 33 (61.1%) | 5 (9.3%) | 51.9 pp | 30/33 (90.9%) |
| Human-rating criteria pass | Related | 21 | 13 (61.9%) | 1 (4.8%) | 57.1 pp | 12/13 (92.3%) |
| Human-rating criteria pass | Unrelated | 21 | 10 (47.6%) | 1 (4.8%) | 42.9 pp | 10/10 (100%) |
| Clean-baseline and human-rating criteria pass | Related | 18 | 10 (55.6%) | 1 (5.6%) | 50.0 pp | 9/10 (90.0%) |
| Clean-baseline and human-rating criteria pass | Unrelated | 18 | 9 (50.0%) | 1 (5.6%) | 44.4 pp | 9/9 (100%) |

The 54-candidate result includes every workbook row. The more restricted 18-pair subset satisfies both the current human probe-pair rating cutoffs and the draft clean-baseline pass status. None of these one-run comparisons is a confirmatory statistical test.

## Interpretation and limitations

- The high full-injection recall should not be read as strong detection of the probe itself: almost all recall comes from the fixed wrapper, while probe-only recall is 24.5% for related and 18.1% for unrelated probes across all candidates.
- The detector's full-span recall differs by only 0.37 percentage points between conditions. Probe-only recall is 6.41 points higher for related probes in the full set, but the screened 18-pair subset is small and the run is exploratory.
- Answer-match rates fell after defense in both conditions. This is a string-match result only; it does not implement the agreed semantic rule, and a string match may be a negation, quotation, or otherwise not a meaningful answer.
- Three related probes are Japanese while the task contexts and other probes are predominantly English. The model's responses to these probes did not contain the exact expected answer in either the baseline or defended response. Do not mix these language-mismatched cases into language-general claims.
- The human-rated and clean-baseline-qualified subsets are useful sensitivity checks, but the workbook's task rubrics remain drafts and some pair-level checks (including semantic answer disclosure and difficulty comparability) are pending.
- One deterministic response per condition gives no estimate of run-to-run variability. The sample and model are not sufficient to support a broad claim about prompt-injection defenses.

## Next required step

Human-adjudicate the baseline and defended responses in the JSON using the agreed semantic success rule. Keep those labels separate from the automatic `baseline_answer_match` and `defended_answer_match` fields. Only then report semantic defense success rates.
