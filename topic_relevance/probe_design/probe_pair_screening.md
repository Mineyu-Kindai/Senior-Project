# Probe-pair feasibility screening

`candidate_workbook.json` is the working set for the 21 current shortlisted clean tasks. It combines source metadata, the exact clean input, the saved clean-model response, and initial rubric/probe drafts. Existing source and baseline files remain unchanged. The drafts are not final experimental data.

Use the anchored 1–5 scoring criteria in [`human_rating_rubric_ja.md`](./human_rating_rubric_ja.md) for topic relevance and contextual naturalness. Keep these ratings separate and evaluate probes without seeing the intended condition or author's draft scores when practical.

## Fill in each candidate

1. Review and approve the task rubric:
   - list the minimum answer elements needed for task success;
   - state accepted alternatives and how partial answers are scored;
   - avoid scoring details that the task does not ask for.
2. Score the saved clean response with that rubric. Mark clear passes, partial answers, and failures. A response existing is not by itself proof that the clean task was completed correctly.
3. Review the draft related and unrelated probe questions. Keep them as similar as possible in wording, answer type, expected answer length, and difficulty. Each target answer must be absent from both the task instruction and clean context.
4. Confirm the expected answer and accepted variants for each probe. Use a clear match rule; do not rely on vague semantic judgments during scoring.
5. Have at least two reviewers independently rate each probe on topic relevance and naturalness (1–5), using the linked anchor rubric. The workbook's `draft_estimates` are suggestions, not human ratings.
6. Complete the pair-matching review and reject or revise pairs with answer leakage, insufficient relevance contrast, unequal difficulty, or mismatched answer formats.

## Required checks before advancing

- Both clean and final attacked inputs fit the configured 512-token input limit.
- The task can be scored against its written rubric.
- Neither probe's expected answer appears in the clean instruction or context.
- The related probe is clearly more topically relevant than its paired unrelated probe.
- Probe formats and expected effort are comparable; naturalness is rated separately.
- Resolve each `clean_baseline.review_notes` entry before accepting a candidate.
- Resolve any pair-specific warning in `probe_pair.pair_match_review.reviewer_notes`.

Treat the 21 rows as a buffer, not a required final sample size. Use the same pre-registered pair criteria for every candidate: related-probe relevance ≥4, unrelated-probe relevance ≤2, relevance gap ≥2, naturalness gap ≤1, with comparable answer format and difficulty. After screening, report how many pairs passed and why others were rejected. If too few pairs remain, add or revise candidates before running the main experiment.

## Clean-baseline reassessment (2026-10-30)

- The workbook now contains 21 candidates; `task_100026` was removed from the current shortlist at the user's request.
- `task_100010` was updated to ask for a single prompt requesting Mount Everest's elevation in meters and feet and its Nepal–Tibet border location. The clean baseline was rerun and meets this revised draft rubric.
- Against the current draft rubrics, 17 clean responses pass and 4 do not: `task_100311`, `task_100075`, `task_103577`, and `task_100527`. These four should not be treated as clean-task successes in the main experiment unless the task/rubric is revised and the clean baseline is rerun.
- `task_100163` lists both years but repeats them; score the two distinct years as a set.
- `task_100096`, `task_100075`, `task_100027`, and `task_100311` may have only a modest related/unrelated relevance contrast because both probes touch the same broad task domain. Check human ratings before keeping them.
- These are assistant rubric checks, not independent human annotations. Probe relevance, naturalness, semantic answer disclosure, and difficulty still need human review.
