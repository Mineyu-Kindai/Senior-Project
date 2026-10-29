# Probe-pair feasibility screening

`candidate_workbook.json` is the working set for the 22 shortlisted clean tasks. It combines source metadata, the exact clean input, the saved clean-model response, and initial rubric/probe drafts. Existing source and baseline files remain unchanged. The drafts are not final experimental data.

## Fill in each candidate

1. Review and approve the task rubric:
   - list the minimum answer elements needed for task success;
   - state accepted alternatives and how partial answers are scored;
   - avoid scoring details that the task does not ask for.
2. Score the saved clean response with that rubric. Mark clear passes, partial answers, and failures. A response existing is not by itself proof that the clean task was completed correctly.
3. Review the draft related and unrelated probe questions. Keep them as similar as possible in wording, answer type, expected answer length, and difficulty. Each target answer must be absent from both the task instruction and clean context.
4. Confirm the expected answer and accepted variants for each probe. Use a clear match rule; do not rely on vague semantic judgments during scoring.
5. Have reviewers rate each probe independently on topic relevance and naturalness (1–5). Relevance is to the clean context; naturalness is how naturally the probe fits that context. Do not use naturalness as a substitute for relevance. The workbook's `draft_estimates` are suggestions, not human ratings.
6. Complete the pair-matching review and reject or revise pairs with answer leakage, insufficient relevance contrast, unequal difficulty, or mismatched answer formats.

## Required checks before advancing

- Both clean and final attacked inputs fit the configured 512-token input limit.
- The task can be scored against its written rubric.
- Neither probe's expected answer appears in the clean instruction or context.
- The related probe is clearly more topically relevant than its paired unrelated probe.
- Probe formats and expected effort are comparable; naturalness is rated separately.
- Resolve each `clean_baseline.review_notes` entry before accepting a candidate.
- Resolve any pair-specific warning in `probe_pair.pair_match_review.reviewer_notes`.

Treat the 22 rows as a buffer, not a required final sample size. After screening, report how many pairs passed and why others were rejected. If too few pairs remain, add or revise candidates before running the main experiment.

## Known issues to review first

- `task_100026` and `task_100527` clean responses do not meet the drafted extraction rubrics.
- `task_100163` returns both years but repeats each one; score unique years as a set.
- `task_100010` may only ask for the source facts rather than incorporating them into the generated prompt; decide how the task instruction should be interpreted before scoring.
- `task_100096`, `task_100075`, `task_100027`, `task_100311`, and `task_100026` may have only a modest related/unrelated relevance contrast because both probes touch the same broad task domain. Check ratings before keeping them.
