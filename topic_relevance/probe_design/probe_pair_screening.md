# Probe-pair feasibility screening

`candidate_workbook.json` is the working set for the 22 shortlisted clean tasks. It combines source metadata, the exact clean input, and the saved clean-model response. Existing source and baseline files remain unchanged.

## Fill in each candidate

1. Write a task rubric before drafting probes:
   - list the minimum answer elements needed for task success;
   - state accepted alternatives and how partial answers are scored;
   - avoid scoring details that the task does not ask for.
2. Draft one topic-related and one unrelated probe. Keep them as similar as possible in wording, answer type, expected answer length, and difficulty. Each target answer must be absent from both the task instruction and clean context.
3. Record an exact expected answer or witness and a matching rule for each probe. Do not rely on vague semantic judgments during scoring.
4. Rate each probe independently on topic relevance and naturalness (1–5). Relevance is to the clean context; naturalness is how naturally the probe fits that context. Do not use naturalness as a substitute for relevance.
5. Complete the pair-matching review and reject or revise pairs with answer leakage, unequal difficulty, or mismatched answer formats.
6. Manually verify the clean response against the task rubric. A response existing is not by itself proof that the clean task was completed correctly.

## Required checks before advancing

- Both clean and final attacked inputs fit the configured 512-token input limit.
- The task can be scored against its written rubric.
- Neither probe's expected answer appears in the clean instruction or context.
- The related probe is clearly more topically relevant than its paired unrelated probe.
- Probe formats and expected effort are comparable; naturalness is rated separately.
- Resolve any `baseline_review` notes in the workbook before accepting a candidate.

Treat the 22 rows as a buffer, not a required final sample size. After screening, report how many pairs passed and why others were rejected. If too few pairs remain, add or revise candidates before running the main experiment.
