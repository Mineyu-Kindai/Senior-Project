import json
import tempfile
import unittest
from pathlib import Path

from senior_project.prompt import redact_spans
from topic_relevance.scripts.capture_threshold_scores_v1 import input_hash
from topic_relevance.scripts.run_threshold_generation_v1 import (
    build_jobs,
    read_progress,
    score_evaluation,
    select_blocks,
    summarize,
)
from topic_relevance.scripts.summarize_threshold_masks_v1 import THRESHOLDS, summarize_record


class ThresholdGenerationTests(unittest.TestCase):
    def setUp(self):
        source = {
            "user_input": "task",
            "external_context": "catdog",
            "clean_context": "cat",
            "gold_task_answer": {"value": "2"},
            "expected_answer": "witness",
            "accepted_variants": [],
            "sanitized_external_context": "cat" + redact_spans("dog", [(0, 3)]),
        }
        self.sources = {"a": source}
        self.records = [
            {
                "input_kind": "injected", "id": "a", "user_input": "task",
                "external_context": "catdog", "input_sha256": input_hash("task", "catdog"),
                "regions": [{"name": "clean_context", "start": 0, "end": 3}],
                "tokens": [
                    {"start": 0, "end": 3, "score": 0.2, "evaluation_region": "clean_context"},
                    {"start": 3, "end": 6, "score": 0.9, "evaluation_region": "probe"},
                ],
            },
            {
                "input_kind": "clean_only", "source_ids": ["a"], "user_input": "task",
                "external_context": "cat", "input_sha256": input_hash("task", "cat"),
                "regions": [{"name": "clean_context", "start": 0, "end": 3}],
                "tokens": [
                    {"start": 0, "end": 3, "score": 0.2, "evaluation_region": "clean_context"},
                ],
            },
        ]
        self.rows = [
            {
                "threshold": threshold,
                "inputs": [summarize_record(record, threshold) for record in self.records],
            }
            for threshold in THRESHOLDS
        ]

    def test_build_jobs_reuses_identical_inputs_without_merging_evaluations(self):
        jobs, evaluations = build_jobs(self.records, self.rows, self.sources)
        self.assertEqual(len(jobs), 3)
        self.assertEqual(len(evaluations), 18)
        self.assertEqual(
            next(e for e in evaluations if e["input_kind"] == "injected" and e["condition"] == "0.50")["generation_input_sha256"],
            next(e for e in evaluations if e["input_kind"] == "injected" and e["condition"] == "0.90")["generation_input_sha256"],
        )
        self.rows[0]["inputs"][0]["masked_spans"] = []
        with self.assertRaisesRegex(ValueError, "mask spans"):
            build_jobs(self.records, self.rows, self.sources)

    def test_scoring_retains_unscorable_and_target_match_as_distinct_outcomes(self):
        jobs, evaluations = build_jobs(self.records, self.rows, self.sources)
        scored = [
            score_evaluation(e, {"response": '{"value": 2} {"value": 2} witness'})
            for e in evaluations
        ]
        metrics = summarize(scored)
        self.assertEqual(metrics["injected"]["0.50"]["content_unscorable"], 1)
        self.assertEqual(metrics["injected"]["0.50"]["target_string_matches"], 1)
        self.assertEqual(metrics["injected"]["0.50"]["content_correct_and_target_absent"], 0)
        self.assertNotIn("target_string_matches", metrics["clean_only"]["0.50"])
        self.assertEqual(len(jobs), 3)

    def test_resume_rejects_mismatched_metadata_or_completed_run(self):
        header = {"type": "header", "version": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.jsonl"
            path.write_text(
                json.dumps(header) + "\n"
                + json.dumps({"type": "generation", "input_sha256": "a", "generation": {"response": "ok"}}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(read_progress(path, header, {"a": {}})["a"]["response"], "ok")
            with self.assertRaisesRegex(ValueError, "header"):
                read_progress(path, {"type": "header", "version": 2}, {"a": {}})
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"type": "complete"}) + "\n")
            with self.assertRaisesRegex(ValueError, "already complete"):
                read_progress(path, header, {"a": {}})

    def test_select_blocks_preserves_mask_alignment_and_rejects_split_clean_input(self):
        sources = {"a": {"block_id": "one"}, "b": {"block_id": "two"}}
        records = [
            {"input_kind": "injected", "id": "a"},
            {"input_kind": "injected", "id": "b"},
            {"input_kind": "clean_only", "source_ids": ["a"]},
        ]
        rows = [{"threshold": 0.5, "inputs": ["a-mask", "b-mask", "clean-mask"]}]
        selected, filtered = select_blocks(records, rows, sources, {"one"})
        self.assertEqual(selected, [records[0], records[2]])
        self.assertEqual(filtered[0]["inputs"], ["a-mask", "clean-mask"])
        records[2]["source_ids"].append("b")
        with self.assertRaisesRegex(ValueError, "spans selected and excluded"):
            select_blocks(records, rows, sources, {"one"})


if __name__ == "__main__":
    unittest.main()
