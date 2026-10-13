#!/bin/bash

uv run python -m senior_project collect \
  --input data/task_samples.json \
  --split train \
  --output data/attention_features.npz

uv run python -m senior_project train \
  --features data/attention_features.npz \
  --output models/attention_detector.pt

uv run python -m senior_project evaluate \
  --input data/task_samples.json \
  --split test \
  --detector models/attention_detector.pt \
  --output data/evaluation.json