#!/usr/bin/env bash
set -euo pipefail

python src/selfplay/smoke_test_openspiel.py
python src/sparse/smoke_test_minigrid.py
python src/llm/offline_sampling.py --num_prompts 16 --group_size 8

echo "Smoke tests finished."
