#!/usr/bin/env bash
set -euo pipefail

python -m scripts.test_json_schema
python -m scripts.test_dataset_generation
python -m scripts.test_data_cache
python -m scripts.test_training_overfit
python -m scripts.test_concretization
python -m scripts.test_planner_build
# End-to-end can be expensive; run it only if explicitly requested.
# python -m scripts.test_end_to_end

echo "All checks completed."
