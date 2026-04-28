These scripts are legacy/manual experiment drivers.

Preferred structure now:

- single-run runners:
  - [script/test_gap.sh](/work/leotsia0416/projects/SDAR/script/test_gap.sh)
  - [script/test.sh](/work/leotsia0416/projects/SDAR/script/test.sh)
- launcher:
  - [script/launch_eval.py](/work/leotsia0416/projects/SDAR/script/launch_eval.py)
- manifests:
  - [script/config/experiments/README.md](/work/leotsia0416/projects/SDAR/script/config/experiments/README.md)

These `.sh` files are kept for compatibility and as concrete references for
older sweeps, comparisons, or one-off evaluations. New experiment definitions
should preferably go into YAML manifests under `script/config/experiments/`.
