#!/bin/bash
# Re-run every computation in the note. Usage: PYTHON=/path/to/python bash run_all.sh
set -eu
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
[ -d results_shipped ] || cp -r results results_shipped
cd code/traffic
echo "certify_counterexample.py";  $PY certify_counterexample.py > ../../results/traffic_counterexample.txt
echo "counterexample_sim.py";      $PY counterexample_sim.py > ../../results/traffic_counterexample_sim.txt
echo "order_transients.py";        $PY order_transients.py > ../../results/traffic_order_transients.txt
echo "order_matters.py (20-60 min)"; $PY order_matters.py > ../../results/traffic_order_matters.txt
cd ..
echo "make_figures.py";            $PY make_figures.py
cd ..
for f in results_shipped/*.txt; do
  name=$(basename "$f")
  if cmp -s "$f" "results/$name"; then echo "identical to the shipped output: $name"
  else echo "DIFFERS from the shipped output: $name (compare the two files)"; fi
done
