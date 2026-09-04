#!/bin/bash
# Train and evaluate seeds 0-4 back to back and write results/seed_sweep_summary.csv.
#
# -e matters as much as the rest of the script: without it a failed training
# run would leave the loop running and still append a CSV row, so the summary
# behind the README table could end up half-populated with no visible error.
set -euo pipefail

# Run from the repository this script lives in, whatever the caller's working
# directory is. The `rm -rf` below is relative, so a cd that silently failed
# would delete directories out of some unrelated tree.
cd "$(dirname "$0")"

mkdir -p results
SUMMARY=results/seed_sweep_summary.csv
echo "seed,episodes,resolved,total,success_rate,mean_reward,std_reward" > "$SUMMARY"

for seed in 0 1 2 3 4; do
  outdir="experiments/checkpoints_seed${seed}"
  rm -rf "$outdir"
  echo "=== SEED $seed: training ==="
  python -m src.train --num-episodes 3000 --lr 1e-4 --rollout-steps 2048 \
    --minibatch-size 64 --seed "$seed" --output-dir "$outdir" --log-interval 500 \
    2>&1 | tail -5

  echo "=== SEED $seed: evaluating ==="
  eval_out=$(python -m src.evaluate --model-path "$outdir/best_model.pt" --episodes 50 --seed 1000 2>&1)
  echo "$eval_out" | tail -6

  # sed rather than `grep -oP`: perl-regex mode is a GNU extension and is not
  # available in the BSD grep that ships with macOS.
  resolved=$(echo "$eval_out" | sed -n 's/^Resolved: \([0-9]\{1,\}\)\/.*/\1/p')
  total=$(echo "$eval_out" | sed -n 's/^Resolved: [0-9]\{1,\}\/\([0-9]\{1,\}\).*/\1/p')
  rate=$(echo "$eval_out" | sed -n 's/^Resolved: [0-9]\{1,\}\/[0-9]\{1,\} (\([0-9.]\{1,\}\)%).*/\1/p')
  mean=$(echo "$eval_out" | sed -n 's/^Mean reward: \(-\{0,1\}[0-9.]\{1,\}\) +\/-.*/\1/p')
  std=$(echo "$eval_out" | sed -n 's/^Mean reward: -\{0,1\}[0-9.]\{1,\} +\/- \([0-9.]\{1,\}\).*/\1/p')

  # A missing field means the evaluation output did not have the shape this
  # script parses, which makes every number downstream of it untrustworthy.
  # Stop rather than write a row with holes in it.
  for field_name in resolved total rate mean std; do
    if [ -z "${!field_name}" ]; then
      echo "ERROR: seed $seed: could not parse '$field_name' from the evaluation output" >&2
      echo "$eval_out" >&2
      exit 1
    fi
  done

  echo "$seed,3000,$resolved,$total,$rate,$mean,$std" >> "$SUMMARY"
  echo "=== SEED $seed done: $resolved/$total ($rate%) ==="
done

echo "=== SWEEP COMPLETE ==="
cat "$SUMMARY"
