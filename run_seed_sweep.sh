#!/bin/bash
set -uo pipefail
cd /tmp/ppo-robot-controller
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

  resolved=$(echo "$eval_out" | grep -oP 'Resolved: \K[0-9]+(?=/)')
  total=$(echo "$eval_out" | grep -oP 'Resolved: [0-9]+/\K[0-9]+')
  rate=$(echo "$eval_out" | grep -oP 'Resolved: [0-9]+/[0-9]+ \(\K[0-9.]+(?=%)')
  mean=$(echo "$eval_out" | grep -oP 'Mean reward: \K-?[0-9.]+(?= \+/-)')
  std=$(echo "$eval_out" | grep -oP 'Mean reward: -?[0-9.]+ \+/- \K[0-9.]+')

  echo "$seed,3000,$resolved,$total,$rate,$mean,$std" >> "$SUMMARY"
  echo "=== SEED $seed done: $resolved/$total ($rate%) ==="
done

echo "=== SWEEP COMPLETE ==="
cat "$SUMMARY"
