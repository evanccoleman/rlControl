#!/bin/bash

mkdir -p ../outputs/training_runs

output_file="best_agents_$(date +%Y-%m-%d).txt"

for agent in ../outputs/training_runs/*/; do
    name=$(basename "$agent")
    echo "=== $name ==="
    python find_best_agent.py -x "$name"
    echo ""
done > "$output_file"
