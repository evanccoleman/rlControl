#!/bin/bash

for agent in ../outputs/runs/*/; do
    name=$(basename "$agent")
    echo "=== $name ==="
    python find_best_agent.py -x "$name"
    echo ""
done
