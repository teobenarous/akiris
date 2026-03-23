#!/bin/bash
set -euo pipefail

echo "Starting Simulation (Speedrun Mode)..."

# (1) Ensure clean state
rm -f state/patients.pkl state/journal.jsonl

# (2) Spin up services, forcing the MESSAGE_DELAY to 0
MESSAGE_DELAY=0 RECONNECT=false docker compose up --build -d simulator app

echo "Waiting 3 seconds for TCP sockets to bind..."
sleep 3

# (3) Teardown
echo "Tearing down infrastructure..."
docker compose stop simulator app
docker compose rm -f simulator app

# (4) Stream the live evaluation
echo "Running Evaluation..."
python scripts/evaluate.py

echo "Simulation Complete."