#!/bin/bash
set -e

if [ -f /app/report.md ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
