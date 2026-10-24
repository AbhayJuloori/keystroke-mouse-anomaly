#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

if [[ ! -f data/raw/DSL-StrongPasswordData.csv ]]; then
  curl --fail --location --output data/raw/DSL-StrongPasswordData.csv \
    https://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv
fi

if [[ ! -f data/raw/balabit/public_labels.csv ]]; then
  if [[ -e data/raw/balabit ]]; then
    echo "data/raw/balabit exists but is incomplete; move it aside and rerun" >&2
    exit 1
  fi
  git clone --depth 1 https://github.com/balabit/Mouse-Dynamics-Challenge data/raw/balabit
  rm -rf data/raw/balabit/.git
fi
