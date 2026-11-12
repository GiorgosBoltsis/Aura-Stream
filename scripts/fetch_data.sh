#!/usr/bin/env bash
set -euo pipefail
DATA="${1:-data}"
mkdir -p "$DATA"
cd "$DATA"
if [ ! -d confiwent_repo ]; then
  git clone --depth 1 https://github.com/confiwent/Real-world-bandwidth-traces confiwent_repo
fi
if [ ! -d ucc5g ]; then
  git clone --depth 1 https://github.com/uccmisl/5Gdataset uccmisl_5g
  mkdir -p ucc5g
  unzip -q uccmisl_5g/5G-production-dataset.zip -d ucc5g
fi
echo "datasets ready in $(pwd)"
