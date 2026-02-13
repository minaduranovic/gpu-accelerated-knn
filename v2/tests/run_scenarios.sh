#!/usr/bin/env bash
set -euo pipefail

BIN="${1:-./main}"

run_case() {
  local name="$1"
  local dim="$2"
  local clusters="$3"
  local points="$4"
  local batch_q="$5"

  echo "=== ${name} ==="
  "$BIN" "$dim" "$clusters" "$points" "$batch_q"
  echo
}

run_case "synth_tiny"          8  64   32768   512
run_case "synth_small_dim"     4  256  131072  1024
run_case "synth_exact_16"      16 256  131072  1024
run_case "synth_large_dim"     12 512  262144  1024
run_case "synth_huge"          8  1024 524288  2048
run_case "low_occupancy"       8  512  65536   2048
run_case "high_dim_stress"     16 512  262144  1024
run_case "cluster_imbalance"   8  128  262144  1024
run_case "degenerate_clusters" 8  512  131072  1024
run_case "worst_pruning"       8  256  131072  8192
