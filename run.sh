#!/usr/bin/env bash
set -euo pipefail

gen_model="gpt-5"
metarm_model="gemini-3-pro-thinking"
gen_base_url=""
metarm_base_url=""
gen_api_key=""
metarm_api_key=""
num_threads=32
gen_max_tokens=8192
metarm_max_tokens=16384
temperature=0.4
overwrite=1
benchmarks="helpsteer3 reward_bench2 ppe_human_preference_v1"

if [[ -z "$gen_base_url" ]]; then
  echo "Please set gen_base_url in this script."
  exit 1
fi
if [[ -z "$metarm_base_url" ]]; then
  echo "Please set metarm_base_url in this script."
  exit 1
fi


run_generate() {
  local benchmark="$1"
  python pub_aug_bmk/generate_model_judgment.py \
    --input_jsonl "pub_aug_bmk/data/${benchmark}_aug.jsonl" \
    --model "$gen_model" \
    --base_url "$gen_base_url" \
    --api_key "$gen_api_key" \
    --temperature "$temperature" \
    --max_tokens "$gen_max_tokens" \
    --num_threads "$num_threads" \
    --overwrite
}

run_metarm() {
  local benchmark="$1"
  python pub_aug_bmk/metarm_judge.py \
    --input_jsonl "pub_aug_bmk/result/${benchmark}_aug__${gen_model}.jsonl" \
    --model "$metarm_model" \
    --base_url "$metarm_base_url" \
    --api_key "$metarm_api_key" \
    --max_tokens "$metarm_max_tokens" \
    --overwrite
}

for benchmark in $benchmarks; do
  run_generate "$benchmark"
done

for benchmark in $benchmarks; do
  run_metarm "$benchmark"
done

