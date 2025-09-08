#!/usr/bin/env bash
set -e

## LLaMA 3.1 ##

# AUTOREGRESSIVE
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024

# EAGLE2
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path /storage/datasets/huggingface/hub/sglang-EAGLE-Llama-3.1-Instruct-8B-bf16 \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --speculative-num-draft-tokens 8 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024

# EAGLE 3
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /storage/datasets/huggingface/hub/sglang-EAGLE3-Llama-3.1-Instruct-8B-bf16 \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --speculative-num-draft-tokens 8 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 4 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024

# SSSD with manual parameters
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --speculative-algorithm SSSD \
    --speculative-draft-model-path /storage/users/mmarzollo/datastores/hitz-magpie-llama3.1-8B_magpie-llama-3.1-MT-500.idx \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --speculative-num-draft-tokens 8 \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 5 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024

# SSSD adaptive
python3 -m sglang.bench_offline_throughput --model /storage/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --speculative-algorithm SSSD \
    --speculative-draft-model-path /storage/users/mmarzollo/datastores/hitz-magpie-llama3.1-8B_magpie-llama-3.1-MT-500.idx \
    --cuda-graph-max-bs 16 \
    --max-running-requests 16 \
    --num-prompts 80 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --dataset-name mt-bench \
    --sharegpt-output-len 1024 \
    --speculative-adaptive


# HYPERPARAMETER SEARCH

python3 -m bench_speculative --model /storage/datasets/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /storage/datasets/huggingface/hub/sglang-EAGLE3-Llama-3.1-Instruct-8B-bf16
