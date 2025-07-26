#!/usr/bin/env bash
# Downloads models, creates SSSD datastores, and runs benchmarks
set -e

export DATA_DIR="${DATA_DIR:-/storage}"
export UPLOAD_RESULTS="${UPLOAD_RESULTS:-true}"  # Set to false to skip uploading results
export UPLOAD_URL="${UPLOAD_URL:-https://sssd-result-receiver-327304000081.europe-west1.run.app}"


echo "Getting user machine specs..."
python3 basic_scripts/collect_env.py

echo "Downloading models..."
./basic_scripts/download_models.sh

echo "Converting EAGLE heads to BF16..."
EAGLE3_LLAMA_3_1_8B_PATH=${DATA_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
python3 basic_scripts/change_model_type.py --model $EAGLE3_LLAMA_3_1_8B_PATH


echo "Creating SSSD datastores..."
LLAMA3_1_8B_PATH=$DATA_DIR/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ 
MAGPIE_DATASTORE_IDX_PATH=$DATA_DIR/datasets/sssd_speculator_sglang/ultrachat_magpie_llama3.1-MT-500.idx

python3 sssd_speculator_sglang/datastore_creation/create_datastore.py --model $LLAMA3_1_8B_PATH --index_file_path $MAGPIE_DATASTORE_IDX_PATH --mode sharegpt_ultrachat_magpie_responses


echo "Hyperparameter search..."
# TODO for EAGLE2 and constrained SSSD

python3 basic_scripts/hyperparameter_search.py --model $LLAMA3_1_8B_PATH  \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path $EAGLE3_LLAMA_3_1_8B_PATH \
    --sharegpt-output-len 256 \
    --disable-ignore-eos \
    --apply-chat-template \
    --enable-metrics \
    --num-successful-trials 10
    # --batch-sizes 1 \
    # --sample-size 1 \
    # --num-successful-trials 1   

python3 basic_scripts/hyperparameter_search.py --model $LLAMA3_1_8B_PATH  \
    --speculative-algorithm SSSD \
    --speculative-draft-model-path $MAGPIE_DATASTORE_IDX_PATH \
    --sharegpt-output-len 256 \
    --enable-metrics \
    --disable-ignore-eos \
    --apply-chat-template \
    --num-successful-trials 10
    # --batch-sizes 1 \
    # --sample-size 1 \
    # --num-successful-trials 1   


# echo "Running benchmarks..."
# # TODO this is not all of them, but just for testing

# echo "Running EAGLE3 x Llama 3.1 8B benchmark..."
# python3 -m sglang.bench_offline_throughput --model $LLAMA3_1_8B_PATH  \
#     --speculative-algorithm EAGLE3 \
#     --speculative-draft-model-path $EAGLE3_LLAMA_3_1_8B_PATH \
#     --speculative-num-steps 5 \
#     --speculative-eagle-topk 3 \
#     --speculative-num-draft-tokens 8 \
#     --cuda-graph-max-bs 2 \
#     --max-running-requests 2 \
#     --num-prompts 10 \
#     --sharegpt-output-len 256 \
#     --enable_metrics
#     # --disable-cuda-graph \

# echo "Running SSSD x Llama 3.1 8B benchmark..."
# python3 -m sglang.bench_offline_throughput --model $LLAMA3_1_8B_PATH  \
#     --speculative-algorithm SSSD \
#     --speculative-draft-model-path $MAGPIE_DATASTORE_IDX_PATH \
#     --speculative-num-steps 6 \
#     --speculative-eagle-topk 4 \
#     --speculative-num-draft-tokens 8 \
#     --cuda-graph-max-bs 2 \
#     --max-running-requests 2 \
#     --num-prompts 10 \
#     --sharegpt-output-len 256 \
#     --enable_metrics

# Collect results
echo "Gathering Results..."
python3 basic_scripts/collect_results.py

# Upload results
RESULTS_FILE="data/collected_results.json"
if [ "$UPLOAD_RESULTS" = "true" ]; then

  # Check if the results file actually exists before trying to upload
  if [ -f "$RESULTS_FILE" ]; then
    echo "Uploading results from $RESULTS_FILE to $UPLOAD_URL..."
    curl -X POST "$UPLOAD_URL" \
         -H "Content-Type: application/json" \
         -d @"$RESULTS_FILE"
  else
    echo "Error: Results file not found at '$RESULTS_FILE'. Cannot upload."
    # Optionally, exit with an error code if the file is missing
    # exit 1 
  fi
else
  echo "Skipping results upload because UPLOAD_RESULTS is not set to 'true'."
  cat $RESULTS_FILE
fi
