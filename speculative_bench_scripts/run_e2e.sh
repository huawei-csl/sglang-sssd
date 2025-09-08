#!/usr/bin/env bash
# Downloads models, creates SSSD datastores, and runs benchmarks
set -e

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

export MODEL_DIR="${MODEL_DIR:-/storage}" # Where to save models (large files)

DATA_DIR="${DATA_DIR:-data}" # Where to save benchmark results and other data
UPLOAD_RESULTS="${UPLOAD_RESULTS:-false}"  # Set to true to upload results to github automatically
RUN_HYPERPARAMETER_SEARCH="${RUN_HYPERPARAMETER_SEARCH:-false}"  # Set to true to run hyperparameter search
UPLOAD_URL="${UPLOAD_URL:-https://sssd-result-receiver-327304000081.europe-west1.run.app}"
SAMPLE_SIZE="${SAMPLE_SIZE:-3}"  # Number of times to run each benchmark to take the median performance.
EVAL_LOGLEVEL="${EVAL_LOGLEVEL:-WARNING}"  # Set to DEBUG or INFO to see more detailed logs
HPARAM_LOGLEVEL="${HPARAM_LOGLEVEL:-WARNING}"  # Set to DEBUG or INFO to see more detailed logs

start=$SECONDS

if [ "${UPLOAD_RESULTS}" = "true" ] && [ -z "${RESULT_REPO_URL}"]; then
  echo "Error: RESULT_REPO_URL must be defined when UPLOAD_RESULTS is true. It should have the shape https://api.github.com/repos/<owner>/<repository>/issues"
  exit 1
fi

echo "Getting user machine specs..."
python3 speculative_bench_scripts/collect_env.py --out-dir "$DATA_DIR"

echo "Downloading models..."
./speculative_bench_scripts/download_models.sh

echo "Converting EAGLE heads to BF16..."
EAGLE2_LLAMA_3_1_8B_PATH=${MODEL_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE-Llama-3.1-Instruct-8B
EAGLE3_LLAMA_3_1_8B_PATH=${MODEL_DIR}/datasets/huggingface/models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B
python3 speculative_bench_scripts/change_model_type.py --model $EAGLE2_LLAMA_3_1_8B_PATH
python3 speculative_bench_scripts/change_model_type.py --model $EAGLE3_LLAMA_3_1_8B_PATH


echo "Creating SSSD datastores..."
LLAMA3_1_8B_PATH=$MODEL_DIR/datasets/huggingface/models/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ 
MAGPIE_DATASTORE_IDX_PATH=$MODEL_DIR/datasets/sssd_speculator/hitz-magpie-llama3.1-8B_magpie-llama-3.1-MT-500.idx

python3 sssd_speculator/datastore_creation/create_datastore.py --model $LLAMA3_1_8B_PATH --index_file_path $MAGPIE_DATASTORE_IDX_PATH --datasets hitz-magpie-llama3.1-8b magpie-llama31-pro

HPARAM_BENCHMARK="sharegpt"
if [ "$RUN_HYPERPARAMETER_SEARCH" = "true" ]; then
  echo "Hyperparameter search..."

  COMMON_ARGS="--model $LLAMA3_1_8B_PATH \
      --dataset-name $HPARAM_BENCHMARK \
      --sharegpt-output-len 256 \
      --disable-ignore-eos \
      --apply-chat-template \
      --enable-metrics \
      --num-successful-trials 15 \
      --max-draft-tokens-in-batch 1024 \
      --sample-size $SAMPLE_SIZE \
      --log-level $HPARAM_LOGLEVEL"

  python3 speculative_bench_scripts/hyperparameter_search.py $COMMON_ARGS \
      --speculative-algorithm EAGLE3 \
      --speculative-draft-model-path $EAGLE3_LLAMA_3_1_8B_PATH \
      --hparam-output-dir "$DATA_DIR/hyperparameter_search/$HPARAM_BENCHMARK"

else
  echo "Skipping hyperparameter search because RUN_HYPERPARAMETER_SEARCH is not set to 'true'."
fi

echo "Running benchmarks..."
BATCH_SIZES=(1 4 8 16 32 48 64)
OUTPUT_LEN=1024
BENCHMARK="mt-bench"
EVAL_DATA_DIR="$DATA_DIR/evaluation/$BENCHMARK"
mkdir -p $EVAL_DATA_DIR


for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    COMMON_ARGS="--model $LLAMA3_1_8B_PATH \
        --dataset-name $BENCHMARK \
        --sharegpt-output-len $OUTPUT_LEN \
        --disable-ignore-eos \
        --apply-chat-template \
        --enable-metrics \
        --sample-size $SAMPLE_SIZE \
        --log-level $EVAL_LOGLEVEL
        --cuda-graph-max-bs $BATCH_SIZE \
        --max-running-requests $BATCH_SIZE"    
    
    echo "Running autoregressive baseline on $BENCHMARK for batch size ${BATCH_SIZE}"
    python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
        --result-filename "$EVAL_DATA_DIR/Autoregressive_$BATCH_SIZE.json"

    echo "Running EAGLE2 (default parameters) on $BENCHMARK for batch size ${BATCH_SIZE}"
    python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
        --speculative-algorithm EAGLE \
        --speculative-draft-model-path $EAGLE2_LLAMA_3_1_8B_PATH \
        --result-filename "$EVAL_DATA_DIR/EAGLE2-default_$BATCH_SIZE.json"

    echo "Running EAGLE3 (default parameters) on $BENCHMARK for batch size ${BATCH_SIZE}"
    python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
        --speculative-algorithm EAGLE3 \
        --speculative-draft-model-path $EAGLE3_LLAMA_3_1_8B_PATH \
        --result-filename "$EVAL_DATA_DIR/EAGLE3-default_$BATCH_SIZE.json"

    if [ "$RUN_HYPERPARAMETER_SEARCH" = "true" ]; then
      echo "Running EAGLE3 (optimised parameters) on $BENCHMARK for batch size ${BATCH_SIZE}"
          HPARAM_FILE="$DATA_DIR/hyperparameter_search/$HPARAM_BENCHMARK/EAGLE3_${BATCH_SIZE}_results.json"
      if [ ! -f $HPARAM_FILE ]; then
          echo "Skipping batch size $BATCH_SIZE: file $HPARAM_FILE does not exist"
          continue
      fi
      python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
          --speculative-algorithm EAGLE3 \
          --speculative-draft-model-path $EAGLE3_LLAMA_3_1_8B_PATH \
          --result-filename "$EVAL_DATA_DIR/EAGLE3-optimised_$BATCH_SIZE.json" \
          --hparam-file "$HPARAM_FILE"

      # Use the same parameters for EAGLE2, the search takes too long and won't make much difference
      echo "Running EAGLE2 (optimised parameters) on $BENCHMARK for batch size ${BATCH_SIZE}"
          HPARAM_FILE="$DATA_DIR/hyperparameter_search/$HPARAM_BENCHMARK/EAGLE3_${BATCH_SIZE}_results.json"
      if [ ! -f $HPARAM_FILE ]; then
          echo "Skipping batch size $BATCH_SIZE: file $HPARAM_FILE does not exist"
          continue
      fi
      python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
          --speculative-algorithm EAGLE \
          --speculative-draft-model-path $EAGLE2_LLAMA_3_1_8B_PATH \
          --result-filename "$EVAL_DATA_DIR/EAGLE2-optimised_$BATCH_SIZE.json" \
          --hparam-file "$HPARAM_FILE"
    fi

    echo "Running SSSD on $BENCHMARK for batch size: ${BATCH_SIZE}"
    python3 speculative_bench_scripts/eval_benchmark.py $COMMON_ARGS \
        --speculative-algorithm SSSD \
        --speculative-draft-model-path $MAGPIE_DATASTORE_IDX_PATH \
        --speculative-adaptive \
        --result-filename "$EVAL_DATA_DIR/SSSD_$BATCH_SIZE.json"
        # Remove --speculative-adaptive and use --use-hparam-mapping if you want to use some decent defaults
        # without dynamicity
done

# Collect results
echo "Gathering Results..."
python3 speculative_bench_scripts/collect_results.py --submission-url "$RESULT_REPO_URL" --data-path "$DATA_DIR"

# Upload results
RESULTS_FILE="$DATA_DIR/collected_results.json"
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

# Calculate elapsed time
runtime=$((SECONDS - start))

# Convert to hours:minutes:seconds format
hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$(((runtime % 3600) % 60))

printf "\nRuntime: $hours:$minutes:$seconds (hh:mm:ss)"
