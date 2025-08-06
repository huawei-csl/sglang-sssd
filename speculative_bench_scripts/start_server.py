import os
from sglang.test.test_utils import is_in_ci
from sglang.utils import wait_for_server, print_highlight, terminate_process

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#### NO SPECULATION ####

# # python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0
# server_process, port = launch_server_cmd(
#     """
# python3 -m sglang.launch_server --model-path /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/ \
#     --host 0.0.0.0 \
#     --cuda-graph-max-bs 4 \

# """
# )

# # #### EAGLE ####

# server_process, port = launch_server_cmd(
#     """
# python3 -m sglang.launch_server --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/  \
#     --speculative-algorithm EAGLE \
#     --speculative-draft-model-path /storage/datasets/huggingface/models/sglang-EAGLE-LLaMA3-Instruct-8B-bf16 \
#     --speculative-num-steps 3 \
#     --speculative-eagle-topk 4 \
#     --speculative-num-draft-tokens 16 \
#     --cuda-graph-max-bs 4
#     """
# )
# # --cuda-graph-max-bs 1 \
# # --tp 2
# wait_for_server(f"http://localhost:{port}")



# #### SSSD ####

server_process, port = launch_server_cmd(
    """
python3 -m sglang.launch_server --model /storage/datasets/huggingface/models/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a/  \
    --speculative-algorithm SSSD \
    --speculative-draft-model-path /storage/users/mmarzollo/datastores/ultrachat_magpie_llama3.idx \
    --speculative-eagle-topk 4 \
    --speculative-num-steps 6 \
    --speculative-num-draft-tokens 16 \
    --cuda-graph-max-bs 4 \
    """
)
# --cuda-graph-max-bs 1 \
# --disable-cuda-graph \
# --tp 2
wait_for_server(f"http://localhost:{port}")