# SGLang + SSSD

This is a fork of the [SGLang Project](https://github.com/sgl-project/sglang) used to evaluate the Simply-Scalable Speculative Decoding (SSSD) method and compare it against baseline approaches.

This branch contains the code used for the experiments reported in the **ACL 2026** [paper](https://arxiv.org/abs/2411.05894). It is based on release `v0.5.3.post3`.

## Prerequisites
1. Running SSSD's datastore and the relevant SGLang benchmarks will require at least 64GB of RAM and 32GB VRAM. Please ensure you have the hardware for this.
2. Ensure you have access to the [Llama 3.1 model family](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). This requires approval by Meta and can take a few minutes to hours. Upon approval make sure to have the [huggingface access token](https://huggingface.co/docs/hub/en/security-tokens) ready.

## Installing & Running
We provide 2 installation routes: using Docker or conda.

### Docker
1. Ensure [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script) is installed. This can be tested with:
```bash
docker run --rm hello-world
```
2. Pull and run the `lmsysorg/sglang:v0.5.3.post3` docker image:

```
docker pull lmsysorg/sglang:v0.5.3.post3
docker run --rm -it --gpus all --ipc=host lmsysorg/sglang:v0.5.3.post3 bash
```

3. Once docker is functioning, clone this repository and run the build script from the root:
```
mkdir /workspace
cd /workspace
git clone --recurse-submodules --branch sssd-v0.5.3.post3 https://github.com/huawei-csl/sglang-sssd.git
cd sglang-sssd
python3 -m pip install --upgrade pip
pip install -e "python"
(cd sssd_speculator && pip install -e . --config-settings editable_mode=compat)
# Additional required command for MT-Bench german
apt-get update && apt-get install -y ed

# Install REST
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
pip install maturin==0.12

cd sssd_speculator/evaluation/REST/DraftRetriever_adapted/
maturin build --release --strip -i python3.12
pip install target/wheels/draftretriever*312*.whl

# (Optional) Install PIA (alternative implementation of the "LOOKAHEAD" method (from SGLang))
cd /workspace
git clone https://github.com/alipay/PainlessInferenceAcceleration.git
cd PainlessInferenceAcceleration/lookahead
pip install -e .

cd /workspace/sglang-sssd
```

4. Get hf tokens to access llama models and The Stack.

For the Stack, go to https://huggingface.co/datasets/bigcode/the-stack-dedup, give your email (same as in your hf account),
and create an authentication token in your account.

Set your tokens with
```
export HF_TOKEN=<your_token>
export STACK_TOKEN=<token_for_the_stack>
```
(usually the token is the same, if you got access from the same account).

5. To prepare models and datastores to run evaluations, run

```
nohup bash ./speculative_bench_scripts/setup_8B.sh > setup_8B.log 2>&1 &
```
or
```
nohup bash ./speculative_bench_scripts/setup_70B.sh > setup_70B.log 2>&1 &
```

6. To launch the llama 3.1 offline benchmark, use:
```
nohup bash ./speculative_bench_scripts/run_e2e_8B.sh > e2e_run.log 2>&1 &
```
or
```
nohup bash ./speculative_bench_scripts/run_e2e_70b.sh > e2e_run.log 2>&1 &
```

7. For disaggregated Prefill/Decode do the following:

If you haven't run the e2e test yet, run `bash ./speculative_bench_scripts/run_e2e.sh` to download the models and prepare the datastore, after having commented out the benchmark part. then run

```
cd speculative_bench_scripts
nohup bash ./run_dispd_grid_eval.sh > dispd_run.log 2>&1 &
```
You should set the correct values for PD, TP etc. within the script, depending on your hardware.

For the multilingual and reasoning benchmarks, after installing SGLang, simply run

```
cd speculative_bench_scripts
nohup bash ./multilingual_bench.sh > multilingual.log 2>&1 &
# or
nohup bash ./reasoning_benchmark.sh > reasoning.log 2>&1 &
```
The commands will download models, create all necessary datastores and run the evaluations (and store the results in `data_multilingual` and `data_reasoning`). For the multilingual evaluation you need to generate the data from the model first. To run the evaluation with the corresponding datasets with data found online, substitute all the `deepinfra_outputs/*` datasets with the corresponding hf datasets in the `create_subdatastores.py` files (e.g. `"sharegpt"`, `"sharegpt-ita"`, `"evol-instruct-en"`,...).

### Conda

**Note**: This route has only been tested for Ubuntu 22.04. For other systems, you may need to install additional dependencies.
1. Ensure [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed, although it is also fine to use `miniconda`.
2. Clone this repository and run the conda installation script:
```
git clone --recurse-submodules --branch add_sssd git@github.com:huawei-csl/sglang-sssd.git
cd sglang-sssd
bash ./speculative_bench_scripts/sssd_install.sh
```
4. Activate the conda environment and then:
```
export HF_TOKEN=<your_token>
export MODEL_DIR=<place_to_download_models_to>
bash ./speculative_bench_scripts/run_e2e.sh
```

When running the benchmarks you can also:
- `export DATA_DIR=<results_directory>` to set where benchmark results should be saved to.

--------------------------------------------------------------------------------

<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

| [**Blog**](https://lmsys.org/blog/2025-05-05-large-scale-ep/)
| [**Documentation**](https://docs.sglang.ai/)
| [**Join Slack**](https://slack.sglang.ai/)
| [**Join Bi-Weekly Development Meeting**](https://meeting.sglang.ai/)
| [**Roadmap**](https://github.com/sgl-project/sglang/issues/7736)
| [**Slides**](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides) |

## News
- [2025/09] 🔥 Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part II): 3.8x Prefill, 4.8x Decode Throughput ([blog](https://lmsys.org/blog/2025-09-25-gb200-part-2/)).
- [2025/09] 🔥 SGLang Day 0 Support for DeepSeek-V3.2 with Sparse Attention ([blog](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] 🔔 SGLang x AMD SF Meetup on 8/22: Hands-on GPU workshop, tech talks by AMD/xAI/SGLang, and networking ([Roadmap](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_roadmap.pdf), [Large-scale EP](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_sglang_ep.pdf), [Highlights](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_highlights.pdf), [AITER/MoRI](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_aiter_mori.pdf), [Wave](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/amd_meetup_wave.pdf)).
- [2025/08] SGLang provides day-0 support for OpenAI gpt-oss model ([instructions](https://github.com/sgl-project/sglang/issues/8833))
- [2025/05] Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).

<details>
<summary>More</summary>

- [2025/06] SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/06] Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs, chunked prefill, quantization (FP4/FP8/INT4/AWQ/GPTQ), and multi-lora batching.
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with wide industry adoption.

## Getting Started
- [Install SGLang](https://docs.sglang.ai/get_started/install.html)
- [Quick Start](https://docs.sglang.ai/basic_usage/send_request.html)
- [Backend Tutorial](https://docs.sglang.ai/basic_usage/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.ai/references/frontend/frontend_tutorial.html)
- [Contribution Guide](https://docs.sglang.ai/developer_guide/contribution_guide.html)

## Benchmark and Performance
Learn more in the release blogs: [v0.2 blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/), [v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/), [v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/), [Large-scale expert parallelism](https://lmsys.org/blog/2025-05-05-large-scale-ep/).

## Roadmap
[Development Roadmap (2025 H2)](https://github.com/sgl-project/sglang/issues/7736)

## Adoption and Sponsorship
SGLang has been deployed at large scale, generating trillions of tokens in production each day. It is trusted and adopted by a wide range of leading enterprises and institutions, including xAI, AMD, NVIDIA, Intel, LinkedIn, Cursor, Oracle Cloud, Google Cloud, Microsoft Azure, AWS, Atlas Cloud, Voltage Park, Nebius, DataCrunch, Novita, InnoMatrix, MIT, UCLA, the University of Washington, Stanford, UC Berkeley, Tsinghua University, Jam & Tea Studios, Baseten, and other major technology organizations across North America and Asia. As an open-source LLM inference engine, SGLang has become the de facto industry standard, with deployments running on over 1,000,000 GPUs worldwide.

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## Contact Us
For enterprises interested in adopting or deploying SGLang at scale, including technical consulting, sponsorship opportunities, or partnership inquiries, please contact us at contact@sglang.ai.

## Acknowledgment
We learned the design and reused code from the following projects: [Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), and [LMQL](https://github.com/eth-sri/lmql).
