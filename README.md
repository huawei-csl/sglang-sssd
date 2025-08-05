This is a fork of the [SGLang Project](https://github.com/sgl-project/sglang) usable to evaluate the Simply-Scalable Speculative Decoding (SSSD) method and compare it to EAGLE3.

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
2. Ensure the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed so docker can access GPUs. This can be checked with:
```bash
docker run --rm --gpus all nvidia/cuda:12.0.1-runtime-ubuntu22.04 nvidia-smi
```
Note: if you plan to run the built image on a cloud instance, GPUs are usually already provided to the Docker container, so this step can be skipped.

3. Once docker is functioning, clone this repository (with `--recurse-submodules`) and run the following from the root:
```
bash ./basic_scripts/build_docker.sh
```

4. If running locally use the following script:
```
export HF_TOKEN=<your_token>
bash ./basic_scripts/run_docker.sh
```
Otherwise you will want to tag the docker image and push it to a registry (e.g `docker tag sglang_sssd:latest <repository_path>/sglang_sssd:latest && docker push <repository_path>/sglang_sssd:latest`) The image will run the benchmark and then exit.

### Conda

**Note**: This route has only been tested for Ubuntu 22.04. For other sytems, you may need to install additional dependencies.
1. Ensure [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed, although it is also fine to use `miniconda`.
2. Clone this repository (with `--recurse-submodules`) and run `bash ./basic_scripts/sssd_install.sh` to create the conda environment and install dependencies.
3. Activate the conda environment and then:
```
export HF_TOKEN=<your_token>
export DATA_DIR=<place_to_download_models_to>
bash ./basic_scripts/run_e2e.sh
```

### Extras
When running the benchmarks you can also:
- `export RUN_HYPERPARAMETER_SEARCH=true` to run a hyper-parameter search on both EAGLE3 and SSSD for better results. This additional step will take 4-12 hours depending on the machine.
- `export UPLOAD_RESULTS=true` to upload the results to this repository's issues. Otherwise they will be printed at the end of the benchmark run and saved to `data/collected_results.json`. Additionally also set `export RESULT_REPO_URL=https://github.com/<owner>/<repository>/issues` to point to the repository where the results should be uploaded.

## Results
If you chose to set `export RUN_HYPERPARAMETER_SEARCH=true` the results will be uploaded automatically to the issues of this repository with a URL provided at the end. The results contain:
1. Benchmark performance over different batch sizes
2. Hyperparameter search results
3. Basic information about the hardware and software used for the benchmark (CPUs, RAM, GPUs, CUDA version, etc.)

--------------------------------------------------------------------------------

<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://img.shields.io/pypi/dm/sglang)
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
| [**Roadmap**](https://github.com/sgl-project/sglang/issues/4042)
| [**Slides**](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides) |

## News
- [2025/06] 🔥 SGLang, the high-performance serving infrastructure powering trillions of tokens daily, has been awarded the third batch of the Open Source AI Grant by a16z ([a16z blog](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/)).
- [2025/06] 🔥 Deploying DeepSeek on GB200 NVL72 with PD and Large Scale EP (Part I): 2.7x Higher Decoding Throughput ([blog](https://lmsys.org/blog/2025-06-16-gb200-part-1/)).
- [2025/05] 🔥 Deploying DeepSeek with PD Disaggregation and Large-scale Expert Parallelism on 96 H100 GPUs ([blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/)).
- [2025/03] Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html))
- [2025/03] SGLang Joins PyTorch Ecosystem: Efficient LLM Serving Engine ([PyTorch blog](https://pytorch.org/blog/sglang-joins-pytorch/))
- [2024/12] v0.4 Release: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs ([blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)).
- [2024/07] v0.2 Release: Faster Llama3 Serving with SGLang Runtime (vs. TensorRT-LLM, vLLM) ([blog](https://lmsys.org/blog/2024-07-25-sglang-llama3/)).

<details>
<summary>More</summary>

- [2025/02] Unlock DeepSeek-R1 Inference Performance on AMD Instinct™ MI300X GPU ([AMD blog](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1_Perf/README.html))
- [2025/01] SGLang provides day one support for DeepSeek V3/R1 models on NVIDIA and AMD GPUs with DeepSeek-specific optimizations. ([instructions](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3), [AMD blog](https://www.amd.com/en/developer/resources/technical-articles/amd-instinct-gpus-power-deepseek-v3-revolutionizing-ai-development-with-sglang.html), [10+ other companies](https://x.com/lmsysorg/status/1887262321636221412))
- [2024/10] The First SGLang Online Meetup ([slides](https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#the-first-sglang-online-meetup)).
- [2024/09] v0.3 Release: 7x Faster DeepSeek MLA, 1.5x Faster torch.compile, Multi-Image/Video LLaVA-OneVision ([blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)).
- [2024/02] SGLang enables **3x faster JSON decoding** with compressed finite state machine ([blog](https://lmsys.org/blog/2024-02-05-compressed-fsm/)).
- [2024/01] SGLang provides up to **5x faster inference** with RadixAttention ([blog](https://lmsys.org/blog/2024-01-17-sglang/)).
- [2024/01] SGLang powers the serving of the official **LLaVA v1.6** release demo ([usage](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#demo)).

</details>

## About
SGLang is a fast serving framework for large language models and vision language models.
It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.
The core features include:

- **Fast Backend Runtime**: Provides efficient serving with RadixAttention for prefix caching, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor parallelism, pipeline parallelism, expert parallelism, structured outputs, chunked prefill, quantization (FP8/INT4/AWQ/GPTQ), and multi-lora batching.
- **Flexible Frontend Language**: Offers an intuitive interface for programming LLM applications, including chained generation calls, advanced prompting, control flow, multi-modal inputs, parallelism, and external interactions.
- **Extensive Model Support**: Supports a wide range of generative models (Llama, Gemma, Mistral, Qwen, DeepSeek, LLaVA, etc.), embedding models (e5-mistral, gte, mcdse) and reward models (Skywork), with easy extensibility for integrating new models.
- **Active Community**: SGLang is open-source and backed by an active community with industry adoption.

## Getting Started
- [Install SGLang](https://docs.sglang.ai/start/install.html)
- [Quick Start](https://docs.sglang.ai/backend/send_request.html)
- [Backend Tutorial](https://docs.sglang.ai/backend/openai_api_completions.html)
- [Frontend Tutorial](https://docs.sglang.ai/frontend/frontend.html)
- [Contribution Guide](https://docs.sglang.ai/references/contribution_guide.html)

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
