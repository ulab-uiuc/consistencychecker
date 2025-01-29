![llmcheck](./llmcheck.png)

<h1 align="center">LLMCheck</h1>

![Python Versions](https://img.shields.io/badge/Supported%20Python-3.8--3.11-blue)
![Mypy Checked](https://img.shields.io/badge/Mypy-Checked-brightgreen)
![isort Checked](https://img.shields.io/badge/isort-Checked-brightgreen)
![ruff Checked](https://img.shields.io/badge/ruff-Checked-brightgreen)

## Introduction

LLMCheck is a Python package that evaluates Large Language Models (LLMs) through a novel tree-based approach. It tests an LLM's consistency by applying a series of transform-reverse operations (like translation) and measuring how well the model maintains semantic consistency across these transformations.

## Features

- 🤖 Support for both API-based (via LiteLLM) and local LLM evaluation
- 🌲 Tree-based evaluation structure with customizable depth and branching
- 📊 Multiple similarity metrics for consistency evaluation
- 🧰 CLI tool for easy evaluation

## Usage

The dependencies are managed by `poetry`. First, install poetry by running:

```bash
pip install poetry
```

To install the dependencies, run:

```bash
poetry install
```
Then simply use `llmcheck` as a CLI tool. There are three parameter combinations:

1. Generate benchmark only:
    ```bash
    llmcheck --config <path_to_config> --benchmark_output <path_for_saving_benchmark> --benchmark_only
    ```

2. Evaluate using existing benchmark:
    ```bash
    llmcheck --config <path_to_config> --benchmark <path_to_benchmark> --result_output_folder <path_for_saving_results>
    ```

3. Generate benchmark and evaluate:
    ```bash
    llmcheck --config <path_to_config> --result_output_folder <path_for_saving_results>
    ```

The acceptable config file format is a little bit complicated but it is explained in the `config.yaml` file. To avoid outputs writing on top of each other, we default the result file to be tagged with the current time.

Here are examples to run the 2 stages separately:

```bash
llmcheck --config config_coding.yaml --benchmark_output bench_coding.yaml --benchmark_only

llmcheck --config config_coding.yaml --benchmark bench_coding.yaml --result_output_folder output_folder
```

## LLM Options

As we use`litellm` to connect to the LLM, you will have access to all the models that `litellm` supports. You should set API keys as environment variables. For example, to use OpenAI's GPT-4, you should set

```bash
export OPENAI_API_KEY=your_api_key
```

## vllm/Ollama

You are recommended to use `vllm` or `ollama` to access modest-sized open-source LLMs.

### vllm

Here is a simple example of how to use `vllm` to serve a model. The number of GPUs varies depending on the model size and your setup. The following command demonstrates how to serve a model on 4 GPUs.

`GPU_MEMORY_UTILIZATION` is the fraction of GPU memory that the model will use. The `TENSOR_PARALLEL_SIZE` is the number of GPUs that will be used for tensor parallelism.

Here `host` is set to `localhost`, for if you use `0.0.0.0`, you might end up exposing your served model to the internet without access control.

```bash
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=32
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve \
    meta-llama/Llama-3.1-70B-Instruct \
    --host 127.0.0.1 \
    --port 8001 \
    --max_model_len 65536 \
    --gpu_memory_utilization 0.8 \
    --tensor-parallel-size 4
```

### ollama

Ollama comes as a installable package. If you do not have permission to install software on your machine, you can download a pre-compiled version of 'ollama' from the [releases page](https://github.com/ollama/ollama/releases).
