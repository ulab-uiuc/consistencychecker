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

Then simply use ```llmcheck``` as a cli tool. There are two parameters:

```
--config: path to the config file, which is yaml format
--result: path to the result file, which is also in yaml format
```

The acceptable config file format is a little bit complicated but it is explained in ```config.yaml``` file. To avoid outputs writing on top of each other, we default the result file to be tagged with the current time.

One example of the usage is:

```bash
llmcheck --config config.yaml --result result.yaml
```

## LLM Options

As we use`litellm` to connect to the LLM, you will have access to all the models that `litellm` supports. You should set API keys as environment variables. For example, to use OpenAI's GPT-4, you should set

```bash
export OPENAI_API_KEY=your_api_key
```

## Ollama

You are recommended to use 'ollama' to access modest-sized open-source LLMs. If you do not have permission to install software on your machine, you can download a pre-compiled version of 'ollama' from the [releases page](https://github.com/ollama/ollama/releases).
