# LLMCheck

🌳 Tree-based LLM self-consistency evaluation through transform-reverse operations.

## Introduction

LLMCheck is a Python package that evaluates Large Language Models (LLMs) through a novel tree-based approach. It tests an LLM's consistency by applying a series of transform-reverse operations (like translation) and measuring how well the model maintains semantic consistency across these transformations.

## Features

- 🤖 Support for both API-based (via LiteLLM) and local LLM evaluation
- 🌲 Tree-based evaluation structure with customizable depth and branching
- 📊 Multiple similarity metrics for consistency evaluation
- 🛠️ Robust error handling and retry mechanisms
- 📈 Rich visualization of evaluation results
- 🧰 CLI tool for easy evaluation

## Usage

Simply use ```llmcheck``` as a cli tool. There are two parameters:

```
```

## LLM Options

As we use`litellm` to connect to the LLM, you will have access to all the models that `litellm` supports.

## Ollama

You are recommended to use 'ollama' to access modest-sized open-source LLMs. If you do not have permission to install software on your machine, you can download a pre-compiled version of 'ollama' from the [releases page](https://github.com/ollama/ollama/releases).
