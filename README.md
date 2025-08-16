# Multilingual Lexical Benchmark for Evaluating Large Language Models LLMs

[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Description

This projec introduces a pipeline for generating and evaluating a fine-grained, multilingual benchmark for Large Language Models (LLMs). The generated dataset is inspired by [BabelNet](https://babelnet.org/about), the multilingual semantic network and knowledge base and the structured knowledge of WordNet. The primary goal is to test an LLM's deep understanding of different lexical meaning tasks.

The benchmark currently focuses on a Multilingual Synonym Identification (MSI) task. This task challenges models by using semantically close but incorrect words (hypernyms and meronyms) as distractors, forcing a more nuanced choice than a simple vocabulary lookup.

The evaluation is performed using the industry-standard `lm-evaluation-harness` framework, ensuring reproducible and comparable results.

## Key Features

- **Automated Benchmark Generation:** Python scripts query the local BabelNet Database to generate the benchmark data in json.
- **Fine-Grained Distractors (to make the tasks more challenging):** Negative examples are generated from semantically related concepts (hypernyms, meronyms) to create challenging multiple-choice questions.
- **Multilingual:** The generation pipeline is designed to support multiple target languages. (currently 50 languages!)
- **Standardized Evaluation:** Uses the `lm-evaluation-harness` framework to test modern Hugging Face `transformers` models.

## Getting Started

Follow these steps to set up the environment and run the project.

### 1. Installation

It is recommended to use a virtual environment to avoid dependency conflicts.

```bash
# 1. Clone this repository (if applicable)
# git clone https://github.com/mahmoud-hanouneh/Cross-Lingual-Lexical-Meanings-Benchmark

# 2. Create and activate a new virtual environment
# On Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate

# 3. Install all required packages from requirements.txt
pip install -r requirements.txt
```

Install the correct version of PyTorch.
The official PyTorch website provides a tool to generate the exact installation command you need. Go to the website [PyTorch](https://pytorch.org/get-started/locally/) and select your sysetm configuration.

_NOTE!_ Local BabelNet works ONLY with Python 3.8, therefore we should select Platform to be 11.8 - CUDA 12.1 (This is the latest and most common version; if you have an older GPU, or in our case working in Python 3.8, we must select CUDA 11.8)

```bash
# the command on windows will look like
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

_IMPORTANT!_ If you get an error saying AssertionError: Torch not compiled with CUDA enabled, you might have to uninstall the old CPU torch version by running the following command and then reinstall PyTorch/

```bash
pip uninstall torch

```

### 2. API Key Setup

This API Key part is only needed for generating the data using `generate_msi_benchmark_babel_api.py` script. This generation script requires a BabelNet API key.
_IMPORTANT!_ This script is used during the early stages of the project to get familiar with BabelNet. However is NOT practical to generate huge dataset.

1.  Register for a free key at [babelnet.org/register](https://babelnet.org/register).
2.  In the root of the project folder, create a file named `.env`.
3.  Add your key to the file like this:
    ```
    MY_API_KEY="your_api_key_here"
    ```

## Usage

The project is divided into two main phases: Data Generation and Evaluation.

### Phase 1: Data Generation

This is critical part, where we generate our data to evaluate LLMs. I'm still actively working on enhancing the quality of this script. The `Multilingual Synonym Identification` still the stable and sole task until the moment as it was tested on many LLMs.

This `generate_msi_benchmark_babel_api.py` script will connect to the BabelNet API and generate the `msi_benchmark_advanced.jsonl` file in the `data/` directory.

#### BabelNet API

```bash
# Make sure your .env file is set up with your API key in the root of the project.
python generate_msi_benchmark_babel_api.py
```

Practically, generating the whole dataset using BabelNet API is not possible as shortly we'll hit the API requests limitation. Such large datasets used for benchmarks are usually generated from the local setup of BabelNet. To set up the local environment, please follow the instructions in BabelNet documentations at PYTHON API section and create the RPC Server [BabelNet Python API]('https://babelnet.org/guide')

#### BabelNet RPC Server (Setting it up using Docker & RPC Server)

```bash
# No need for API key at this case, however, a lisence to get the local copy of BabekNet dataset is required.
python scripts/generate_msi_benchmark_local.py
```

### Phase 2: Running the Evaluation

It's time to run the evaluation. Make sure the data was generated and saved correctly and quickly take a look at the quality of the questions, answers and distractors.

This step uses `lm-evaluation-harness` to test a model on the benchmark you just created. The command requires you to point to your custom task configuration files, located in the `lm_harness_tasks/` directory.

#### **Example 1: Running a Lightweight Model (on a local machine)**

This command evaluates the small `openai-community/gpt2` model. It's great for testing our setup.

```powershell
# In PowerShell (IMPORTANT!! use backticks ` for line breaks)
lm_eval `
    --model hf `
    --model_args pretrained=openai-community/gpt2,dtype=float16 `
    --tasks csi_custom_task `
    --include_path lm_harness_tasks `
    --batch_size auto `
    --output_path ./results/harness_gpt2_results.json
```

#### **Example 2: Running a State-of-the-Art Model (on HPC)**

This command evaluates the powerful `Llama-3-8B` model. This requires a powerful GPU and is intended to be run on a university High-Performance Computer. Note the use of the backslash `\` for line breaks in a Linux shell.

```bash
# In a Linux/Bash shell
lm_eval \
    --model hf \
    --model_args pretrained=meta-llama/Llama-3-8B-Instruct,dtype=bfloat16 \
    --tasks csi_custom_task \
    --include_path lm_harness_tasks \
    --batch_size auto \
    --output_path ./results/harness_llama3_results.json
```

After running, the results will be saved to the specified `.json` file in the `results/` folder and printed to the console.

## Project Structure

- **`data/`**: Contains the generated benchmark `.jsonl` files.
- **`lm_harness_tasks/`**: Contains the `.yaml` configuration files that define your custom tasks for the evaluation framework.
- **`generate_msi_benchmark_local.py`**: The script used to generate the benchmark data.
- **`requirements.txt`**: A list of all necessary Python packages.
- **`.env`**: A file for storing private API keys securely.
- **`README.md`**: This file.
