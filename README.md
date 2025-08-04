# BAMAS: Budget-Aware Multi-Agent Systems

This repository contains the implementation for the paper **"BAMAS: Structuring Budget-Aware Multi-Agent Systems"**.

## Overview

BAMAS is a framework for constructing budget-aware multi-agent systems that optimally balance performance and cost. The system automatically selects appropriate LLMs and collaboration topologies to solve tasks within specified budget constraints.

## Project Structure

```
BAMAS/
├── eapae_agent_sys/          # Core system implementation
│   ├── agents/               # Agent implementations and configurations
│   ├── planning/             # Resource planning and topology selection
│   ├── execution/            # Task execution engine
│   ├── utils/                # Utility functions and API wrappers
│   └── data_processing/      # Dataset loaders and preprocessing
├── configs/                  # Configuration files
│   ├── 0_agent_library*.yml  # Agent specifications
│   ├── 1_role_compatibility.yml # Role definitions
│   ├── 2_collaboration_patterns.json # Topology configurations
│   ├── 3_training_params*.yml # Training parameters
│   └── 4_*_dataset_config.yml # Dataset configurations
├── scripts/                  # Training data generation scripts
├── experiments/              # Experiment execution scripts
├── main_train_offline.py     # Main training script
└── data/                     # Dataset storage (not included)
```

## Key Components

### Core System (`eapae_agent_sys/`)

- **Agents**: Modular agent implementations for different roles (planner, executor, critic)
- **Planning**: Budget-constrained resource allocation and collaboration pattern selection
- **Execution**: Multi-agent task execution engine with various topology support
- **Utils**: Configuration management, LLM API wrappers, and evaluation utilities
- **Data Processing**: Dataset loaders for MATH, MBPP, and GSM8K benchmarks

### Configuration System

- **Agent Libraries**: Define available LLMs and their capabilities
- **Collaboration Patterns**: Specify different multi-agent topologies (linear, star, feedback, planner-driven)
- **Training Parameters**: Control reinforcement learning and optimization settings
- **Dataset Configs**: Manage benchmark-specific configurations

### Experiment Framework

- **Training**: Offline reinforcement learning for topology selection
- **Evaluation**: Performance assessment across different budgets and datasets

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys in configs/secrets.yml
```

### 2. Training

```bash
# Train the topology selection policy
python main_train_offline.py --dataset math --config configs/3_training_params_math.yml
```

### 3. Evaluation

```bash
# Run experiments with trained model
python experiments/main_experiment_math.py --model_path models/trained_model.pth
```

## Supported Datasets

- **MATH**: Mathematical reasoning problems
- **MBPP**: Python code generation tasks  
- **GSM8K**: Grade school math word problems

## Key Features

- **Budget-aware optimization**: Automatically balances performance and cost
- **Flexible topologies**: Supports linear, star, feedback, and planner-driven collaboration patterns
- **Modular design**: Easy to extend with new agents and topologies
- **Multi-dataset support**: Works across different task domains

## Citation

If you use this code in your research, please cite our paper:

To be released

## License

This project is licensed under the terms specified in the LICENSE file.

## Notes

- The `data/` directory should contain the benchmark datasets
- API keys for LLM providers should be configured in `configs/secrets.yml`