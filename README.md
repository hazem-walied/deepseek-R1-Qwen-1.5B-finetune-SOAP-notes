# Medical SOAP Note Generation with Fine-Tuned LLMs

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/medical-soap-llm/)
[![Hugging Face Models](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-blue)](https://huggingface.co/hazem74)

A comprehensive study comparing the fine-tuning and performance of two state-of-the-art language models for medical documentation generation:
- Qwen2.5-1.5B-Instruct
- DeepSeek-R1-Distill-Qwen-1.5B

## Overview

This project implements and evaluates an automated system for generating structured SOAP (Subjective, Objective, Assessment, Plan) notes from doctor-patient dialogues. The system uses fine-tuned language models to transform unstructured medical conversations into standardized clinical documentation.

## Key Features

* **Dual-Model Architecture**: Comparative analysis of Qwen-1.5B and DeepSeek-R1 models
* **LoRA Fine-Tuning**: Efficient parameter-efficient fine-tuning using LLaMA-Factory
* **Structured Output**: JSON-formatted SOAP notes with consistent schema
* **Clinical Accuracy**: Comprehensive evaluation of medical documentation quality
* **Cost-Effective**: Optimized for single T4 GPU deployment
* **Open Source**: Complete training and inference pipeline available

## Technical Approach

### 1. Data Processing Pipeline
- Utilizes the OMI-Health medical dialogue dataset
- Implements Pydantic schema for structured SOAP note generation
- Custom JSON parsing and validation for clinical data
- 90/10 train-evaluation split with reproducible shuffling

### 2. Model Fine-Tuning
- Parameter-efficient fine-tuning using LoRA adapters
- System prompt engineering for medical context
- JSON schema enforcement in model outputs
- Gradient checkpointing for memory efficiency
- Mixed precision training (FP16) for speed optimization

### 3. Evaluation Framework
- Clinical accuracy assessment
- JSON validity rate measurement
- Inference speed benchmarking
- Cost per sample analysis
- Comprehensive metrics tracking via Weights & Biases

## Repository Structure

```text
deepseek+qwen-finetune-soap/
├── docs/
│   └── results/                # Training and evaluation visualizations
├── notebooks/
│   ├── qwen_finetune_v2.py    # Qwen model fine-tuning pipeline
│   └── deepseek_finetune_v2.py # DeepSeek model fine-tuning pipeline
├── samples/
│   ├── input_example.txt      # Sample medical dialogue
│   ├── qwen_output.json       # Qwen model output example
│   └── deepseek_output.json   # DeepSeek model output example
├── src/
│   ├── qwen_finetune_v2.py    # Core fine-tuning implementation
│   ├── deepseek_finetune_v2.py # Core fine-tuning implementation
├── requirements.txt           # Project dependencies
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/medical-soap-llm.git
cd medical-soap-llm
pip install -r requirements.txt
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_soap(dialogue: str) -> str:
    """
    Generate a SOAP note from a doctor-patient dialogue.
    
    Args:
        dialogue (str): Raw medical dialogue text
        
    Returns:
        str: Structured SOAP note in JSON format
    """
    model_name = "hazem74/qwen-soap-summary-v2"  # or deepseek-soap-summary-v2
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    system_prompt = (
        "You are a medical assistant. "
        "Generate a structured SOAP note (Subjective, Objective, Assessment, Plan)."
    )
    inputs = tokenizer(
        f"System: {system_prompt}\nUser: {dialogue}",
        return_tensors="pt"
    )
    outputs = model.generate(**inputs, max_new_tokens=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Model Performance

### Quantitative Metrics

| Metric                   | Qwen-1.5B | DeepSeek-R1 |
| ------------------------ | --------- | ----------- |
| Training Time            | 6 hrs     | 7 hrs       |
| Inference Speed (tok/s)  | 23.2      | 14.8        |
| JSON Validity Rate       | 95%       | 92%         |
| Clinical Accuracy        | 89%       | 85%         |
| Cost per Sample (T4 GPU) | \$0.0011  | \$0.0019    |

### Key Findings
- Qwen-1.5B demonstrates superior inference speed and JSON validity
- DeepSeek-R1 shows slightly better clinical accuracy in complex cases
- Both models maintain high performance on single T4 GPU
- Cost-effective for production deployment

## Training Visualizations

### Qwen-1.5B
![Qwen Training](docs/results/qwen-training.png)
![Qwen Evaluation](docs/results/qwen-evaluation.png)

### DeepSeek-R1
![DeepSeek Training](docs/results/deepseek-training.png)
![DeepSeek Evaluation](docs/results/deepseek-evaluation.png)

## Available Models

* **Qwen SOAP Summary**: [hazem74/qwen-soap-summary-v2](https://huggingface.co/hazem74/qwen-soap-summary-v2)
* **DeepSeek SOAP Summary**: [hazem74/deepseek-soap-summary-v2](https://huggingface.co/hazem74/deepseek-soap-summary-v2)

## Training Reports

* **Qwen**: [WandB Report](https://api.wandb.ai/links/hazemwalied2003-cairo-university/3crlwa8v)
* **DeepSeek-R1**: [WandB Report](https://api.wandb.ai/links/hazemwalied2003-cairo-university/83xed34j)

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/name`)
5. Open a Pull Request

## License

MIT License. See [`LICENSE`](LICENSE) for details.

## Acknowledgments

* **OMI-Health** for providing the medical dialogue dataset
* **Hugging Face** for model hosting and development tools
* **LLaMA-Factory** for the fine-tuning framework
* **Google Colab** for computational resources
* **Weights & Biases** for experiment tracking





