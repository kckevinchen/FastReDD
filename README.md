# ReDD: Relational Deep Dive

**Error-Aware Queries Over Unstructured Data**

ReDD is a research project that focuses on transforming unstructured natural language documents into structured relational schemas and populating them with extracted data using Large Language Models (LLMs). The system provides error-aware query processing capabilities over unstructured data collections.

## 📋 Prerequisites

- **Python**: 3.10 or higher
- **API Keys**: Required for cloud-based LLM providers (OpenAI, DeepSeek, etc.)

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ReDD_Dev
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Configuration

Configuration files are located in the `configs/` directory. Each configuration file follows the pattern `{component}_{model}.yaml`:

- `schemagen.yaml` - Schema generation with ChatGPT
- `schemagen_deepseek.yaml` - Schema generation with DeepSeek
- `datapop_deepseek.yaml` - Data population with DeepSeek
- `datapop_qwen3.yaml` - Data population with Qwen models

### Example Configuration Structure
```yaml
default_settings: &default
  data_main: "dataset/"
  spider_path: "dataset/spider/"
  mode: "cgpt"
  out_main: "outputs/schema_gen/"
  llm_model: "gpt-4o"
```

## 📊 Usage

### Schema Generation

Generate relational schemas from unstructured documents:

```bash
# General schema generation (no specific query)
python scripts/main_schemagen.py --config configs/schemagen.yaml --exp spider_4d0_1 --api-key <YOUR_API_KEY>

# Query-specific schema generation
python scripts/main_schemagen.py --config configs/schemagen.yaml --exp spider_4d1_1 --api-key <YOUR_API_KEY>
```

### Data Population

Extract and populate data from documents into generated schemas:

```bash
# Run data population experiments
python scripts/main_datapop.py --config configs/datapop.yaml --exp spider_1 --api-key <YOUR_API_KEY>
```

### Supported LLM Providers

| Provider | Mode | Models |
|----------|------|---------|
| OpenAI | `cgpt` | GPT-5, GPT-4o |
| DeepSeek | `deepseek` | deepseek-chat |
| TogetherAI | `together` | Various models |
| SiliconFlow | `siliconflow` | Various models |
| Local | `local` | Local model inference |

## 📁 Project Structure

```
ReDD_Dev/
├── configs/             # Configuration files
├── core/                # Core functionality
│   ├── data_population/  # Data extraction and population
│   ├── schema_gen/       # Schema generation modules
│   ├── evaluation/       # Evaluation frameworks
│   ├── utils/            # Utility functions
│   └── data_loader/      # Dataset loading utilities
├── dataset/             # Dataset files (Spider, etc.)
├── prompts/             # LLM prompt templates
├── scripts/             # Entry point scripts
├── outputs/             # Generated results
└── logs/                # Execution logs
```

## 🗃️ Supported Datasets

### Spider

### 

### 

### 


## 📈 Evaluation

The project includes comprehensive evaluation tools:

```bash
# Run schema generation evaluation
python scripts/main_schemagen.py --config configs/schemagen_deepseek.yaml --exp spider_4d0_1 --eval

# Run data population evaluation
python scripts/main_datapop.py --config configs/datapop_deepseek.yaml --exp spider_1 --eval
```

Evaluation metrics include:
- **Schema Generation**: 
- **Data Population**: 
- **Error Analysis**: 

## 🔬 Demo


## 📝 Example Workflow


## 🛠️ Development

### Adding New LLM Providers

1. Create a new module in `core/data_population/` or `core/schema_gen/`, which inherits from the appropriate base class
2. Implement the class interface
3. Add configuration support in YAML files
4. Update the main scripts to handle the new provider

### Custom Datasets

1. Add dataset files to the `dataset/` directory
2. 
3. 
