# Cheat Sheet

## API Keys Configuration

Create `api_keys.json` in project root:

```json
{
    "GEMINI_API_KEY": "your-gemini-key",
    "DEEPSEEK_API_KEY": "sk-your-deepseek-key",
    "OPENAI_API_KEY": "sk-your-openai-key",
    "SILICONFLOW_API_KEY": "sk-your-siliconflow-key",
    "TOGETHER_API_KEY": "your-together-key"
}
```

Note: `api_keys.json` is gitignored. Alternatively, set via `--api-key` or environment variables.

## Schema Generation

### ReDDv0

General Schema: 

```bash
# College dataset - ReDDv0
python scripts/main_schemagen.py --config configs/schema_gen/siliconflow_qwen30B.yaml --exp college_2_5d0_reddv0
# Wine dataset - ReDDv0
python scripts/main_schemagen.py --config configs/schema_gen/siliconflow_qwen30B.yaml --exp wine_1_5d0_reddv0
# Flight dataset - ReDDv0
python scripts/main_schemagen.py --config configs/schema_gen/siliconflow_qwen30B.yaml --exp flight_4_5d0_reddv0
# Soccer dataset - ReDDv0
python scripts/main_schemagen.py --config configs/schema_gen/siliconflow_qwen30B.yaml --exp soccer_1_5d0_reddv0
# Bike dataset - ReDDv0 (no adaptive sampling)
python scripts/main_schemagen.py --config configs/schema_gen/siliconflow_qwen30B.yaml --exp bike_5d0_reddv0
# Apartment Rentals dataset - ReDDv0
python scripts/main_schemagen.py --config configs/schema_gen/siliconflow_qwen30B.yaml --exp apartment_rentals_5d0_reddv0
```

Query-Specific Schema: 

```bash
# (To be added)
```

## Data Population

### ReDDv0

SiliconFlow API Qwen3-30B-A3B-Instruct-2507

```bash
# College dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp college_reddv0
# Wine dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp wine_reddv0
# Flight dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp flight_reddv0
# Soccer dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp soccer_reddv0
# Bike dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp bike_reddv0
# Apartment Rentals dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp apartment_rentals_reddv0

# Bird: California Schools dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp california_schools_reddv0
# Bird: Debit Card Specializing dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp debit_card_specializing_reddv0
# Bird: Student Club dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp student_club_reddv0

# Galois: Fortune dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp fortune_reddv0
# Galois: Premier League dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp premierleague_reddv0

# FDA: 510k Device Submission dataset - ReDDv0
python scripts/main_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp fda_reddv0
```

Local Model Qwen3-30B-A3B-Instruct-2507

```bash
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp college_reddv0
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp wine_reddv0
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp flight_reddv0
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp soccer_reddv0
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp bike_reddv0
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp apartment_rentals_reddv0
# Bird datasets
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp california_schools_reddv0
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp debit_card_specializing_reddv0
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp student_club_reddv0
# Galois datasets
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp fortune_reddv0
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp premierleague_reddv0
# FDA dataset
python scripts/main_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp fda_reddv0
```

## Evaluation (Data Population)

### ReDDv0

SiliconFlow API Qwen3-30B-A3B-Instruct-2507

```bash
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp college_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp wine_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp flight_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp soccer_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp bike_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp apartment_rentals_reddv0
# Bird datasets
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp california_schools_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp debit_card_specializing_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp student_club_reddv0
# Galois datasets
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp fortune_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp premierleague_reddv0
# FDA dataset
python scripts/main_eval_datapop.py --config configs/data_pop/siliconflow_qwen30B.yaml --exp fda_reddv0
```

Local Model Qwen3-30B-A3B-Instruct-2507

```bash
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp college_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp wine_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp flight_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp soccer_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp bike_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp apartment_rentals_reddv0
# Bird datasets
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp california_schools_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp debit_card_specializing_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp student_club_reddv0
# Galois datasets
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp fortune_reddv0
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp premierleague_reddv0
# FDA dataset
python scripts/main_eval_datapop.py --config configs/data_pop/local_qwen30B.yaml --exp fda_reddv0
```

## Classifier Training (Data Population)

### ReDDv0

Local Model Qwen3-30B-A3B-Instruct-2507 (Note: Classifier training only supports local mode)

```bash
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp college_reddv0
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp wine_reddv0
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp flight_reddv0
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp soccer_reddv0
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp bike_reddv0
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp apartment_rentals_reddv0
# Bird datasets
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp california_schools_reddv0
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp debit_card_specializing_reddv0
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp student_club_reddv0
# Galois datasets
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp fortune_reddv0
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp premierleague_reddv0
# FDA dataset
python scripts/main_train_classifier.py --config configs/data_pop/local_qwen30B.yaml --exp fda_reddv0
```

