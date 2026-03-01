# Prompt Format

All field names are defined in `core/utils/constants.py`

---

## Schemna Discovery Prompt Input/Output Format



---

## Data Population Prompt Input/Output Format

### 1. Table Assignment (`datapop_table_json.txt`)

**Input:**
```json
{
  "Document": "Document text...",
  "Schema": [
    {
      "Schema Name": "TableName1",
      "Attributes": ["attr1", "attr2"]
    },
    {
      "Schema Name": "TableName2",
      "Attributes": ["attr3", "attr4"]
    }
  ]
}
```

**Output:**
```json
{
  "Table Assignment": "TableName1"  // or "None"
}
```

---

### 2. Attribute Extraction (`datapop_attr_json.txt`)

**Input:**
```json
{
  "Document": "Document text...",
  "Schema": {
    "Schema Name": "TableName",
    "Attributes": [
      {
        "Attribute Name": "attribute_name",
        "Description": "Description of this attribute"
      }
    ]
  },
  "Target Attribute": "attribute_name"
}
```

**Output:**
```json
{
  "attribute_name": "extracted_value"  // or "None"
}
```

---

### 3. Semantic Comparison (`eval_datapop_cmp_str.txt`)

**Input:**
```json
{
  "Prediction": {
    "Attribute Name": "course_title",
    "Attribute Value": "Intro to ML"
  },
  "Ground Truth": {
    "Attribute Name": "course_title",
    "Attribute Value": "Introduction to Machine Learning"
  }
}
```

**Output:**
```json
{
  "Result": true/false,
  "Reasoning": "..."
}
```

---

### Constants Reference

All constants are defined in `core/utils/constants.py`:

```python
# ============================================================================
# Schema Field Names (used in schema_general.json, schema_query_*.json)
# ============================================================================
SCHEMA_NAME_KEY = "Schema Name"
ATTRIBUTES_KEY = "Attributes"
ATTRIBUTE_NAME_KEY = "Attribute Name"
ATTRIBUTE_DESC_KEY = "Description"

# ============================================================================
# Prompt Input Keys (used in LLM requests)
# ============================================================================
DOCUMENT_KEY = "Document"
SCHEMA_KEY = "Schema"
TARGET_ATTRIBUTE_KEY = "Target Attribute"

# ============================================================================
# LLM Output Keys (expected in LLM responses)
# ============================================================================
TABLE_ASSIGNMENT_KEY = "Table Assignment"
DATA_POPULATED_KEY = "Data Populated"
REASONING_KEY = "Reasoning"

# ============================================================================
# Evaluation Keys (used in semantic comparison)
# ============================================================================
PREDICTION_KEY = "Prediction"
GROUND_TRUTH_KEY = "Ground Truth"
ATTRIBUTE_VALUE_KEY = "Attribute Value"
# Note: ATTRIBUTE_NAME_KEY is also used here

# ============================================================================
# Result Storage Keys (used in output files)
# ============================================================================
RESULT_TABLE_KEY = "res"
RESULT_DATA_KEY = "data"
RESULT_REASON_KEY = "reason"

# ============================================================================
# Special Values
# ============================================================================
NULL_VALUE = "None"  # Used for missing/null values

# ============================================================================
# Processing Limits
# ============================================================================
DEFAULT_MAX_TABLE_RETRIES = 10
DEFAULT_MAX_ATTR_RETRIES = 10
MAX_ATTRIBUTE_VALUE_LENGTH = 100
```
