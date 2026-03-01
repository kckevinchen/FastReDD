# ============================================================================
# Dataset Configuration
# ============================================================================

SPIDER_DATASET_TASK_LIST = ["college_2/course-teaches-instructor"]

SPIDER_DATASET_LIST = ["store_1", "wine_1", "college_2", "flight_4"]  # , "soccer_1"

SPIDER_DATASET2TASKS = {
  "store_1": ["albums", "employees", "customers", "invoices", "tracks"],
  "wine_1": ["grapes", "appellations", "wine"],
  "soccer_1": ["player_attributes", "team_attributes", "player", "league", "team"],
  "bike_1": [],
  "apartment_rentals": [],
  "college_2": ["classroom", "section", "takes", "department", "course", "teaches", "student", "instructor",
                "advisor", "prereq"],
  "flight_4": ["routes", "airports", "airlines"]
}

EXP_DATASET2TASKS = {
  ## spider dataset
  "store_1": ["albums", "customers", "invoices", "tracks",
              "customers-invoices", "customers-employees", "albums-tracks",
              "employees-customers-invoices"],
  "wine_1": ["wine-grapes", "wine-appellations"],  
  # "grapes", "appellations", "wine"
  "soccer_1": ["player_attributes", "team_attributes", "player", "league", "team", "all"],
  "bike_1": ["all"],
  "apartment_rentals": ["all"],
  "college_2": ["course-section", "instructor-teaches", "course-teaches", "instructor-department",
                "department-student-instructor", "course-teaches-instructor", "course-section-classroom"],
  # "classroom", "section", "takes", "department", "course", "teaches", "student", "instructor", "advisor", "prereq"
  "flight_4": ["routes-airports", "routes-airlines","routes-airports-airlines"],
  # "routes", "airports", "airlines"
  ## bird dataset
  "california_schools": ["all"],
  "debit_card_specializing": ["all"],
  "student_club": ["all"],
  ## galois dataset
  "fortune": ["all"],
  "premierleague": ["all"]
}

ASSIGN_THRESHOLD = 5


# ============================================================================
# Configuration Files
# ============================================================================

API_KEYS_FILENAME = "api_keys.json"


# ============================================================================
# Data Population Constants
# ============================================================================

# Null value representation
NULL_VALUE = "None"

# Schema field names
SCHEMA_NAME_KEY = "Schema Name"
ATTRIBUTES_KEY = "Attributes"
ATTRIBUTE_NAME_KEY = "Attribute Name"
ATTRIBUTE_DESC_KEY = "Description"

# Prompt input field names
DOCUMENT_KEY = "Document"
SCHEMA_KEY = "Schema"
TARGET_ATTRIBUTE_KEY = "Target Attribute"

# Prompt output field names
TABLE_ASSIGNMENT_KEY = "Table Assignment"
DATA_POPULATED_KEY = "Data Populated"
REASONING_KEY = "Reasoning"

# Result entry field names
RESULT_TABLE_KEY = "res"
RESULT_DATA_KEY = "data"
RESULT_REASON_KEY = "reason"

# Evaluation field names
PREDICTION_KEY = "Prediction"
GROUND_TRUTH_KEY = "Ground Truth"
ATTRIBUTE_VALUE_KEY = "Attribute Value"

# Processing limits
DEFAULT_MAX_TABLE_RETRIES = 10
DEFAULT_MAX_ATTR_RETRIES = 10
MAX_ATTRIBUTE_VALUE_LENGTH = 100


# ============================================================================
# File Path Templates
# ============================================================================

class FilePathTemplates:
    """
    Centralized management of file path templates used across the project.
    
    Usage:
        from core.utils.constants import PATH_TEMPLATES
        
        # Generate a file path
        file_path = PATH_TEMPLATES.data_population_result(qid="Q1", param_str="ds_v1")
        # Returns: "res_tabular_data_Q1_ds_v1.json"
        
        # Access raw template string
        template = PATH_TEMPLATES.DATA_POPULATION_RESULT
        # Returns: "res_tabular_data_{qid}_{param_str}.json"
    """
    
    # ============ Data Population Templates ============
    DATA_POPULATION_RESULT = "res_tabular_data_{qid}_{param_str}.json"
    HIDDEN_STATES_DIR = "hidden_states_{qid}_{param_str}"
    
    # ============ Evaluation Templates ============
    EVAL_RESULT = "eval_{qid}_{param_str}.json"
    EVAL_COMPARISON_CACHE = "cmp_results.json"
    EVAL_NAME_MAPPING = "name_map_{qid}.json"
    
    # ============ Schema Generation Templates ============
    # General schemas (no specific query)
    SCHEMA_GENERAL = "schema_general_{param_str}.json"
    SCHEMA_GENERAL_ORIGINAL = "schema_general_{param_str}.original.json"
    SCHEMA_GENERAL_CURRENT = "schema_general_{param_str}.current.json"
    SCHEMA_GENERAL_FINAL = "schema_general_{param_str}.final.json"  # TODO: not used yet
    SCHEMA_GEN_RESULT_GENERAL = "res_{param_str}.json" 
    
    # General schemas (for specific query)
    SCHEMA_GENERAL_QUERY = "schema_general_{qid}_{param_str}.json"
    
    # Query-specific schemas
    SCHEMA_QUERY_TAILORED = "res_schema_{qid}_{param_str}.json"
    SCHEMA_QUERY_ORIGINAL = "res_schema_{qid}_{param_str}.original.json"
    SCHEMA_QUERY_CURRENT = "res_schema_{qid}_{param_str}.current.json"
    SCHEMA_QUERY_FINAL = "res_schema_{qid}_{param_str}.final.json"  # TODO: not used yet
    SCHEMA_GEN_RESULT_QUERY = "res_{qid}_{param_str}.json"
    
    # ============ Helper Methods ============
    
    @classmethod
    def data_population_result(cls, qid: str, param_str: str) -> str:
        """Generate data population result filename."""
        return cls.DATA_POPULATION_RESULT.format(qid=qid, param_str=param_str)
    
    @classmethod
    def hidden_states_dir(cls, qid: str, param_str: str) -> str:
        """Generate hidden states directory name."""
        return cls.HIDDEN_STATES_DIR.format(qid=qid, param_str=param_str)
    
    @classmethod
    def eval_result(cls, qid: str, param_str: str) -> str:
        """Generate evaluation result filename."""
        return cls.EVAL_RESULT.format(qid=qid, param_str=param_str)
    
    @classmethod
    def eval_comparison_cache(cls) -> str:
        """Generate semantic comparison cache filename."""
        return cls.EVAL_COMPARISON_CACHE
    
    @classmethod
    def eval_name_mapping(cls, qid: str) -> str:
        """Generate name mapping filename for a specific query."""
        return cls.EVAL_NAME_MAPPING.format(qid=qid)
    
    @classmethod
    def schema_query_tailored(cls, qid: str, param_str: str) -> str:
        """Generate tailored query-specific schema filename."""
        return cls.SCHEMA_QUERY_TAILORED.format(qid=qid, param_str=param_str)
    
    @classmethod
    def schema_query_original(cls, qid: str, param_str: str) -> str:
        """Generate original query-specific schema filename."""
        return cls.SCHEMA_QUERY_ORIGINAL.format(qid=qid, param_str=param_str)
    
    @classmethod
    def schema_query_current(cls, qid: str, param_str: str) -> str:
        """Generate current query-specific schema filename (updated after each document)."""
        return cls.SCHEMA_QUERY_CURRENT.format(qid=qid, param_str=param_str)
    
    @classmethod
    def schema_gen_result_query(cls, qid: str, param_str: str) -> str:
        """Generate schema generation result filename for a specific query."""
        return cls.SCHEMA_GEN_RESULT_QUERY.format(qid=qid, param_str=param_str)
    
    @classmethod
    def schema_general(cls, param_str: str, qid: str = None) -> str:
        """Generate general schema filename (optionally query-specific)."""
        if qid is None:
            return cls.SCHEMA_GENERAL.format(param_str=param_str)
        else:
            return cls.SCHEMA_GENERAL_QUERY.format(qid=qid, param_str=param_str)
    
    @classmethod
    def schema_general_original(cls, param_str: str) -> str:
        """Generate original general schema filename."""
        return cls.SCHEMA_GENERAL_ORIGINAL.format(param_str=param_str)
    
    @classmethod
    def schema_general_current(cls, param_str: str) -> str:
        """Generate current general schema filename (updated after each document)."""
        return cls.SCHEMA_GENERAL_CURRENT.format(param_str=param_str)
    
    @classmethod
    def schema_gen_result_general(cls, param_str: str) -> str:
        """Generate schema generation result filename for general schema."""
        return cls.SCHEMA_GEN_RESULT_GENERAL.format(param_str=param_str)


# Global instance for easy access
PATH_TEMPLATES = FilePathTemplates()