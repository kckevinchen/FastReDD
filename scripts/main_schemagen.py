import os
import yaml
import logging
import argparse

from core.utils import logging_utils
# from dataset import schema_gen_prep
from core.schema_gen import SchemaGenUnified
# from core.evaluation import EvalSchemaGen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/schemagen.yaml")
    parser.add_argument("--exp", type=str, default="spider_4d1_1")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as file: 
            config = yaml.safe_load(file)
        config = config[args.exp]
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit()

    # Setup logging with console log level from config
    console_log_level = logging_utils.get_log_level(config.get("console_log_level", "WARNING"))
    logging_utils.setup_logging(prefix_str=args.exp, log_dir="logs", console_log_level=console_log_level)
    
    # Unified instantiation for all modes
    if args.api_key:
        config["api_key"] = args.api_key
        
    try:
        schema_gen = SchemaGenUnified(config, api_key=args.api_key)
        schema_gen(config.get("exp_dataset_task_list"))
    except Exception as e:
        logging.error(f"Error running Schema Generation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # if args.eval:
    #     eval = EvalSchemaGen(config)
    #     eval(config["exp_dataset_task_list"])
