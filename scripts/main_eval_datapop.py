import yaml
import logging
import argparse

from core.utils import logging_utils
from core.evaluation import EvalDataPop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate data population results")
    parser.add_argument("--config", type=str, default="configs/datapop_ds7b.yaml")
    parser.add_argument("--exp", type=str, default="spider_1")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    try:
        with open(args.config, "r") as file: 
            config = yaml.safe_load(file)
        config = config[args.exp]
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit(1)

    # Setup logging with console log level from config
    console_log_level = logging_utils.get_log_level(config.get("console_log_level", "WARNING"))
    logging_utils.setup_logging(prefix_str=args.exp, log_dir="logs", console_log_level=console_log_level)
    
    # Initialize evaluator
    eval_config = config.get("eval", {})
    if "committee" in eval_config or eval_config.get("mode") in ["deepseek", "cgpt"]:
        eval = EvalDataPop(config, api_key=args.api_key)
    else:
        logging.error(f"Invalid eval mode {eval_config.get('mode', 'not specified')}")
        exit(1)

    # Run evaluation
    try:
        eval(config.get("exp_dataset_task_list"))
    except Exception as e:
        logging.error(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
