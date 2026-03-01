import yaml
import logging
import argparse

from core.utils import logging_utils
from core.data_population import DataPop, DataPopLocal


# Supported API modes for DataPop
API_MODES = {"cgpt", "deepseek", "together", "siliconflow", "gemini"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    
    mode = config.get("mode", "deepseek")
    if mode in API_MODES:
        # Use unified DataPop for all API-based modes
        datapop = DataPop(config, api_key=args.api_key)
    elif mode == "local":
        # Use DataPopLocal for local GPU models
        datapop = DataPopLocal(config)
    else:
        logging.error(f"Invalid mode '{mode}'. Supported modes: {API_MODES | {'local'}}")
        exit(1)

    datapop(config["exp_dataset_task_list"])
