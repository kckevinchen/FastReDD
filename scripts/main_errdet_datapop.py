import yaml
import logging
import argparse

from core.utils import logging_utils
from core.correction import ClassifierTrainer, ClassifierVal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train error classifier and detect errors for data population")
    parser.add_argument("--config", type=str, default="configs/datapop_ds7b.yaml")
    parser.add_argument("--exp", type=str, default="spider_1")
    parser.add_argument("--mode", type=str, choices=["train", "detect"], default="train",
                        help="Mode: 'train' to train classifier, 'detect' to detect errors")
    parser.add_argument("--test-mode", type=str, default="diffsize",
                        help="Test mode for error detection: 'diffsize', 'ensemble', 'incremental', 'errorbound', 'multiconformal', 'multihead'")
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
    
    # Check if mode is local (required for ClassifierTrainer and ClassifierVal)
    mode = config.get("mode", "deepseek")
    if mode != "local":
        logging.error(f"[main_errdet_datapop] ClassifierTrainer and ClassifierVal only support 'local' mode, but got '{mode}'")
        exit(1)

    if args.mode == "train":
        # Train error classifier
        logging.info(f"[main_errdet_datapop] Starting error classifier training...")
        try:
            trainer = ClassifierTrainer(config)
        except Exception as e:
            logging.error(f"[main_errdet_datapop] Error initializing ClassifierTrainer: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

        # Run training
        try:
            trainer(config.get("exp_dataset_task_list"))
            logging.info(f"[main_errdet_datapop] Error classifier training completed successfully")
        except Exception as e:
            logging.error(f"[main_errdet_datapop] Error running classifier training: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    elif args.mode == "detect":
        # Detect errors using trained classifier
        logging.info(f"[main_errdet_datapop] Starting error detection...")
        try:
            val = ClassifierVal(config)
        except Exception as e:
            logging.error(f"[main_errdet_datapop] Error initializing ClassifierVal: {e}")
            import traceback
            traceback.print_exc()
            exit(1)

        # Run error detection
        try:
            # Check required config parameters
            if "model_dataset_task_list" not in config:
                logging.error(f"[main_errdet_datapop] Missing required config parameter: 'model_dataset_task_list'")
                exit(1)
            if "test_dataset_task" not in config:
                logging.error(f"[main_errdet_datapop] Missing required config parameter: 'test_dataset_task'")
                exit(1)
            
            val(config["model_dataset_task_list"], config["test_dataset_task"], test_mode=args.test_mode)
            logging.info(f"[main_errdet_datapop] Error detection completed successfully with test_mode='{args.test_mode}'")
        except Exception as e:
            logging.error(f"[main_errdet_datapop] Error running error detection: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
