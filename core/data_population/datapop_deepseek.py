import logging

from .datapop_api import DataPopAPI
from ..utils.prompt_utils import PromptDeepSeek, get_api_key

__all__ = ["DataPopDeepSeek"]


class DataPopDeepSeek(DataPopAPI):

    def __init__(self, config, api_key=None):
        super().__init__(config)
        
        self.mode = config["mode"]
        if self.mode in ["deepseek"]:
            self.api_key = get_api_key(config, self.mode, api_key)

            self.param_str = config["res_param_str"]
            self.prompt_table = PromptDeepSeek(
                    self.mode,
                    config["prompts"]["prompt_table"],
                    llm_model=config["llm_model"], 
                    api_key=self.api_key
                )
            self.prompt_attr = PromptDeepSeek(
                    self.mode,
                    config["prompts"]["prompt_attr"],
                    llm_model=config["llm_model"], 
                    api_key=self.api_key
                )
            
            logging.info(f"[{self.__class__.__name__}:__init__] Initialized with model: {config['llm_model']}")