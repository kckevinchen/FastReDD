import os
import re
import tqdm
import json
import string
import logging
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler, random_split
if torch.cuda.is_available():
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from ..utils import constants
from ..utils.constants import PATH_TEMPLATES
from .classifier_structure import BinaryClassifier0 as BinaryClassifier
from .classifier_structure import MultiHeadBinaryClassifier, diversity_loss
from .hidden_states_loader import LazyHiddenStatesDataset, SequentialOversampler
from ..data_loader import create_data_loader


# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassifierTrainer:
    def __init__(self, config):
        self.config = config

        if not torch.cuda.is_available():
            logging.error(f"[{self.__class__.__name__}:__init__] CUDA is not available. Exiting...")
            exit()
        # torch.manual_seed(config["torch_seed"])

        if config["mode"] not in ["local"]:
            return
        
        self.llm_model_path = config["llm_model_path"]
        self.llm_model_name = config["llm_model"]
        self.param_str = config["res_param_str"]
        self.tokenizer = None
        self.load_tokenizer()

        self.train_data_percentage = float(config["trainer"]["train_percentage"])
        self.batch_size = int(config["trainer"]["batch_size"])
        self.epochs = int(config["trainer"]["epochs"])
        self.early_stop_patience = int(config["trainer"]["early_stop_patience"])
        self.learning_rate = float(config["trainer"]["learning_rate"])

        self.qids = None  # TODO
        self.num_layers = config.get("num_layers", 28)
        self.hidden_size = config.get("hidden_size", 2048)
        self.num_heads = config.get("num_heads", 3)
        self.train_size = config.get("train_size", 32)
        self.exp_layers = config.get("exp_layers", list(range(0, self.num_layers)))
        self.all_dids = []

        self.valid_token_pattern = re.compile(r'[a-zA-Z0-9]')
        logging.info(f"[{self.__class__.__name__}] Using device: {device}")

    def __call__(self, dataset_task_list=None):
        if dataset_task_list is None:
            if "exp_dataset_task_list" in self.config:
                dataset_task_list = self.config["exp_dataset_task_list"]
            else:
                logging.warning(f"[{self.__class__.__name__}] No dataset tasks specified, using default SPIDER_DATASET_TASK_LIST")
                dataset_task_list = constants.SPIDER_DATASET_TASK_LIST
        total_tasks = len(dataset_task_list)
        logging.info(f"[{self.__class__.__name__}] Error Classifier Training {total_tasks} tasks: {dataset_task_list}")

        for dataset_task in dataset_task_list:
            # Separate data path and output path
            data_path = os.path.join(self.config.get("data_main", self.config["out_main"]), dataset_task)
            out_root = os.path.join(self.config["out_main"], dataset_task)
            os.makedirs(out_root, exist_ok=True)
            logging.info(f"[{self.__class__.__name__}:__init__] Start processing dataset: data_path={data_path}, out_root={out_root}")
            self.process_dataset(data_path, out_root)
    
    def process_dataset(self, data_path, out_root):
        """ Process all queries and documents in a dataset. """
        loader = create_data_loader(
            data_path=data_path,
            loader_type=self.config.get("data_loader_type", "sqlite"),
            loader_config=self.config.get("data_loader_config", {})
        )
        query_dict = loader.load_query_dict()
        if not self.qids:
            self.qids = list(query_dict.keys())
        for qid in self.qids:
            logging.info(f"[{self.__class__.__name__}:process_dataset] Start processing query {qid}")
            res_data_path = os.path.join(out_root, PATH_TEMPLATES.data_population_result(qid, self.param_str))
            res_data_dict = self.load_processed_res(res_data_path)
            eval_path = os.path.join(out_root, PATH_TEMPLATES.eval_result(qid, self.param_str))
            eval_output = self.load_json(eval_path)
            states_dir = os.path.join(out_root, PATH_TEMPLATES.hidden_states_dir(qid, self.param_str))

            self.all_dids = sorted([int(did) for did in eval_output.keys()])
            logging.info(f"[{self.__class__.__name__}:process_dataset] Loading hidden states done for query {qid}, {self.num_layers} layers each token.")
            self.process_documents(loader, out_root, qid, res_data_dict, states_dir, eval_output)
    
    def process_documents(self, data_loader, out_root, qid, res_data_dict, states_dir, eval_output):
        cls_train_trial = 0
        model_save_dir = os.path.join(out_root, "classifiers", f"classifiers{cls_train_trial}_{qid}")
        while os.path.exists(model_save_dir):
            cls_train_trial += 1
            model_save_dir = os.path.join(out_root, "classifiers", f"classifiers{cls_train_trial}_{qid}")
        os.makedirs(model_save_dir, exist_ok=True)
        logging.info(f"[{self.__class__.__name__}:process_dataset] Model save directory: {model_save_dir}")
        for layer_index in tqdm.tqdm(self.exp_layers):
            logging.info(f"[{self.__class__.__name__}:process_dataset] Start training for layer {layer_index} ...")
            size = f's{self.train_size}'
            training_dids = list(random.sample(self.all_dids, self.train_size))
            logging.info(f"[{self.__class__.__name__}:process_dataset] Training with size {size}. training_dids: {training_dids}")
            train_loader_s, val_loader_s = self.prepare_data(
                data_loader, res_data_dict, states_dir, eval_output, layer_index, training_dids, qid)
            final_model_save_path_s = os.path.join(model_save_dir, f"final_model_layer{layer_index}_{size}.pt")
            best_model_save_path_s = os.path.join(model_save_dir, f"best_model_layer{layer_index}_{size}.pt")
            self.train(train_loader_s, val_loader_s, final_model_save_path_s, best_model_save_path_s)
            # self.train_multihead(train_loader_s, val_loader_s, final_model_save_path_s, best_model_save_path_s)

    def prepare_data(self, data_loader, res_data_dict, states_dir, eval_output, layer_index, training_dids, qid, oversample=True):
        dataset = LazyHiddenStatesDataset(self, data_loader, res_data_dict, states_dir, eval_output, layer_index, training_dids, qid)
        total_samples = len(dataset)
        train_size = int(self.train_data_percentage * total_samples)
        val_size = total_samples - train_size
        
        # 使用顺序索引进行划分，而非 random_split（它会随机打乱索引）
        indices = list(range(total_samples))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # 直接利用 dataset.samples 获取标签，避免多次 __getitem__ 调用
        train_labels = [dataset.samples[i][3] for i in train_indices]
        val_labels = [dataset.samples[i][3] for i in val_indices]
        logging.info(f"[{self.__class__.__name__}:prepare_data] Train set class counts: {np.bincount(train_labels)}")
        logging.info(f"[{self.__class__.__name__}:prepare_data] Validation set class counts: {np.bincount(val_labels)}")

        if oversample:
            logging.info(f"[{self.__class__.__name__}:prepare_data] Oversampling ...")
            train_sampler = SequentialOversampler(train_indices, train_labels)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler, shuffle=False)
            oversampled_train_labels = [dataset.samples[i][3] for i in train_sampler.oversampled_indices]
            logging.info(f"[{self.__class__.__name__}:prepare_data] Train set class counts after oversampling: {np.bincount(oversampled_train_labels)}")
        
        logging.info(f"[{self.__class__.__name__}:prepare_data] Total Samples: {total_samples}, Train Size: {train_size}, Val Size: {val_size}")
        return train_loader, val_loader

    def train(self, train_loader, val_loader, final_model_save_path, best_model_save_path):
        model = BinaryClassifier(self.hidden_size).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        best_val_acc = 0.0
        no_improve_epochs = 0
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            total_train_samples = 0
            for inputs, targets in train_loader:
                if inputs.size(0) == 1:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze(1)
                # print(inputs.shape, targets.shape, outputs.shape)
                loss = criterion(outputs, targets.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                total_train_samples += inputs.size(0)
            
            avg_train_loss = total_loss / total_train_samples

            model.eval()
            val_loss = 0
            correct = 0
            total_val_samples = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs).squeeze(1)
                    loss = criterion(outputs, targets.float())
                    val_loss += loss.item() * inputs.size(0)
                    preds = (torch.sigmoid(outputs) > 0.5).int()
                    correct += (preds == targets.int()).sum().item()
                    total_val_samples += targets.size(0)

            avg_val_loss = val_loss / total_val_samples
            val_acc = correct / total_val_samples

            scheduler.step(val_acc)

            # early stop
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                no_improve_epochs = 0
                # store best
                self.save_model(model, best_model_save_path)
            else:
                no_improve_epochs += 1

            # if epoch >= 20 and no_improve_epochs >= self.early_stop_patience:
            #     logging.info(f"Early stopping at epoch {epoch}")
            #     break

            if (epoch + 1) % 1 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | "
                             f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

        self.save_model(model, final_model_save_path)

    def train_multihead(self, train_loader, val_loader, final_model_save_path, best_model_save_path):
        model = MultiHeadBinaryClassifier(self.hidden_size, self.num_heads).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

        best_val_acc = 0.0
        no_improve_epochs = 0
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            total_train_samples = 0
            for inputs, targets in train_loader:
                if inputs.size(0) == 1:  # Skip single-sample batches
                    continue
                inputs = inputs.to(device)
                targets = targets.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)  # List of outputs from each head
                # Compute CE Loss for each head
                ce_loss = sum(criterion(output.squeeze(), targets) for output in outputs) / self.num_heads
                # Compute Diversity Loss
                div_loss = diversity_loss(outputs, beta=1.0)
                # Total Loss
                total_loss_batch = ce_loss + div_loss
                total_loss_batch.backward()
                optimizer.step()
                total_loss += total_loss_batch.item() * inputs.size(0)
                total_train_samples += inputs.size(0)
            
            avg_train_loss = total_loss / total_train_samples

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total_val_samples = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device).float()
                    outputs = model(inputs)
                    # Average predictions across heads for validation
                    avg_output = torch.mean(torch.stack([output.squeeze() for output in outputs]), dim=0)
                    loss = criterion(avg_output, targets)
                    val_loss += loss.item() * inputs.size(0)
                    preds = (torch.sigmoid(avg_output) > 0.5).int()
                    correct += (preds == targets.int()).sum().item()
                    total_val_samples += targets.size(0)

            avg_val_loss = val_loss / total_val_samples
            val_acc = correct / total_val_samples

            scheduler.step(val_acc)

            # Early stopping and model saving
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                no_improve_epochs = 0
                self.save_model(model, best_model_save_path)
            else:
                no_improve_epochs += 1

            # if epoch >= 20 and no_improve_epochs >= self.early_stop_patience:
            #     logging.info(f"Early stopping at epoch {epoch}")
            #     break

            if (epoch + 1) % 1 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | "
                             f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")

        self.save_model(model, final_model_save_path)

    def load_tokenizer(self):
        """ Load the LLM model. """
        if not os.path.exists(self.llm_model_path):
            logging.info(f"[{self.__class__.__name__}:load_tokenizer] Downloading model ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
            self.tokenizer.save_pretrained(self.llm_model_path)
        else:
            logging.info(f"[{self.__class__.__name__}:load_tokenizer] Loading model from local ...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path, trust_remote_code=True)

    def save_model(self, model, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
    
    def get_res_schema(self, res_schema_path):
        """ Load Result Schema from <res_schema_path> """
        if not os.path.exists(res_schema_path):
            logging.error(f"[{self.__class__.__name__}:get_res_schema] Result Schema not found: {res_schema_path}")
            exit()
        return self.load_json(res_schema_path)
    
    def save_results(self, res_path, res_dict, encoding="utf-8"):
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w", encoding=encoding) as f:
            json.dump(res_dict, f, indent=2)
    
    def load_json(self, file_path, encoding="utf-8"):
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)

    def load_processed_res(self, res_path):
        """ Load Processed Results from <res_path> """
        res_dict = dict()
        if os.path.exists(res_path):
            res_dict = self.load_json(res_path)
        return res_dict
