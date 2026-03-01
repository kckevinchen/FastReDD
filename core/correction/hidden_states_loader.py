import os
import re
import string
import logging
import random
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, Sampler

from ..data_loader import DataLoaderBase


# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LazyHiddenStatesDataset(Dataset):
    """
    Dataset for lazy loading of hidden state data.
    During initialization, builds a sample index based on eval_output and res_data_dict. Each sample corresponds to:
      - Table sample: doc id, "table" type, no attribute name, label determined by eval_output["table"]
      - Attribute (attr) sample: doc id, "attr" type, attribute name, label determined by eval_output["attr"][attr]
    In __getitem__, loads .pt files from disk on demand based on sample information and extracts hidden state for the specified layer_index.
    Implements block caching for hidden states with limited cache size (e.g., cache at most 2 blocks).
    Each block (e.g., every 100 dids) saves results to file after first computation, then loads directly from file next time.
    """
    def __init__(self, trainer, loader: DataLoaderBase, res_data_dict, states_dir, eval_output, 
                 layer_index, sample_dids, qid, cache_batch_size=100):
        self.trainer = trainer
        self.loader = loader
        self.res_data_dict = res_data_dict          # Results loaded from res_tabular_data_xxx.json
        self.states_dir = states_dir                # Directory containing hidden state files
        self.eval_output = eval_output              # Evaluation information, determines labels
        self.layer_index = layer_index              # Layer index to extract
        self.sample_dids = set(sample_dids)         # Only process specified doc id set
        self.row_size = len(sample_dids)            # Parameter for constructing cache file names
        self.cache_batch_size = cache_batch_size    # Number of dids per cache file (default 100)

        # Build mapping from table names to token sequences based on schema_general
        self.table2tokens = {}
        for item in loader.load_schema_general():
            table = item["Schema Name"]
            self.table2tokens[table] = self.trainer.tokenizer.encode(table, add_special_tokens=False)
        # Build mapping from attributes within tables to token sequences based on schema_query
        self.table2attr2tokens = {}
        for item in loader.load_schema_query(qid):
            table = item["Schema Name"]
            if table in self.table2tokens:
                self.table2attr2tokens[table] = {}
                for attr_item in item["Attributes"]:
                    attr = attr_item["Attribute Name"]
                    self.table2attr2tokens[table][attr] = self.trainer.tokenizer.encode(attr, add_special_tokens=False)
        
        # Build sample index list
        self.samples = []
        # Note: eval_output keys are string-formatted doc ids
        for did_str, info in self.eval_output.items():
            did = int(did_str)
            if did not in self.sample_dids:
                continue
            # Table sample: label convention is 0 if eval_output[did]["table"] is True, otherwise 1
            table_label = 0 if info["table"] else 1
            self.samples.append((did, "table", None, table_label))
            # Attribute sample: each attribute is a separate sample, label follows same convention
            for attr, attr_val in info["attr"].items():
                attr_label = 0 if attr_val else 1
                self.samples.append((did, "attr", attr, attr_label))
        self.samples = sorted(self.samples, key=lambda x: x[0])
        self.all_dids = sorted([int(did_str) for did_str in self.eval_output.keys()])

        self.did_to_sampleindices = {}
        for idx, sample in enumerate(self.samples):
            did = sample[0]
            self.did_to_sampleindices.setdefault(did, []).append(idx)

        self.cache = OrderedDict()
        self.max_cache_blocks = 2

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        visited = set()  # Track visited indices to prevent infinite loops
        while True:
            if index in visited:
                logging.error(f"[{self.__class__.__name__}:__getitem__] "
                             f"All samples have no valid hidden_state! index={index}")
                raise ValueError("All samples have no valid hidden_state!")
            visited.add(index)
            
            did, sample_type, attr, label = self.samples[index]
            block_index = did // self.cache_batch_size
            # logging.info(f"[LazyDataset:__getitem__] Getting sample: did={did}, index={index}, block={block_index}")  # TODO

            # Check if current block is in cache, if so move it to the end
            if block_index in self.cache:
                doc_cache = self.cache.pop(block_index)
                self.cache[block_index] = doc_cache
            else:
                if len(self.cache) >= self.max_cache_blocks:
                    self.cache.popitem(last=False)
                doc_cache = self.load_block_cache(block_index)
                self.cache[block_index] = doc_cache
                # logging.info(f"[LazyDataset:__getitem__] Loaded cache for block {block_index}. did: {did}")  # TODO

            # Get hidden state for corresponding doc from cache
            if did not in doc_cache:
                # If did is not in cache (abnormal case), directly call computation function
                if sample_type == "table":
                    hidden_state = self.get_table_hidden_state(did)
                else:
                    hidden_state = self.get_attr_hidden_state(did, attr)
            else:
                if sample_type == "table":
                    hidden_state = doc_cache[did].get("table", None)
                elif sample_type == "attr":
                    hidden_state = doc_cache[did].get("attr", {}).get(attr, None)
                else:
                    logging.error(f"[{self.__class__.__name__}:__getitem__] "
                                 f"Unknown sample type: {sample_type}")
                    raise ValueError(f"Unknown sample type: {sample_type}")

            # If hidden_state cannot be obtained, skip and use next index
            if hidden_state is None:
                index = (index + 1) % len(self.samples)
                continue
            else:
                return hidden_state.float(), torch.tensor(label, dtype=torch.long)

    def load_block_cache(self, block_index):
        """
        Load cache for specified block:
          1. Get all dids in current block range based on all_dids
          2. Check if corresponding cache file exists, load if exists; otherwise compute and save
        Returns a dictionary with structure: { did: {"table": hidden_state, "attr": { attr: hidden_state, ... } } }
        """
        cache_file = os.path.join(self.states_dir, f"cache_hstates_{block_index}.pt")
        if os.path.exists(cache_file):
            return torch.load(cache_file)

        # Get all dids in current block based on all_dids (assuming all_dids is already sorted)
        block_start = block_index * self.cache_batch_size
        block_end = block_start + self.cache_batch_size
        block_dids = [did for did in self.all_dids if block_start <= did < block_end]

        doc_cache = {}
        for did in block_dids:
            doc_cache[did] = {}
            doc_cache[did]["table"] = self.get_table_hidden_state(did)
            doc_cache[did]["attr"] = {}
            did_str = str(did)
            if did_str in self.eval_output:
                for attr in self.eval_output[did_str]["attr"]:
                    doc_cache[did]["attr"][attr] = self.get_attr_hidden_state(did, attr)
        torch.save(doc_cache, cache_file)
        logging.info(f"[LazyDataset:load_block_cache] Cache file {cache_file} does not exist, computed and saved.")
        return doc_cache
    
    def get_did_hidden_state(self, did):
        """ Return all hidden states corresponding to the given did """
        if did not in self.did_to_sampleindices:
            logging.warning(f"did {did} does not exist in dataset")
            return []
        hidden_states = []
        for idx in self.did_to_sampleindices[did]:
            hs, _ = self[idx]
            hidden_states.append(hs)
        return torch.stack(hidden_states)

    def get_table_hidden_state(self, did):
        state_file = os.path.join(self.states_dir, f"doc-{did}-table.pt")
        if not os.path.exists(state_file):
            logging.warning(f"[LazyDataset:get_table_hidden_state] state file {state_file} does not exist (did: {did}.")
            return None
        state_dict_table = torch.load(state_file, weights_only=False)
        hidden_state = torch.stack([state_dict_table[i]["hidden_states"][self.layer_index] 
                                    for i in range(len(state_dict_table))])
        hidden_state = hidden_state.squeeze(1)  # N x D
        mean_pooled = torch.mean(hidden_state, dim=0)  # D
        max_pooled = torch.max(hidden_state, dim=0)[0]  # D
        return torch.cat([mean_pooled, max_pooled])  # 2D
        
    def get_attr_hidden_state(self, did, attr):
        state_file = os.path.join(self.states_dir, f"doc-{did}-attr-{attr}.pt")
        if not os.path.exists(state_file):
            logging.warning(f"[LazyDataset:get_attr_hidden_state] state file {state_file} does not exist (did: {did}, attr: {attr}).")
            return None
        state_dict_attr = torch.load(state_file, weights_only=False)
        hidden_state = torch.stack([state_dict_attr[i]["hidden_states"][self.layer_index] 
                                    for i in range(len(state_dict_attr))])
        hidden_state = hidden_state.squeeze(1)  # N x D
        mean_pooled = torch.mean(hidden_state, dim=0)  # D
        max_pooled = torch.max(hidden_state, dim=0)[0]  # D
        return torch.cat([mean_pooled, max_pooled])  # 2D

    def get_table_hidden_state_old(self, did):
        res_data = self.res_data_dict[str(did)]
        table_name = res_data["res"]
        if table_name not in self.table2tokens:
            # logging.info(f"[LazyDataset:get_table_hidden_state] Table name {table_name} not found in token mapping (did: {did}).")
            return None
        res_table_tokens = self.table2tokens[table_name]
        state_file = os.path.join(self.states_dir, f"doc-{did}-table.pt")
        if not os.path.exists(state_file):
            logging.warning(f"[LazyDataset:get_table_hidden_state] State file {state_file} does not exist (did: {did}).")
            return None
        state_dict_table = torch.load(state_file, weights_only=False)
        # Iterate through state_dict_table to find matching token sequence position
        for i in range(len(state_dict_table) - len(res_table_tokens) + 1):
            i_tokens = [item["token_id"] for item in state_dict_table[i:i+len(res_table_tokens)]]
            if i_tokens == res_table_tokens:
                return state_dict_table[i]["hidden_states"][self.layer_index]
        # logging.info(f"[LazyDataset:get_table_hidden_state] No matching table tokens found in file (did: {did}).")
        return None
    
    def get_attr_hidden_state_old(self, did, attr):
        res_data = self.res_data_dict[str(did)]
        table_name = res_data["res"]
        if table_name not in self.table2attr2tokens or attr not in self.table2attr2tokens[table_name]:
            # logging.info(f"[LazyDataset:get_attr_hidden_state] Attribute {attr} not found in token mapping for table {table_name} (did: {did}).")
            return None
        if attr not in res_data["data"]:
            # logging.info(f"[LazyDataset:get_attr_hidden_state] Attribute {attr} missing in res_data (did: {did}).")
            return None
        res_value = res_data["data"][attr].strip().strip(string.punctuation)
        res_value = re.sub(r'^€', '', res_value)
        res_value_tokens = self.trainer.tokenizer.encode(res_value, add_special_tokens=False)
        state_file = os.path.join(self.states_dir, f"doc-{did}-attr-{attr}.pt")
        if not os.path.exists(state_file):
            logging.warning(f"[LazyDataset:get_attr_hidden_state] State file {state_file} does not exist (did: {did}, attr: {attr}).")
            return None
        state_dict_attr = torch.load(state_file, weights_only=False)
        # Iterate through state_dict_attr to find matching token sequence
        for i in range(len(state_dict_attr) - len(res_value_tokens) + 1):
            i_tokens = [item["token_id"] for item in state_dict_attr[i:i+len(res_value_tokens)]]
            if len(res_value_tokens) > 0 and i_tokens[0] == res_value_tokens[0]:
                if i_tokens == res_value_tokens:
                    return state_dict_attr[i]["hidden_states"][self.layer_index]
                # Or compare token_text (after decoding)
                i_text = ''.join([self.trainer.decode_str(item["token_text"]) for item in state_dict_attr[i:i+len(res_value_tokens)]])
                i_text = i_text.strip().strip(string.punctuation)
                if i_text == res_value:
                    return state_dict_attr[i]["hidden_states"][self.layer_index]
        # logging.info(f"[LazyDataset:get_attr_hidden_state] No matching attribute tokens found (did: {did}, attr: {attr}).")
        return None


class SequentialOversampler(Sampler):
    """
    Custom sampler: samples each class in original order, and oversamples classes with fewer samples.
    Finally returns an index list sorted in order.
    """
    def __init__(self, indices, labels):
        self.indices = indices
        self.labels = labels
        # Group indices by label, maintaining original order
        self.class_to_indices = {}
        for idx, label in zip(indices, labels):
            self.class_to_indices.setdefault(label, []).append(idx)
        # Number of samples to sample per class: take the maximum count among all classes
        self.max_count = max(len(lst) for lst in self.class_to_indices.values())
        
        # Build oversampled index list: repeat and expand each class to reach max_count samples
        oversampled_indices = []
        for label in sorted(self.class_to_indices.keys()):
            lst = self.class_to_indices[label]
            repeat_factor = self.max_count // len(lst)
            remainder = self.max_count % len(lst)
            new_indices = lst * repeat_factor + lst[:remainder]
            oversampled_indices.extend(new_indices)
        
        # To ensure final index order matches the input train_indices, directly sort oversampled_indices
        self.oversampled_indices = sorted(oversampled_indices)
        # random.shuffle(self.oversampled_indices)

    def __iter__(self):
        return iter(self.oversampled_indices)

    def __len__(self):
        return len(self.oversampled_indices)
    