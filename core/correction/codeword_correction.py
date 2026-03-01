import os
import logging
import random 
import numpy as np

from .test_classifier import ClassifierVal
from ..data_loader import create_data_loader
from ..utils.constants import PATH_TEMPLATES


class ClassifierValCodeCorrection(ClassifierVal):
    def __init__(self, config):
        super().__init__(config) 
        self.out_main = config["out_main"]
        self.cls_train_trials = config["cls_train_trials"]

        self.exp_layers = config.get("exp_layers", [20, 21, 22, 23, 24, 25, 26])
        self.train_size = config.get("train_size", 256)
        self.classifier_threshold = config.get("classifier_threshold", 0.5)
        self.ensemble_sample_trials = config.get("ensemble_sample_trials", 1)
        
        self.min_test_did = self.max_recal_did
        self.max_test_did = 99999
        self.test_dids = None

    def correction_marginal_voting(self, model_dataset_task_list, test_dataset_task):
        """
        Evaluate the performance of the classifiers on the test set with marginal voting correction.
        """
        for self.cls_train_trial in self.cls_train_trials:
            # model_dict: model_dataset_task -> model_qid -> layer_index -> training_size -> model_path
            model_dict = self._get_model_dict(model_dataset_task_list)
            logging.info(f"[{self.__class__.__name__}:correction_marginal_voting] **START EXP** test_dataset_task: {test_dataset_task}; model_dict: {model_dict}")
            results = self._apply_code_correction_common(test_dataset_task, model_dict, mode="marginalvoting")
            for qid in results:
                exp_save_path = os.path.join(self.out_main, test_dataset_task, "eval_classifiers", f"eval_correction{self.cls_train_trial}_{qid}marginalvoting.json")
                self.save_results(exp_save_path, results[qid])
            logging.info(f"[{self.__class__.__name__}:correction_marginal_voting] **END EXP** test_dataset_task: {test_dataset_task}; results: {results}")

    def _apply_code_correction_common(self, test_dataset_task, model_dict, mode="marginalvoting"):
        """
        Apply the correction methods. Depending on the mode:
        - mode "marginalvoting": apply marginal voting correction.
        """
        # Separate data path and output path
        data_path = os.path.join(self.config.get("data_main", self.out_main), test_dataset_task)
        out_root = os.path.join(self.out_main, test_dataset_task)
        loader = create_data_loader(
            data_path=data_path,
            loader_type=self.config.get("data_loader_type", "sqlite"),
            loader_config=self.config.get("data_loader_config", {})
        )
        query_dict = loader.load_query_dict()

        results = {}
        for qid in query_dict:
            logging.info(f"[{self.__class__.__name__}:_apply_code_correction_common] Processing qid: {qid}")
            eval_output = self.load_json(os.path.join(out_root, PATH_TEMPLATES.eval_result(qid, self.param_str)))
            self.all_dids = [int(did_str) for did_str in eval_output]
            self.test_dids = [did for did in self.all_dids if self.min_test_did <= did < self.max_test_did]
            classifier_outputs = self._get_classifier_outputs(loader, model_dict, out_root, qid, self.all_dids, eval_output)
            gt_all = {}
            for did in self.all_dids:
                gt_all[did] = 0 if eval_output[str(did)]["final"] else 1

            if mode == "xxx":
                pass
            else:  # mode == "marginalvoting"
                results[qid] = self._correction_marginalvoting(qid, gt_all, classifier_outputs)
        return results

    def _correction_marginalvoting(self, test_qid, test_gt: dict, classifier_outputs):
        """
        Evaluate the performance of the classifiers on the test set with marginal voting correction.
        Structure of results:
            model_dataset_task -> model_qid -> size -> layers -> results
            model_dataset_task -> model_qid -> layer_index -> training_size -> did -> outputs
        """
        results = {}
        for model_dataset_task in classifier_outputs:
            results[model_dataset_task] = {}
            for model_qid in classifier_outputs[model_dataset_task]:
                if model_qid != test_qid:  # TODO: remove?
                    continue
                results[model_dataset_task][model_qid] = {}
                size = f's{self.train_size}'
                results[model_dataset_task][model_qid][size] = {}
                for _ in range(self.ensemble_sample_trials):
                    layers = self._random_layers()
                    marginal_name = "_".join(str(l) for l in layers)
                    cls_outputs = [classifier_outputs[model_dataset_task][model_qid][str(layer)][size] for layer in layers]
                    res = self._evaluate_correction_marginalvoting(test_gt, cls_outputs)
                    results[model_dataset_task][model_qid][size][marginal_name] = res
        return results

    def _random_layers(self, size=7):
        return random.sample(self.exp_layers, size)

    def _evaluate_correction_marginalvoting(self, test_gt, cls_outputs):
        results = {}
        for self.classifier_threshold in [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]:

            cls_preds, all_labels = [], []
            for did in self.test_dids:
                votes = []
                for layer_i in range(len(cls_outputs)):
                    outputs = np.array(cls_outputs[layer_i][str(did)])
                    vote = 1 if (outputs >= self.classifier_threshold).any() else 0
                    votes.append(vote)
                cls_preds.append(votes)
                all_labels.append(test_gt[did])

            disagree2preds, disagree2labels = {0: [], 1: [], 2: [], 3: []}, {0: [], 1: [], 2: [], 3: []}
            for gt, preds in zip(all_labels, cls_preds):
                majority_vote = 1 if sum(preds) >= (len(preds) / 2) else 0
                disagree = min(preds.count(1), preds.count(0))
                if disagree > 3:
                    disagree = 3
                disagree2preds[disagree].append(majority_vote)
                disagree2labels[disagree].append(gt)

            disagree2errorrate, disagree2count = {}, {}
            for disagree in disagree2preds:
                disagree2count[disagree] = len(disagree2preds[disagree])
                if not disagree2labels[disagree]:
                    error_rate = 0
                else:
                    labels = np.array(disagree2labels[disagree])
                    preds = np.array(disagree2preds[disagree])
                    fn_disagree = np.sum((labels == 1) & (preds == 0))
                    n_disagree = len(labels)
                    error_rate = fn_disagree / n_disagree
                    # error_rate = 1 - float(accuracy_score(labels, preds))
                disagree2errorrate[disagree] = round(error_rate, 4)

            # abstain threshold
            threshold2extracostrate = {}
            threshold2correctedaccuracy = {}
            n = len(all_labels)
            p = sum(1 for gt in all_labels if gt == 1)
            for threshold in [0, 1, 2]:
                fp, fn = 0, 0
                for disagree in range(0, 4):
                    labels = np.array(disagree2labels[disagree])
                    preds = np.array(disagree2preds[disagree])
                    if disagree <= threshold:
                        fp += np.sum((labels == 0) & (preds == 1))
                        fn += np.sum((labels == 1) & (preds == 0))
                    else:
                        fp += np.sum(labels == 0)

                extra_cost_rate = fp / (n - p) if (n - p) > 0 else 0
                post_correction_accuracy = 1 - fn / n if n > 0 else 0
                threshold2extracostrate[threshold] = round(extra_cost_rate, 4)
                threshold2correctedaccuracy[threshold] = round(post_correction_accuracy, 4)

            res = {
                "disagree2errorrate": disagree2errorrate,
                "disagree2count": disagree2count,
                "threshold2extracostrate": threshold2extracostrate,
                "threshold2correctedaccuracy": threshold2correctedaccuracy,
            }
            results[self.classifier_threshold] = res
        return results