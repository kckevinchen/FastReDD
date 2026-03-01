import os
import json
import numpy as np
import matplotlib.pyplot as plt


markers = ['o', 's', '^', '*', 'x', 'D', 'v', '<', '>', 'p', 'h', 'H', '+', '|', '_']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


class EnsembleAnalyses:
    def __init__(self, config):
        self.config = config
        self.out_main = config["out_main"]
        self.cls_train_trials = config["cls_train_trials"]

        self.eval_template_ensemble = "eval_classifiers{trial}_{qid}_ensemble.json"
        self.eval_template_diffsize = "eval_classifiers{trial}_{qid}_diffsize.json"
        self.eval_template_incremental = "eval_classifiers{trial}_{qid}_incremental.json"
        self.eval_template_multiconformal = "eval_classifiers{trial}_{qid}_multiconformal.json"

        # self.qids = ["CQ1"]
        self.qids = ["CQ1", "CQ2"]
        # self.sizes_plot = ["s1024", "s512", "s256", "s128", "s64", "s32"]
        # self.qids = ["CQ2", "CQ3"]
        # self.sizes_plot = ["s256", "s128", "s64"]
        # self.qids = ["CQ1", "CQ2", "CQ3", "CQ4"]
        self.sizes_plot = ["s256", "s128", "s64", "s32", "s16", "s8"]
        # self.sizes_plot = ["s128", "s32", "s16", "s8"]
        # self.sizes_plot = ["s256", "s64", "s8"]

    def analyse_multiconformal(self, dataset_task):
        self.qids = self.load_json(os.path.join(self.out_main, dataset_task, "queries.json")) if self.qids is None else self.qids
        for qid in self.qids:
            size2alpha2data = {}
            for cls_train_trial in self.cls_train_trials:
                exp_path = os.path.join(self.out_main, dataset_task, "eval_classifiers", self.eval_template_multiconformal.format(trial=cls_train_trial, qid=qid))
                exp_data = self.load_json(exp_path) 
                for model_dataset_task in exp_data:
                    for model_qid in exp_data[model_dataset_task]:
                        for size in exp_data[model_dataset_task][model_qid]:
                            if size not in size2alpha2data:
                                size2alpha2data[size] = {}
                            for alpha in exp_data[model_dataset_task][model_qid][size]:
                                if alpha not in size2alpha2data[size]:
                                    size2alpha2data[size][alpha] = {}
                                for measurement in exp_data[model_dataset_task][model_qid][size][alpha]:
                                    if measurement not in size2alpha2data[size][alpha]:
                                        size2alpha2data[size][alpha][measurement] = []
                                    size2alpha2data[size][alpha][measurement].append(exp_data[model_dataset_task][model_qid][size][alpha][measurement])
            for size in ["s256"]:
                for alpha in size2alpha2data[size]:
                    for measurement in size2alpha2data[size][alpha]:
                        print(size, alpha, measurement, np.mean(size2alpha2data[size][alpha][measurement]))

    def analyses(self, dataset_task):
        """
        eval classifiers diffsize structure: 
            model_dataset_task -> model_qid -> layer -> size -> accuracy_results
        eval classifiers ensemble structure: 
            model_dataset_task -> model_qid -> size -> layer_index -> accuracy_results
            precision, recall, f1, total_errors, predicted_errors, detected_errors, undetected_errors
        """
        self.qids = self.load_json(os.path.join(self.out_main, dataset_task, "queries.json")) if self.qids is None else self.qids

        # for qid in self.qids:
        #     exp_ensemble_path = os.path.join(self.out_main, dataset_task, "eval_classifiers", self.eval_template_ensemble.format(trial=0, qid=qid))
        #     exp_diffsize_path = os.path.join(self.out_main, dataset_task, "eval_classifiers", self.eval_template_diffsize.format(trial=0, qid=qid))
        #     exp_ensemble = self.load_json(exp_ensemble_path)
        #     exp_diffsize = self.load_json(exp_diffsize_path) 
        #     for model_dataset_task in exp_ensemble:
        #         for model_qid in exp_ensemble[model_dataset_task]:
        #             for size in exp_ensemble[model_dataset_task][model_qid]:
        #                 for layers in exp_ensemble[model_dataset_task][model_qid][size]:
        #                     f1 = exp_ensemble[model_dataset_task][model_qid][size][layers]["f1"]
        #                     print(f"Model: {model_dataset_task}, Query: {model_qid}, Size: {size}, Layers: {layers}" + \
        #                           f"F1: {f1:.4f}")
        #                     for layer in layers.split("_"):
        #                         f1_single = exp_diffsize[model_dataset_task][model_qid][layer][size]["f1"]
        #                         cmp = "<=" if round(f1_single, 4) <= round(f1, 4) else ">"
        #                         print(f"\tLayer-{layer} F1: {f1_single:.4f} {cmp};", end="\t")
        #                     print()
        #             exit()

        for qid in self.qids:
            size2num2data = {}
            for cls_train_trial in self.cls_train_trials:
                exp_incremental_path = os.path.join(self.out_main, dataset_task, "eval_classifiers", self.eval_template_incremental.format(trial=cls_train_trial, qid=qid))
                exp_incremental = self.load_json(exp_incremental_path) 
                for model_dataset_task in exp_incremental:
                    for model_qid in exp_incremental[model_dataset_task]:
                        for size in exp_incremental[model_dataset_task][model_qid]:
                            for layer_acc in exp_incremental[model_dataset_task][model_qid][size]:
                                for layer in layer_acc:
                                    ensemble_num = len(layer.split("__"))
                                    if size not in size2num2data:
                                        size2num2data[size] = {}
                                    if ensemble_num not in size2num2data[size]:
                                        size2num2data[size][ensemble_num] = {}
                                    for measurement in layer_acc[layer]:
                                        if measurement not in size2num2data[size][ensemble_num]:
                                            size2num2data[size][ensemble_num][measurement] = []
                                        size2num2data[size][ensemble_num][measurement].append(layer_acc[layer][measurement])

            # # Plot F1 scores
            save_path = os.path.join("fig", dataset_task, f"ensemble_f1_{qid}.png")
            size2ensembles, size2f1, size2std = self.get_xy(size2num2data, "f1")
            self.plot_measurements(size2ensembles, size2f1, size2std, 'Number of Ensembles', 'F1 Score', save_path)
            
            # # Plot Recall
            save_path = os.path.join("fig", dataset_task, f"ensemble_recall_{qid}.png")
            size2ensembles, size2recall, size2std = self.get_xy(size2num2data, "recall")
            self.plot_measurements(size2ensembles, size2recall, size2std, 'Number of Ensembles', 'Recall', save_path)
            
            # # Plot Precision
            save_path = os.path.join("fig", dataset_task, f"ensemble_precision_{qid}.png")
            size2ensembles, size2precision, size2std = self.get_xy(size2num2data, "precision")
            self.plot_measurements(size2ensembles, size2precision, size2std, 'Number of Ensembles', 'Precision', save_path)
            
            # # Plot Error Rate
            save_path = os.path.join("fig", dataset_task, f"ensemble_error_rate_{qid}.png")
            size2ensembles, size2errorrate, size2std = self.get_xy(size2num2data, "error_rate")
            self.plot_measurements(size2ensembles, size2errorrate, size2std, 'Number of Ensembles', 'Error Rate', save_path)
            
            # # Plot Residual Error Rate
            save_path = os.path.join("fig", dataset_task, f"ensemble_residual_error_rate_{qid}.png")
            size2ensembles, size2errorrate, size2std = self.get_xy(size2num2data, "residual_error_rate")
            self.plot_measurements(size2ensembles, size2errorrate, size2std, 'Number of Ensembles', 'Residual Error Rate', save_path)
            
            # # Plot Bar Chart
            save_path = os.path.join("fig", dataset_task, f"ensemble_bar_{qid}.png")
            size2ensembles, size2predicted, _ = self.get_xy(size2num2data, "predicted_errors")  # , nums_plot=[1, 5, 10, 15, 20]
            size2ensembles, size2correct, _ = self.get_xy(size2num2data, "detected_errors")
            size2ensembles, size2undetected, _ = self.get_xy(size2num2data, "undetected_errors")
            self.bar_measurements(size2ensembles, [size2predicted, size2correct, size2undetected], 'Number of Ensembles', 'Count', save_path)
            
            # # Plot Data Pop Accuracy
            save_path = os.path.join("fig", dataset_task, f"ensemble_corrected_accuracy_{qid}.png")
            size2ensembles, size2acc, size2std = self.get_xy(size2num2data, "corrected_accuracy")
            original_accuracy = size2num2data[list(size2num2data.keys())[0]][1]["original_accuracy"][0]
            self.plot_measurements(size2ensembles, size2acc, size2std, 'Number of Ensembles', 'Corrected Accuracy', save_path,
                                   ylim=(0.7, 1.0), horizontal_line=[original_accuracy, "Original Accuracy"])

            # # Plot Voting Error Estimation
            save_path = os.path.join("fig", dataset_task, f"voting_error_estimation_{qid}.png")
            size2x_err, size2y_err, size2std_err = self.get_xy(size2num2data, "error_rate")
            size2x_est, size2y_est, size2std_est = self.get_xy(size2num2data, "error_rate_estimation")
            self.plot_line_compare(size2x_err, size2y_err, size2std_err, 
                                   size2x_est, size2y_est, size2std_est,
                                   'Number of Ensembles', 'Error Rate', save_path)

            # plt.figure(figsize=(10, 5))
            # estimates_32 = [error_s32[0], error_s32[1]]
            # C1 = -0.06
            # C2 = 0.17
            # for N in range(2, len(ensembles)):
            #     est = (3 * (N - 1) / (2 * N)) * C1 + C2
            #     estimates_32.append(est)
            # plt.plot(ensembles, error_s32, marker='^', label='s32')
            # plt.plot(ensembles, estimates_32, linestyle="--", marker='v', label='estimate')
            # plt.xlabel('Number of Ensembles')
            # plt.ylabel('Error Rate')
            # # plt.ylim(0.5, 1.0)
            # plt.legend()
            # plt.grid(True)
            # plt.show()

    def get_xy(self, size2num2data, measurement, nums_plot=None):
        size2ensembles = {}
        size2avg, size2std = {}, {}
        for size in self.sizes_plot:
            if size not in size2avg:
                size2ensembles[size], size2avg[size], size2std[size] = [], [], []
            if not nums_plot:
                nums_plot = size2num2data[size]
            for ensemble_num in nums_plot:
                if measurement not in size2num2data[size][ensemble_num]:
                    continue
                size2ensembles[size].append(int(ensemble_num))
                size2avg[size].append(float(np.mean(size2num2data[size][ensemble_num][measurement])))
                size2std[size].append(float(np.std(size2num2data[size][ensemble_num][measurement])))
        return size2ensembles, size2avg, size2std

    def plot_measurements(self, size2x, size2y, size2std, xlabel, ylabel, save_path=None, figsize=(10, 5),
                          ylim=None, horizontal_line=None):
        plt.figure(figsize=figsize)
        for i, size in enumerate(self.sizes_plot):
            plt.plot(size2x[size], size2y[size], marker=markers[i], label=size)
            plt.fill_between(size2x[size], size2y[size] - np.std(size2std[size]), size2y[size] + np.std(size2std[size]), alpha=0.2)
        if horizontal_line:
            plt.axhline(y=horizontal_line[0], color='r', linestyle='--', label=horizontal_line[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        plt.legend()
        plt.grid(True)
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)

    def plot_line_compare(self, size2x, size2y, size2std, size2x_dot, size2y_dot, size2std_dot, 
                          xlabel, ylabel, save_path=None, figsize=(10, 5), ylim=None, horizontal_line=None):
        plt.figure(figsize=figsize)
        for i, size in enumerate(self.sizes_plot):
            plt.plot(size2x[size], size2y[size], marker=markers[i], color=colors[i], linestyle='-', label=size)
            plt.plot(size2x_dot[size], size2y_dot[size], marker=markers[i], color=colors[i], linestyle='--', label=size)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ylim:
            plt.ylim(ylim[0], ylim[1])
        plt.legend()
        plt.grid(True)
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)

    def bar_measurements(self, size2x, size2y_list, xlabel, ylabel, save_path=None, figsize=(15, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        bar_width = 0.14
        n = len(self.sizes_plot)
        for i, size in enumerate(self.sizes_plot):
            xticklabels = size2x[size]
            x = np.arange(len(size2x[size]))
            predicted_errors = size2y_list[0][size]
            correct_detected = size2y_list[1][size]
            undetected_errors = size2y_list[2][size]
            ax.bar(x + (i - n / 2 + 0.5) * bar_width, predicted_errors, width=bar_width, bottom=undetected_errors, color="white", edgecolor="black")
            ax.bar(x + (i - n / 2 + 0.5) * bar_width, correct_detected, width=bar_width, bottom=undetected_errors, color=colors[i], label=f"{size}")
            ax.bar(x + (i - n / 2 + 0.5) * bar_width, undetected_errors, width=bar_width, color="gray")
        
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        # ax.grid(True)
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)

    def save_results(self, res_path, res_dict, encoding="utf-8"):
        os.makedirs(os.path.dirname(res_path), exist_ok=True)
        with open(res_path, "w", encoding=encoding) as f:
            json.dump(res_dict, f, indent=2)
    
    def load_json(self, file_path, encoding="utf-8"):
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
