"""
Error Detection and Correction Module

This module contains classes and utilities for error detection and correction
in data population tasks, including classifier training, validation, and
correction methods.
"""

from .classifier_structure import BinaryClassifier0, BinaryClassifier1, MultiHeadBinaryClassifier
from .codeword_correction import ClassifierValCodeCorrection
from .ensemble_analyses import EnsembleAnalyses
from .hidden_states_loader import LazyHiddenStatesDataset
from .test_classifier import ClassifierVal
from .train_classifier import ClassifierTrainer
from .voting_error_estimation import VotingErrorEstimation

__all__ = [
    "BinaryClassifier0",
    "BinaryClassifier1", 
    "MultiHeadBinaryClassifier",
    "ClassifierValCodeCorrection",
    "EnsembleAnalyses",
    "LazyHiddenStatesDataset",
    "ClassifierVal",
    "ClassifierTrainer",
    "VotingErrorEstimation"
]
