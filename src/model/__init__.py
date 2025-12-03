"""
IDS Classifier Model Package

This package contains the machine learning model for network intrusion detection.

Modules:
    train: Model training pipeline
    inference: Production inference interface
    preprocessing: Shared preprocessing utilities
    verify_artifacts: Artifact verification tool
    evaluate: Model evaluation script
"""

from .preprocessing import log2_transform
from .inference import IDSClassifier
import os
import pandas as pd
import numpy as np
from pathlib import Path

__all__ = ['IDSClassifier', 'log2_transform']


#*****