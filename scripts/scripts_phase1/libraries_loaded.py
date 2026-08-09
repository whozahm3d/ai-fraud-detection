import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# REQUIRED IMPORTS for neural network
from sklearn.neural_network import MLPClassifier as SklearnMLP

# Install required third-party packages in the active notebook environment
%pip install -q xgboost imbalanced-learn
from xgboost import XGBClassifier

import joblib
import gc
import psutil
import warnings
warnings.filterwarnings("ignore")

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

print("Libraries installed successfully!")



# ── Global Plot Style
plt.rcParams.update({
    'figure.dpi'      : 130,
    'font.family'     : 'DejaVu Sans',
    'axes.spines.top' : False,
    'axes.spines.right': False,
    'axes.grid'       : True,
    'grid.alpha'      : 0.35,
    'grid.linestyle'  : '--',
})
sns.set_style("darkgrid")

# Consistent colour palette used throughout
C = {
    'green'  : '#27ae60',
    'red'    : '#e74c3c',
    'blue'   : '#2980b9',
    'orange' : '#e67e22',
    'purple' : '#8e44ad',
    'teal'   : '#16a085',
    'gray'   : '#7f8c8d',
}
MODEL_COLORS = {
    'Logistic Regression': C['red'],
    'Random Forest': C['blue'],
    'XGBoost': C['green'],
    'Neural Network': C['orange'],
}
