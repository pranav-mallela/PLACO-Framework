# accuracies = [0.80, 0.77, 0.75, 0.72, 0.70, 0.67, 0.65, 0.63, 0.60, 0.57, 0.55, 0.53, 0.50] # must be in non-increasing order
# accuracies = [0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50] # must be in non-increasing order
# accuracies = [0.80, 0.75, 0.70, 0.65] # must be in non-increasing order
# accuracies = [0.7] * 5
# accuracies = [0.8,0.77,0.28,0.53,0.67,0.15,0.35,0.12,0.45,0.64] #variance = 0.0607
# accuracies = [0.01,0.01,0.01,0.01,0.01,0.099,0.099,0.099,0.099,0.099] #variance = 0.0022
# accuracies = [0,0,0,0,0,0,0,0] #variance = 0

accuracies = [0.35, 0.65, 0.70, 0.56, 0.66]
# accuracies = [0.89, 0.47, 0.41, 0.45, 0.62, 0.37, 0.45]
# accuracies = [0.70, 0.29, 0.36, 0.92, 0.54, 0.55, 0.52, 0.44, 0.68, 0.61]
# accuracies = [0.53, 0.36, 0.50, 0.32, 0.35, 0.56, 0.52, 0.86, 0.72, 0.31, 0.69, 0.85, 0.36]
# accuracies = [0.48, 0.94, 0.53, 0.70, 0.43, 0.58, 0.30, 0.88, 0.62 , 0.57, 0.38, 0.49, 0.39, 0.55, 0.29]

accuracies.sort(reverse=1)
NUM_HUMANS = len(accuracies)

import numpy
import os
# import deepdish
import time
import warnings
import csv

from collections.abc import Mapping
import matplotlib.pyplot as plt

from   collections import defaultdict

import pandas as pd
import numpy as np

import torch
from   torch import nn
from   torch import nn, optim
from   torch.distributions.log_normal import LogNormal
from   torch.nn.functional import softmax

import scipy.cluster.vq
import scipy
import scipy.stats
import scipy.integrate as integrate
import scipy.sparse as sp
from   scipy import optimize

from   sklearn.isotonic import IsotonicRegression
from   sklearn.utils.extmath import stable_cumsum, row_norms  
from   sklearn.metrics.pairwise import euclidean_distances
from   sklearn.metrics import confusion_matrix
from   sklearn.model_selection import GridSearchCV, StratifiedKFold
from   sklearn.linear_model import LogisticRegression
from   sklearn.model_selection import train_test_split

# import pyro
# import pyro.distributions as dist
# from   pyro.infer import MCMC, NUTS

import calibration as cal
import contextlib

from tqdm import tqdm

from policy import *

EPS = 1e-50
rng = np.random.default_rng(1234)
PROJECT_ROOT = "."
warnings.filterwarnings("ignore")
