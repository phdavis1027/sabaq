# stdlib
import random
import warnings
import time
import os

# 3rd party
import pymongo

from transformers import (
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    CamembertTokenizerFast,
    CamembertForTokenClassification,
    TrainingArguments,
    Trainer,
)

import datasets
from datasets import DatasetDict, Dataset, load_metric

import numpy as np

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import torch

from googletrans import Translator

from nltk.translate import meteor_score
import nltk

from sklearn.metrics.pairwise import cosine_similarity

import spacy
import spacy.util as spacy_util

# local

import util
