import os
import json
import ijson
import re
import nltk
import spacy
import community
import seaborn
import collections
import shutil
import runpy
import scipy
import sys
import re
import gensim
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import networkx as nx

from nltk import flatten
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from natsort import natsorted
from cdlib import viz, algorithms, evaluation, readwrite
from datetime import datetime
from scipy import sparse
from collections import Counter, defaultdict
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from gensim.test.utils import common_corpus, common_dictionary

from sentence_transformers import SentenceTransformer
from gensim.corpora.dictionary import Dictionary
from gensim import models

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

stemmer = nltk.stem.SnowballStemmer('english')
nlp_eng = spacy.load('en_core_web_sm')
sp = spacy.load('en_core_web_sm')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')