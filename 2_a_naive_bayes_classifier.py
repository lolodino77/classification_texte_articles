import pandas as pd
import numpy as np
import nltk
import re
import string
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
pd.set_option('display.max_colwidth', 600)
pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)

df = pd.read_csv('dataset_philosophy_preprocessed.csv')
# print(df)

from tabulate import tabulate
print(tabulate(df, headers = 'keys', tablefmt = 'psql'))