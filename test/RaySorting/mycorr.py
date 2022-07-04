import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Read file into a Pandas dataframe
from pandas import DataFrame, read_csv
f = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
df = read_csv(f)
df=df[0:10]
df