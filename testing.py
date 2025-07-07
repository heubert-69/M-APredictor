#Used to merge and fix the dataset
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.metrics import AUC(), ROC()
from sklearn.metrics import f1_score, classification_matrix


test_1 = pd.read_csv("Filtered Data/test1.csv")
t1_answer = pd.read_csv("Filtered Data/test_answers.csv")
