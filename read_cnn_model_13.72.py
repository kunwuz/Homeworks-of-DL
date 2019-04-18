import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense

test_articles_df = pd.read_pickle('test_articles_df.pkl')

test_articles_df = test_articles_df.drop(['id'],1)

PADDING_LENGTH = 200
X = text_to_index(train_df.words)
X = pad_sequences(X, maxlen=PADDING_LENGTH)
print("X.Shape:", X.shape)
print("X.Sample:", X[0])