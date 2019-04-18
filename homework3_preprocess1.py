import numpy as np
import pandas as pd
import os
from gensim.models.word2vec import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense

TRAINING_PATH = 'training/'
TESTING_PATH = 'testing/'

categories = [dirname for dirname in os.listdir(TRAINING_PATH) if dirname[-3:] != 'cut']

category2idx = {'Japan_Travel': 0, 'KR_ENTERTAIN': 1, 'Makeup': 2, 'Tech_Job': 3, 'WomenTalk': 4, 'babymother': 5, 'e-shopping': 6, 'graduate': 7, 'joke': 8, 'movie': 9}

train_list = []
for category in categories:
    category_idx = category2idx[category]
    category_path = TRAINING_PATH + category + '_cut/'

    for filename in os.listdir(category_path):
        filepath = category_path + filename

        with open(filepath, encoding='utf-8') as file:
            words = file.read().strip().split(' / ')
            train_list.append([words, category_idx])

train_df = pd.DataFrame(train_list, columns=['text', 'category'])
print(train_df.sample(5))
train_df.to_pickle('train.pkl')

test_list = []

for idx in range(1000):
    filepath = TESTING_PATH + str(idx) + '.txt'

    with open(filepath, encoding='utf-8') as file:
        words = file.read().strip().split(' / ')
        test_list.append([idx, words])

test_df = pd.DataFrame(test_list, columns = ['id', 'text'])
print(test_df.sample(5))
test_df.to_pickle('test.pkl')

