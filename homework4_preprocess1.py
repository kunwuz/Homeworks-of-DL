import pandas as pd
import os
import re

TRAINING_ARTICLE_PATH = 'Training/AllTogether_cut/'
TRAINING_ANSWER_PATH = 'Training/Answer_of_Training/'
TESTING_PATH = 'Test/'

# preprocess train_articles
categories = [dirname for dirname in os.listdir(TRAINING_ARTICLE_PATH)]
train_list = []
idx = 0
for category in categories:
    category_idx = idx
    category_path = TRAINING_ARTICLE_PATH + category
    with open(category_path, encoding='utf-8') as file:
        words = file.read().strip().replace(' ', '').split('/')
        train_list.append([idx,words])
    idx += 1

train_articles_df = pd.DataFrame(train_list, columns=['id','words'])
print(train_articles_df.sample(5))

# preprocess train_answers
answers = pd.read_table(TRAINING_ANSWER_PATH+'AllTogether_ans_0_1799.txt',header = None)
train_answers = []
for index,row in answers.iterrows():
    row = re.findall(r"\d+\.?\d*",row[0])  # find the digit
    train_answers.append(row)

train_answers_df = pd.DataFrame(train_answers, columns=['id','good','bad'])
print(train_answers_df.sample(5))


# preprocess test_articles
test_categories = [dirname for dirname in os.listdir(TESTING_PATH)]
test_list = []
test_idx = 0
for test_category in test_categories:
    test_category_idx = test_idx
    test_category_path = TESTING_PATH+test_category
    with open(test_category_path, encoding='utf-8') as file:
        test_words = file.read().strip().replace(' ', '').split('/')
        test_list.append([test_idx, test_words])
    test_idx += 1

test_articles_df = pd.DataFrame(test_list, columns=['id', 'words'])
print(test_articles_df.sample(5))

# to pickle
train_articles_df.to_pickle('train_articles_df.pkl')
train_answers_df.to_pickle('train_answers_df.pkl')
test_articles_df.to_pickle('test_articles_df.pkl')


