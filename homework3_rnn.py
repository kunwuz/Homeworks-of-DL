import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense






import time

start = time.time()
train_df = pd.read_pickle("train.pkl").sample(frac=1, random_state=1)
test_df = pd.read_pickle("test.pkl")
w2v_model = Word2Vec.load("w2v2019-02-08-19_14_43.model")

embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1

embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            trainable=False)

def text_to_index(corpus):
    new_corpus = []
    for doc in corpus:
        new_doc = []
        for word in doc:
            try:
                new_doc.append(word2idx[word])
            except:
                new_doc.append(0)
        new_corpus.append(new_doc)
    return np.array(new_corpus)

PADDING_LENGTH = 200
X = text_to_index(train_df.text)
X = pad_sequences(X, maxlen=PADDING_LENGTH)
# print("Shape:", X.shape)
# print("Sample:", X[0])
print(train_df.category)
Y = to_categorical(train_df.category)
# print("Shape:", Y.shape)
print("Sample:", Y)

for i in range(10,80,10):
    def new_model():
        model = tf.keras.Sequential()
        model.add(embedding_layer)
        model.add(tf.keras.layers.GRU(20,dropout=0.5,recurrent_dropout=0.5))
        model.add(tf.keras.layers.Dense(120+i, activation='relu'))
        model.add(tf.keras.layers.Dense(80+i/2, activation='relu'))

        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    model = new_model()
    model.summary()


    # def data_generator(data, targets, batch_size):
    #     batches = (len(data) + batch_size - 1)
    #     while(True):
    #         for i in range(batches):
    #             X = data[i*batch_size : (i+1)*batch_size]
    #             Y = targets[i*batch_size : (i+1)*batch_size]
    #             yield (X, Y)

    model.fit(x=X, y=Y, batch_size=1620, epochs=1, validation_split=0.1)
    # model.fit_generator(data_generator(X,Y,300),epochs=100,steps_per_epoch=(len(X)+300-1))
    score = model.evaluate(X,Y)

    score_set = []
    score_set.append(score)

    X_test = text_to_index(test_df.text)
    X_test = pad_sequences(X_test, maxlen=PADDING_LENGTH)
    Y_preds = model.predict(X_test)
    Y_preds_label = np.argmax(Y_preds, axis=1)
    submit = test_df[['id']]
    submit['category'] = Y_preds_label
    end = time.time()
    print(start-end)
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    submit.to_csv("submit"+str(i)+".csv", index=False)