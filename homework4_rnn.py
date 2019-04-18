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
train_articles_df = pd.read_pickle('train_articles_df.pkl')
train_answers_df = pd.read_pickle('train_answers_df.pkl')
train_articles_df = train_articles_df.drop(['id'],1)
train_answers_df = train_answers_df.drop(['id'],1)
w2v_model = Word2Vec.load("w2v2019-02-10-23_19_06.model")

train_df = pd.concat([train_answers_df,train_articles_df],axis=1)
train_df = train_df.sample(frac=1)
# print(train_df.head)

embedding_matrix = np.zeros((len(w2v_model.wv.vocab.items()) + 1, w2v_model.vector_size))
word2idx = {}

vocab_list = [(word, w2v_model.wv[word]) for word, _ in w2v_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1

embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                            output_dim=256,
                            input_length=200,
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
X = text_to_index(train_df.words)
X = pad_sequences(X, maxlen=PADDING_LENGTH)
# print("X.Shape:", X.shape)
# print("X.Sample:", X[0])
Y = train_df.drop(['words'],1)
# print("Y.Shape:", Y.shape)
# print("Y.Sample:", Y.head)
Y['good'] = pd.to_numeric(Y.good, downcast='signed')
Y['bad'] = pd.to_numeric(Y.bad, downcast='signed')

test_articles_df = pd.read_pickle('test_articles_df.pkl')
submit = test_articles_df[['id']]
# print(submit)
test_articles_df = test_articles_df.drop(['id'],1)


for i in range(10,80,10):
    def new_model():
        model = tf.keras.Sequential()
        model.add(embedding_layer)
        model.add(tf.keras.layers.GRU(20,dropout=0.5,recurrent_dropout=0.5))
        model.add(tf.keras.layers.Dense(120+i, activation='relu'))
        model.add(tf.keras.layers.Dense(80+i/2, activation='relu'))

        model.add(tf.keras.layers.Dense(2))

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    model = new_model()



    model.compile(optimizer='adam',loss='mae')
    model.summary()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 20)
    model.fit(X, Y, batch_size=525, epochs=5000, validation_split=0.125, callbacks=[early_stopping])
    model.save('first_try_rnn_'+str(i)+'.model')

    model.load_weights('first_try_rnn_'+str(i)+'.model')
    X_test = text_to_index(test_articles_df.words)
    X_test = pad_sequences(X_test, maxlen=PADDING_LENGTH)
    Y_preds = model.predict(X_test)
    print(Y_preds)
    Y_preds_label = np.around(Y_preds)
    print(Y_preds_label)
    submit['good'] = Y_preds_label[:,0]
    submit['bad'] = Y_preds_label[:,1]
    submit[submit<0] = (-submit)



    end = time.time()
    print(end-start)

    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    submit.to_csv("submit"+str(i)+".csv", index=False)



