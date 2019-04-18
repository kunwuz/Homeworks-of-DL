import pandas as pd
from gensim.models.word2vec import Word2Vec



train_df = pd.read_pickle("train.pkl")
test_df = pd.read_pickle("test.pkl")
corpus = pd.concat([train_df.text, test_df.text]).sample(frac=1)

print(corpus[0])


model = Word2Vec(corpus, size=250, iter=10, workers=3, min_count=5, negative=3, max_vocab_size=None, window=5)

def most_similar(w2v_model, words, topn=10):
    similar_df = pd.DataFrame()
    for word in words:
        try:
            similar_words = pd.DataFrame(w2v_model.wv.most_similar(word, topn=topn),
                                         columns=[word, 'cos'])
            similar_df = pd.concat([similar_df, similar_words], axis=1)
        except:
            print(word, 'not found in word2vec model database')
    return similar_df
print(most_similar(model,['請問', '已經', '收到', '一份', '紙本', 'offer', '了還', '繼續', '部門', '面試', '之前']))
import time
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
model.save("w2v"+now+'.model')
