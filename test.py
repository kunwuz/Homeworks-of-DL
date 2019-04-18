import pandas as pd
from gensim.models.word2vec import Word2Vec

model = Word2Vec.load('w2v2019-02-09-15_16_12.model')


print(len("合同  "))
print("相似詞前 100 排序:")
# for word in model.wv.vocab:
#     print(len(word))
res = model.most_similar(' 票口 ', topn=2)
for v in res:
    print(v[0],",",v[1])