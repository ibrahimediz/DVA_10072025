# kütüphaneleri import edelim
from gensim.models import Word2Vec
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import re


# veri yükleme ve hazırlama
df = pd.read_csv("NLP/IMDB Dataset.csv")

documents = df["review"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+","",text)
    text = re.sub(r"[^\w\s]","",text)
    text = " ".join([word for word in text.split() if len(word)>2])
    return text

documents = documents.apply(clean_text)

tokenized_sentences = [simple_preprocess(doc) for doc in documents]

# model tanımlama
word2vec_model = Word2Vec(sentences=tokenized_sentences,vector_size=50,window=5,min_count=2,workers=4,sg=0)
# fastext_model = Fastext(sentences=tokenized_sentences,vector_size=100,window=5,min_count=2,workers=4,sg=0)

word_vectors = word2vec_model.wv

words = list(word_vectors.key_to_index.keys())[:300]
# fastext_vectors = fastext_model.wv

kmodel = KMeans(n_clusters=2)
kmodel.fit(word2vec_model.wv.vectors)
cluster = kmodel.labels_
pca = PCA(n_components=2)

pca_results = pca.fit_transform(word2vec_model.wv.vectors)


plt.figure(figsize=(12,8))
plt.scatter(pca_results[:,0],pca_results[:,1],c=cluster,cmap='winter')
centers = pca.transform(kmodel.cluster_centers_)
plt.scatter(centers[:,0],centers[:,1],c='red',marker='x',s=200,label='Centers',alpha=0.5)
plt.legend()

for i,word in enumerate(words):
    plt.text(pca_results[i, 0], pca_results[i, 1], word, ha='center', va='center', color='black')
plt.show()