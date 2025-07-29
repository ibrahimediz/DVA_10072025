from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
metinler = ["Bugün hava çok güzel ve güneşli.",
    "Yarın hava yağmurlu olacak.",
    "Bugün hava yağmurlu olacak mı?",
    "Bugün parka gittik ve mogulu eğlendik",
    "Yarında parka gitmek istiyorum",
    "Bugün havuza gitmek istiyoruz."]

unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_X = unigram_vectorizer.fit_transform(metinler)
unigram_features = unigram_vectorizer.get_feature_names_out()

bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
bigram_X = bigram_vectorizer.fit_transform(metinler)
bigram_features = bigram_vectorizer.get_feature_names_out()

trigram_vectorizer = CountVectorizer(ngram_range=(3, 3))
trigram_X = trigram_vectorizer.fit_transform(metinler)
trigram_features = trigram_vectorizer.get_feature_names_out()

# print(unigram_features)
# print(bigram_features)
# print(trigram_features)

# print(unigram_X.toarray())
# print(bigram_X.toarray())
# print(trigram_X.toarray())

unigramdf = pd.DataFrame(unigram_X.toarray(), columns=unigram_features)
bigramdf = pd.DataFrame(bigram_X.toarray(), columns=bigram_features)
trigramdf = pd.DataFrame(trigram_X.toarray(), columns=trigram_features)

print(unigramdf)
print(bigramdf)
print(trigramdf)