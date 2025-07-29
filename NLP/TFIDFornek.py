# Kütüphaneler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
english_stopwords_nltk = stopwords.words('english')


df = pd.read_csv('NLP/spam.csv',encoding='latin-1',usecols=['v1','v2'])
print(df.head())
print(df.shape)

def remove_stopwords_nltk(text):
    """NLTK ile stopwords kaldırma"""

    
    # Tokenize et
    tokens = word_tokenize(text.lower(), language='turkish')
    
    # Sadece alfabetik karakterleri al
    words = [word for word in tokens if word.isalpha()]
    
    # Stopwords'leri kaldır
    filtered_words = [word for word in words if word not in english_stopwords_nltk]
    
    return filtered_words, words
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]','',text)
    text = re.sub(r'\s+',' ',text)
    text = " ".join(remove_stopwords_nltk(text)[0])
    return text

df['v2'] = df['v2'].apply(clean_text)
# print(df.head())
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['v2'])
features = vectorizer.get_feature_names_out()
tfidfdegerleri = X.mean(axis=0).A1
df = pd.DataFrame({'word':features,'tfidf':tfidfdegerleri})
print(df.sort_values('tfidf',ascending=False).head(10))
print(df.sort_values('tfidf',ascending=True).head(10))