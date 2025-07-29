"""
# Gelişmiş Word Embedding Yöntemleri - Word2Vec, GloVe ve FastText

## Word Embedding Yöntemleri Karşılaştırması

### 1. Word2Vec
- **Geliştirici**: Google (Mikolov et al., 2013)
- **Yaklaşım**: Neural network tabanlı
- **Modeller**: CBOW (Continuous Bag of Words) ve Skip-gram
- **Avantajlar**: Hızlı, etkili, anlamsal ilişkileri iyi yakalar
- **Dezavantajlar**: OOV (Out-of-Vocabulary) kelimelerle sorun

### 2. GloVe (Global Vectors)
- **Geliştirici**: Stanford (Pennington et al., 2014)
- **Yaklaşım**: Global istatistiksel bilgi + yerel bağlam
- **Özellik**: Co-occurrence matrisini kullanır
- **Avantajlar**: Global ve yerel bilgiyi birleştirir
- **Dezavantajlar**: Büyük veri setlerinde yavaş

### 3. FastText
- **Geliştirici**: Facebook AI (Bojanowski et al., 2017)
- **Yaklaşım**: Subword information (alt-kelime bilgisi)
- **Özellik**: Karakter n-gramları kullanır
- **Avantajlar**: OOV kelimelerle başa çıkabilir, morfolojik zengin diller için ideal
- **Dezavantajlar**: Daha fazla bellek kullanır

## Türkçe için Öneriler
- **FastText**: Türkçe'nin morfolojik yapısı için en uygun
- **Word2Vec**: Hız ve basitlik için
- **GloVe**: Büyük korpuslar için global bilgi gerektiğinde

## Kullanım Alanları
- Metin sınıflandırma
- Duygu analizi
- Makine çevirisi
- Bilgi çıkarımı
- Öneri sistemleri
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import re
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme için
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

print("=== GELİŞMİŞ WORD EMBEDDING YÖNTEMLERİ ===\n")

# Daha kapsamlı Türkçe örnek metin
ornek_metin = """
Türkiye Cumhuriyeti'nin başkenti Ankara'dır. İstanbul ise nüfus bakımından en büyük şehirdir.
Karadeniz, Akdeniz ve Ege denizleri Türkiye'yi çevreler. Boğaziçi İstanbul'u ikiye böler.
Anadolu ve Trakya olmak üzere iki ana bölgesi vardır. Türk mutfağı dünyaca ünlüdür.
Kebap, döner, baklava, künefe ve Türk kahvesi meşhur lezzetlerdir. İstanbul'da Galata Kulesi,
Ayasofya ve Sultanahmet Camii önemli tarihi yapılardır. Kapadokya'da peri bacaları ve
yeraltı şehirleri bulunur. Pamukkale'nin beyaz travertenleri çok güzeldir.
Türkiye dört mevsim yaşanır. İlkbahar çiçeklerle, yaz sıcakla, sonbahar yapraklarla,
kış karla gelir. Türk halkı misafirperverdir. Çay içmek Türk kültürünün önemli parçasıdır.
Futbol en popüler spordur. Galatasaray, Fenerbahçe ve Beşiktaş büyük takımlardır.
Türk edebiyatında Nazim Hikmet, Orhan Pamuk gibi önemli yazarlar vardır.
"""

print("Örnek Metin:")
print(ornek_metin[:200] + "...")
print("\n" + "="*70 + "\n")

# Metin ön işleme fonksiyonu
def preprocess_turkish_text(text):
    """
    Türkçe metin için ön işleme
    """
    # Küçük harfe çevir
    text = text.lower()
    
    # Türkçe karakterleri koru, diğer noktalama işaretlerini kaldır
    text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
    
    # Çoklu boşlukları tek boşluğa çevir
    text = re.sub(r'\s+', ' ', text)
    
    # Kelimelere böl ve kısa kelimeleri filtrele
    words = [word for word in text.strip().split() if len(word) > 2]
    
    return words

# Metni ön işle
words = preprocess_turkish_text(ornek_metin)
print(f"Ön işlenmiş kelime sayısı: {len(words)}")
print(f"Benzersiz kelime sayısı: {len(set(words))}")
print(f"İlk 15 kelime: {words[:15]}")
print("\n" + "="*70 + "\n")

# =============================================================================
# 1. WORD2VEC İMPLEMENTASYONU
# =============================================================================

print("1. WORD2VEC İMPLEMENTASYONU")
print("-" * 50)

class Word2VecTurkish:
    """
    Türkçe için basitleştirilmiş Word2Vec implementasyonu
    Skip-gram modeli kullanır
    """
    
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=10):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_vectors = {}
        self.vocabulary_size = 0
        
    def build_vocabulary(self, words):
        """
        Kelime dağarcığını oluşturur
        """
        word_counts = Counter(words)
        
        # Min count filtresi uygula
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.min_count]
        
        # Kelime-indeks eşleştirmesi
        for i, word in enumerate(set(filtered_words)):
            self.word_to_index[word] = i
            self.index_to_word[i] = word
            
        self.vocabulary_size = len(self.word_to_index)
        print(f"Kelime dağarcığı boyutu: {self.vocabulary_size}")
        
    def generate_training_data(self, words):
        """
        Skip-gram için eğitim verisi oluşturur
        """
        training_data = []
        
        for i, target_word in enumerate(words):
            if target_word not in self.word_to_index:
                continue
                
            # Pencere içindeki bağlam kelimelerini al
            start = max(0, i - self.window)
            end = min(len(words), i + self.window + 1)
            
            for j in range(start, end):
                if i != j and words[j] in self.word_to_index:
                    training_data.append((target_word, words[j]))
        
        return training_data
    
    def train(self, words):
        """
        Word2Vec modelini eğitir (basitleştirilmiş versiyon)
        """
        print("Word2Vec eğitimi başlıyor...")
        
        # Kelime dağarcığını oluştur
        self.build_vocabulary(words)
        
        # Eğitim verisi oluştur
        training_data = self.generate_training_data(words)
        print(f"Eğitim örneği sayısı: {len(training_data)}")
        
        # Rastgele vektörlerle başla
        for word in self.word_to_index:
            self.word_vectors[word] = np.random.uniform(-0.5, 0.5, self.vector_size)
        
        # Basit eğitim döngüsü (gerçek Word2Vec daha karmaşık)
        for epoch in range(self.epochs):
            total_loss = 0
            
            for target_word, context_word in training_data:
                # Basit güncelleme (gerçekte negative sampling kullanılır)
                target_vec = self.word_vectors[target_word]
                context_vec = self.word_vectors[context_word]
                
                # Cosine similarity hesapla
                similarity = np.dot(target_vec, context_vec) / (
                    np.linalg.norm(target_vec) * np.linalg.norm(context_vec) + 1e-8
                )
                
                # Basit gradient güncelleme
                learning_rate = 0.01
                error = 1 - similarity
                
                # Vektörleri güncelle
                self.word_vectors[target_word] += learning_rate * error * context_vec
                self.word_vectors[context_word] += learning_rate * error * target_vec
                
                total_loss += error ** 2
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss:.4f}")
        
        # Vektörleri normalize et
        for word in self.word_vectors:
            norm = np.linalg.norm(self.word_vectors[word])
            if norm > 0:
                self.word_vectors[word] = self.word_vectors[word] / norm
        
        print("Word2Vec eğitimi tamamlandı!")
    
    def find_similar_words(self, word, top_k=5):
        """
        Verilen kelimeye en benzer kelimeleri bulur
        """
        if word not in self.word_vectors:
            return []
        
        target_vector = self.word_vectors[word]
        similarities = []
        
        for other_word, other_vector in self.word_vectors.items():
            if other_word != word:
                similarity = np.dot(target_vector, other_vector)
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def word_analogy(self, word1, word2, word3):
        """
        Kelime analojisi: word1 - word2 + word3 = ?
        Örnek: kral - erkek + kadın = kraliçe
        """
        if not all(word in self.word_vectors for word in [word1, word2, word3]):
            return []
        
        # Vektör aritmetiği
        result_vector = (self.word_vectors[word1] - 
                        self.word_vectors[word2] + 
                        self.word_vectors[word3])
        
        # En benzer kelimeyi bul
        similarities = []
        for word, vector in self.word_vectors.items():
            if word not in [word1, word2, word3]:
                similarity = np.dot(result_vector, vector) / (
                    np.linalg.norm(result_vector) * np.linalg.norm(vector) + 1e-8
                )
                similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]

# Word2Vec örneği
print("Word2Vec modeli eğitiliyor...")
word2vec_model = Word2VecTurkish(vector_size=50, window=3, epochs=5)
word2vec_model.train(words)

# Benzer kelimeleri test et
test_words = ['istanbul', 'türkiye', 'güzel']
for test_word in test_words:
    if test_word in word2vec_model.word_vectors:
        similar = word2vec_model.find_similar_words(test_word)
        print(f"\n'{test_word}' kelimesine benzer kelimeler:")
        for word, similarity in similar:
            print(f"  {word}: {similarity:.3f}")

print("\n" + "="*70 + "\n")

# =============================================================================
# 2. GLOVE İMPLEMENTASYONU
# =============================================================================

print("2. GLOVE (GLOBAL VECTORS) İMPLEMENTASYONU")
print("-" * 50)

class GloVeTurkish:
    """
    Türkçe için basitleştirilmiş GloVe implementasyonu
    """
    
    def __init__(self, vector_size=100, window=5, epochs=10, learning_rate=0.05):
        self.vector_size = vector_size
        self.window = window
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.word_to_index = {}
        self.index_to_word = {}
        self.cooccurrence_matrix = {}
        self.word_vectors = {}
        self.vocabulary_size = 0
        
    def build_vocabulary(self, words):
        """
        Kelime dağarcığını oluşturur
        """
        word_counts = Counter(words)
        
        for i, word in enumerate(set(words)):
            self.word_to_index[word] = i
            self.index_to_word[i] = word
            
        self.vocabulary_size = len(self.word_to_index)
        print(f"GloVe kelime dağarcığı boyutu: {self.vocabulary_size}")
    
    def build_cooccurrence_matrix(self, words):
        """
        Co-occurrence matrisini oluşturur
        """
        print("Co-occurrence matrisi oluşturuluyor...")
        
        # Matrisi başlat
        for word1 in self.word_to_index:
            self.cooccurrence_matrix[word1] = defaultdict(float)
        
        # Pencere içindeki birlikte geçmeleri say
        for i, target_word in enumerate(words):
            if target_word not in self.word_to_index:
                continue
                
            start = max(0, i - self.window)
            end = min(len(words), i + self.window + 1)
            
            for j in range(start, end):
                if i != j and words[j] in self.word_to_index:
                    # Mesafeye göre ağırlıklandır
                    distance = abs(i - j)
                    weight = 1.0 / distance
                    
                    self.cooccurrence_matrix[target_word][words[j]] += weight
        
        # Simetrik hale getir
        for word1 in self.cooccurrence_matrix:
            for word2 in self.cooccurrence_matrix[word1]:
                if word2 in self.cooccurrence_matrix:
                    avg_count = (self.cooccurrence_matrix[word1][word2] + 
                               self.cooccurrence_matrix[word2][word1]) / 2
                    self.cooccurrence_matrix[word1][word2] = avg_count
                    self.cooccurrence_matrix[word2][word1] = avg_count
    
    def weighting_function(self, x, x_max=100, alpha=0.75):
        """
        GloVe ağırlıklandırma fonksiyonu
        """
        if x < x_max:
            return (x / x_max) ** alpha
        else:
            return 1.0
    
    def train(self, words):
        """
        GloVe modelini eğitir
        """
        print("GloVe eğitimi başlıyor...")
        
        # Kelime dağarcığını oluştur
        self.build_vocabulary(words)
        
        # Co-occurrence matrisini oluştur
        self.build_cooccurrence_matrix(words)
        
        # Rastgele vektörlerle başla
        W = np.random.uniform(-0.5, 0.5, (self.vocabulary_size, self.vector_size))
        W_tilde = np.random.uniform(-0.5, 0.5, (self.vocabulary_size, self.vector_size))
        b = np.random.uniform(-0.5, 0.5, self.vocabulary_size)
        b_tilde = np.random.uniform(-0.5, 0.5, self.vocabulary_size)
        
        # Eğitim döngüsü
        for epoch in range(self.epochs):
            total_loss = 0
            count = 0
            
            for word1 in self.cooccurrence_matrix:
                i = self.word_to_index[word1]
                
                for word2, x_ij in self.cooccurrence_matrix[word1].items():
                    if x_ij > 0:
                        j = self.word_to_index[word2]
                        
                        # GloVe loss hesapla
                        weight = self.weighting_function(x_ij)
                        
                        # Tahmin
                        prediction = np.dot(W[i], W_tilde[j]) + b[i] + b_tilde[j]
                        
                        # Loss
                        loss = weight * (prediction - np.log(x_ij)) ** 2
                        total_loss += loss
                        
                        # Gradient güncelleme (basitleştirilmiş)
                        error = prediction - np.log(x_ij)
                        grad_factor = weight * error * self.learning_rate
                        
                        # Vektörleri güncelle
                        W[i] -= grad_factor * W_tilde[j]
                        W_tilde[j] -= grad_factor * W[i]
                        b[i] -= grad_factor
                        b_tilde[j] -= grad_factor
                        
                        count += 1
            
            if epoch % 2 == 0:
                avg_loss = total_loss / count if count > 0 else 0
                print(f"Epoch {epoch + 1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Final vektörleri oluştur (W + W_tilde)
        for word in self.word_to_index:
            i = self.word_to_index[word]
            self.word_vectors[word] = (W[i] + W_tilde[i]) / 2
            
            # Normalize et
            norm = np.linalg.norm(self.word_vectors[word])
            if norm > 0:
                self.word_vectors[word] = self.word_vectors[word] / norm
        
        print("GloVe eğitimi tamamlandı!")
    
    def find_similar_words(self, word, top_k=5):
        """
        Benzer kelimeleri bulur
        """
        if word not in self.word_vectors:
            return []
        
        target_vector = self.word_vectors[word]
        similarities = []
        
        for other_word, other_vector in self.word_vectors.items():
            if other_word != word:
                similarity = np.dot(target_vector, other_vector)
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# GloVe örneği
print("GloVe modeli eğitiliyor...")
glove_model = GloVeTurkish(vector_size=50, window=3, epochs=5)
glove_model.train(words)

# Benzer kelimeleri test et
for test_word in ['türkiye', 'güzel']:
    if test_word in glove_model.word_vectors:
        similar = glove_model.find_similar_words(test_word)
        print(f"\n'{test_word}' kelimesine benzer kelimeler (GloVe):")
        for word, similarity in similar:
            print(f"  {word}: {similarity:.3f}")

print("\n" + "="*70 + "\n")

# =============================================================================
# 3. FASTTEXT İMPLEMENTASYONU
# =============================================================================

print("3. FASTTEXT İMPLEMENTASYONU")
print("-" * 50)

class FastTextTurkish:
    """
    Türkçe için basitleştirilmiş FastText implementasyonu
    Subword information kullanır
    """
    
    def __init__(self, vector_size=100, window=5, min_n=3, max_n=6, epochs=5):
        self.vector_size = vector_size
        self.window = window
        self.min_n = min_n  # Minimum n-gram boyutu
        self.max_n = max_n  # Maximum n-gram boyutu
        self.epochs = epochs
        
        self.word_to_index = {}
        self.subword_to_index = {}
        self.word_vectors = {}
        self.subword_vectors = {}
        self.vocabulary_size = 0
        
    def get_subwords(self, word):
        """
        Kelimeden subword'leri (karakter n-gramları) çıkarır
        """
        # Kelimeyi <> ile çevrele
        padded_word = f"<{word}>"
        subwords = []
        
        # N-gramları oluştur
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(padded_word) - n + 1):
                subword = padded_word[i:i + n]
                subwords.append(subword)
        
        return subwords
    
    def build_vocabulary(self, words):
        """
        Kelime ve subword dağarcığını oluşturur
        """
        print("FastText kelime ve subword dağarcığı oluşturuluyor...")
        
        # Kelime dağarcığı
        for i, word in enumerate(set(words)):
            self.word_to_index[word] = i
            
        self.vocabulary_size = len(self.word_to_index)
        
        # Subword dağarcığı
        all_subwords = set()
        for word in self.word_to_index:
            subwords = self.get_subwords(word)
            all_subwords.update(subwords)
        
        for i, subword in enumerate(all_subwords):
            self.subword_to_index[subword] = i
        
        print(f"Kelime sayısı: {len(self.word_to_index)}")
        print(f"Subword sayısı: {len(self.subword_to_index)}")
        
        # Örnek subword'ler göster
        sample_word = list(self.word_to_index.keys())[0]
        sample_subwords = self.get_subwords(sample_word)
        print(f"'{sample_word}' kelimesinin subword'leri: {sample_subwords[:5]}...")
    
    def train(self, words):
        """
        FastText modelini eğitir
        """
        print("FastText eğitimi başlıyor...")
        
        # Dağarcığı oluştur
        self.build_vocabulary(words)
        
        # Rastgele vektörlerle başla
        for word in self.word_to_index:
            self.word_vectors[word] = np.random.uniform(-0.5, 0.5, self.vector_size)
        
        for subword in self.subword_to_index:
            self.subword_vectors[subword] = np.random.uniform(-0.5, 0.5, self.vector_size)
        
        # Eğitim döngüsü
        for epoch in range(self.epochs):
            total_loss = 0
            count = 0
            
            for i, target_word in enumerate(words):
                if target_word not in self.word_to_index:
                    continue
                
                # Bağlam kelimelerini al
                start = max(0, i - self.window)
                end = min(len(words), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j and words[j] in self.word_to_index:
                        context_word = words[j]
                        
                        # Target kelimesinin subword'lerini al
                        target_subwords = self.get_subwords(target_word)
                        
                        # Subword vektörlerinin ortalamasını al
                        target_vector = np.zeros(self.vector_size)
                        valid_subwords = 0
                        
                        for subword in target_subwords:
                            if subword in self.subword_vectors:
                                target_vector += self.subword_vectors[subword]
                                valid_subwords += 1
                        
                        if valid_subwords > 0:
                            target_vector /= valid_subwords
                        
                        # Context kelimesinin vektörü
                        context_vector = self.word_vectors[context_word]
                        
                        # Similarity hesapla ve güncelle
                        similarity = np.dot(target_vector, context_vector) / (
                            np.linalg.norm(target_vector) * np.linalg.norm(context_vector) + 1e-8
                        )
                        
                        # Basit güncelleme
                        learning_rate = 0.01
                        error = 1 - similarity
                        
                        # Subword vektörlerini güncelle
                        for subword in target_subwords:
                            if subword in self.subword_vectors:
                                self.subword_vectors[subword] += learning_rate * error * context_vector / valid_subwords
                        
                        # Context vektörünü güncelle
                        self.word_vectors[context_word] += learning_rate * error * target_vector
                        
                        total_loss += error ** 2
                        count += 1
            
            if epoch % 1 == 0:
                avg_loss = total_loss / count if count > 0 else 0
                print(f"Epoch {epoch + 1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")
        
        # Final kelime vektörlerini oluştur (subword vektörlerinin ortalaması)
        for word in self.word_to_index:
            subwords = self.get_subwords(word)
            word_vector = np.zeros(self.vector_size)
            valid_subwords = 0
            
            for subword in subwords:
                if subword in self.subword_vectors:
                    word_vector += self.subword_vectors[subword]
                    valid_subwords += 1
            
            if valid_subwords > 0:
                word_vector /= valid_subwords
                # Normalize et
                norm = np.linalg.norm(word_vector)
                if norm > 0:
                    word_vector = word_vector / norm
                
                self.word_vectors[word] = word_vector
        
        print("FastText eğitimi tamamlandı!")
    
    def get_word_vector(self, word):
        """
        Kelime vektörünü döndürür (OOV kelimeler için subword'lerden hesaplar)
        """
        if word in self.word_vectors:
            return self.word_vectors[word]
        
        # OOV kelime için subword'lerden vektör oluştur
        subwords = self.get_subwords(word)
        word_vector = np.zeros(self.vector_size)
        valid_subwords = 0
        
        for subword in subwords:
            if subword in self.subword_vectors:
                word_vector += self.subword_vectors[subword]
                valid_subwords += 1
        
        if valid_subwords > 0:
            word_vector /= valid_subwords
            norm = np.linalg.norm(word_vector)
            if norm > 0:
                word_vector = word_vector / norm
        
        return word_vector
    
    def find_similar_words(self, word, top_k=5):
        """
        Benzer kelimeleri bulur (OOV kelimeler dahil)
        """
        target_vector = self.get_word_vector(word)
        
        if np.linalg.norm(target_vector) == 0:
            return []
        
        similarities = []
        
        for other_word in self.word_to_index:
            if other_word != word:
                other_vector = self.word_vectors[other_word]
                similarity = np.dot(target_vector, other_vector)
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# FastText örneği
print("FastText modeli eğitiliyor...")
fasttext_model = FastTextTurkish(vector_size=50, window=3, epochs=3)
fasttext_model.train(words)

# Benzer kelimeleri test et
for test_word in ['istanbul', 'güzel']:
    if test_word in fasttext_model.word_to_index:
        similar = fasttext_model.find_similar_words(test_word)
        print(f"\n'{test_word}' kelimesine benzer kelimeler (FastText):")
        for word, similarity in similar:
            print(f"  {word}: {similarity:.3f}")

# OOV kelime testi
oov_word = "istanbullu"  # Eğitim setinde olmayan kelime
print(f"\nOOV kelime testi: '{oov_word}'")
oov_vector = fasttext_model.get_word_vector(oov_word)
if np.linalg.norm(oov_vector) > 0:
    print(f"OOV kelime için vektör oluşturuldu (norm: {np.linalg.norm(oov_vector):.3f})")
    similar_to_oov = fasttext_model.find_similar_words(oov_word)
    print(f"'{oov_word}' kelimesine benzer kelimeler:")
    for word, similarity in similar_to_oov:
        print(f"  {word}: {similarity:.3f}")

print("\n" + "="*70 + "\n")

# =============================================================================
# 4. EK KÜTÜPHANE: TRANSFORMERS İLE WORD EMBEDDING
# =============================================================================

print("4. EK KÜTÜPHANE: TRANSFORMERS İLE WORD EMBEDDING")
print("-" * 50)

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    class TransformersWordEmbedding:
        """
        Hugging Face Transformers ile önceden eğitilmiş modeller
        """
        
        def __init__(self, model_name='dbmdz/bert-base-turkish-cased'):
            try:
                print(f"Model yükleniyor: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                self.model_loaded = True
                print("Model başarıyla yüklendi!")
            except Exception as e:
                print(f"Model yüklenemedi: {e}")
                self.model_loaded = False
        
        def get_word_embeddings(self, text):
            """
            Metindeki kelimeler için BERT embeddings çıkarır
            """
            if not self.model_loaded:
                return {}
            
            # Tokenize et
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            
            # Model çıktısını al
            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden_states = outputs.last_hidden_state
            
            # Token'ları kelimelerle eşleştir
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            embeddings = {}
            
            for i, token in enumerate(tokens):
                if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                    # Token'ı temizle
                    clean_token = token.lower()
                    if len(clean_token) > 2:
                        embedding = last_hidden_states[0][i].numpy()
                        embeddings[clean_token] = embedding
            
            return embeddings
        
        def find_similar_words_bert(self, word, embeddings, top_k=5):
            """
            BERT embeddings kullanarak benzer kelimeleri bulur
            """
            if word not in embeddings:
                return []
            
            target_vector = embeddings[word]
            similarities = []
            
            for other_word, other_vector in embeddings.items():
                if other_word != word:
                    # Cosine similarity
                    similarity = np.dot(target_vector, other_vector) / (
                        np.linalg.norm(target_vector) * np.linalg.norm(other_vector)
                    )
                    similarities.append((other_word, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
    
    # Transformers örneği
    transformer_embedder = TransformersWordEmbedding()
    
    if transformer_embedder.model_loaded:
        # BERT embeddings çıkar
        bert_embeddings = transformer_embedder.get_word_embeddings(ornek_metin[:200])
        print(f"BERT embeddings sayısı: {len(bert_embeddings)}")
        
        if bert_embeddings:
            print(f"BERT embedding boyutu: {list(bert_embeddings.values())[0].shape}")
            
            # Benzer kelimeleri bul
            sample_word = list(bert_embeddings.keys())[0]
            similar_bert = transformer_embedder.find_similar_words_bert(sample_word, bert_embeddings)
            print(f"\n'{sample_word}' kelimesine benzer kelimeler (BERT):")
            for word, similarity in similar_bert:
                print(f"  {word}: {similarity:.3f}")

except ImportError:
    print("Transformers kütüphanesi yüklü değil.")
    print("Kurulum için: pip install transformers torch")
    print("\nTransformers ile yapılabilecek işlemler:")
    print("- BERT, RoBERTa, GPT gibi önceden eğitilmiş modeller")
    print("- Çok dilli ve Türkçe özel modeller")
    print("- Contextual embeddings (bağlama duyarlı)")
    print("- Fine-tuning imkanı")

print("\n" + "="*70 + "\n")

# =============================================================================
# 5. MODEL KARŞILAŞTIRMASI VE VİZUALİZASYON
# =============================================================================

print("5. MODEL KARŞILAŞTIRMASI VE VİZUALİZASYON")
print("-" * 50)

def visualize_embeddings(embeddings_dict, title, max_words=20):
    """
    Word embeddings'leri 2D'de görselleştirir
    """
    if not embeddings_dict:
        print(f"{title} için embedding bulunamadı.")
        return
    
    # Sadece ilk max_words kelimeyi al
    words = list(embeddings_dict.keys())[:max_words]
    vectors = [embeddings_dict[word] for word in words]
    
    if len(vectors) < 2:
        print(f"{title} için yeterli embedding yok.")
        return
    
    # PCA ile boyut azaltma
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
    
    # Kelime etiketleri
    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.title(f'{title} - Word Embeddings Visualization (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Model karşılaştırma tablosu
comparison_data = {
    'Model': ['Word2Vec', 'GloVe', 'FastText', 'BERT/Transformers'],
    'Yaklaşım': ['Neural Network', 'Matrix Factorization', 'Subword + Neural', 'Transformer'],
    'OOV Handling': ['Hayır', 'Hayır', 'Evet', 'Evet'],
    'Türkçe Uygunluk': ['İyi', 'İyi', 'Mükemmel', 'Mükemmel'],
    'Eğitim Hızı': ['Hızlı', 'Orta', 'Orta', 'Yavaş'],
    'Bellek Kullanımı': ['Az', 'Orta', 'Fazla', 'Çok Fazla'],
    'Contextual': ['Hayır', 'Hayır', 'Hayır', 'Evet']
}

comparison_df = pd.DataFrame(comparison_data)
print("MODEL KARŞILAŞTIRMASI:")
print(comparison_df.to_string(index=False))

# Embedding kalitesi karşılaştırması
print(f"\nEMBEDDİNG KALİTESİ KARŞILAŞTIRMASI:")
print("-" * 40)

models = {
    'Word2Vec': word2vec_model.word_vectors,
    'GloVe': glove_model.word_vectors,
    'FastText': fasttext_model.word_vectors
}

test_word = 'türkiye'
if test_word in word2vec_model.word_vectors:
    print(f"'{test_word}' kelimesi için benzer kelimeler:")
    
    for model_name, embeddings in models.items():
        if test_word in embeddings:
            # Basit benzerlik hesaplama
            target_vector = embeddings[test_word]
            similarities = []
            
            for word, vector in embeddings.items():
                if word != test_word:
                    similarity = np.dot(target_vector, vector)
                    similarities.append((word, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_similar = similarities[:3]
            
            print(f"\n{model_name}:")
            for word, sim in top_similar:
                print(f"  {word}: {sim:.3f}")

# Öneriler
print(f"\nÖNERİLER VE SONUÇ:")
print("-" * 40)
print("🎯 TÜRKÇE İÇİN EN İYİ SEÇİMLER:")
print("  • FastText: Morfolojik zenginlik ve OOV handling")
print("  • BERT: Contextual embeddings ve yüksek kalite")
print("  • Word2Vec: Hız ve basitlik gerektiğinde")

print(f"\n📊 KULLANIM ALANLARI:")
print("  • Metin Sınıflandırma: FastText veya BERT")
print("  • Duygu Analizi: BERT")
print("  • Bilgi Çıkarımı: BERT")
print("  • Hızlı Prototipler: Word2Vec")

print(f"\n⚡ PERFORMANS İPUÇLARI:")
print("  • Büyük veri setleri için: Word2Vec veya FastText")
print("  • Yüksek kalite gerektiğinde: BERT")
print("  • OOV kelimeler varsa: FastText veya BERT")
print("  • Bellek kısıtlı ortamlar: Word2Vec")

print(f"\n🔧 UYGULAMA ÖNERİLERİ:")
print("  • Preprocessing: Türkçe karakterleri koru")
print("  • Corpus boyutu: En az 1M kelime öneriliyor")
print("  • Hyperparameter tuning: Grid search kullan")
print("  • Evaluation: Intrinsic ve extrinsic metrikler")

print("\n" + "="*70)
print("Word Embedding eğitimi tamamlandı! 🎉")
print("="*70)