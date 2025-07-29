import os
import re
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Türkçe Wikipedia veri setini indirme fonksiyonu
def indir_turkce_wikipedia():
    """
    Türkçe Wikipedia dump dosyasını indirir ve ayıklar
    """
    url = "https://dumps.wikimedia.org/trwiki/latest/trwiki-latest-pages-articles.xml.bz2"
    dosya_adi = "turkce_wikipedia.xml.bz2"
    
    print("Türkçe Wikipedia veri seti indiriliyor...")
    try:
        urllib.request.urlretrieve(url, dosya_adi)
        print(f"İndirme tamamlandı: {dosya_adi}")
        return dosya_adi
    except Exception as e:
        print(f"İndirme hatası: {e}")
        print("Alternatif olarak örnek Türkçe metin dosyası oluşturuluyor...")
        return olustur_ornek_turkce_veri()

# 2. Alternatif örnek Türkçe veri seti oluşturma
def olustur_ornek_turkce_veri():
    """
    Örnek Türkçe metinler içeren bir veri seti oluşturur
    """
    turkce_metinler = [
        "Türkiye Cumhuriyeti, topraklarının büyük bölümü Anadolu'ya, küçük bir bölümü ise Balkan Yarımadası'nın güneydoğu uzantısı olan Trakya'ya yayılmış bir ülkedir.",
        "İstanbul, Türkiye'nin en kalabalık şehridir ve ekonomik, kültürel açıdan en önemli merkezidir.",
        "Ankara, Türkiye'nin başkentidir ve ikinci büyük şehridir. Türkiye Büyük Millet Meclisi burada yer alır.",
        "Ege Denizi, Türkiye'nin batısında yer alır ve birçok güzel sahil kentine ev sahipliği yapar.",
        "Karadeniz, Türkiye'nin kuzeyinde yer alır ve balıkçılık açısından oldukça önemlidir.",
        "Akdeniz, Türkiye'nin güneyinde yer alır ve turizm açısından büyük öneme sahiptir.",
        "Türk mutfağı, dünya mutfakları arasında önemli bir yere sahiptir. Kebap, baklava ve Türk kahvesi en bilinen yiyeceklerindendir.",
        "Osmanlı İmparatorluğu, 600 yıldan uzun süre hüküm sürmüş büyük bir imparatorluktur.",
        "Mustafa Kemal Atatürk, Türkiye Cumhuriyeti'nin kurucusudur ve modern Türkiye'nin mimarıdır.",
        "Türkçe, Ural-Altay dil ailesine mensup bir dildir ve yaklaşık 80 milyon kişi tarafından konuşulmaktadır.",
        "İzmir, Ege Bölgesi'nin en büyük şehridir ve liman kenti olarak önemli bir ticaret merkezidir.",
        "Bursa, Osmanlı İmparatorluğu'nun ilk başkentidir ve kestane şekeri ile meşhurdur.",
        "Antalya, Akdeniz Bölgesi'nin turizm başkentidir ve milyonlarca turist her yıl burayı ziyaret eder.",
        "Kapadokya, peri bacaları ile ünlü bir bölgedir ve UNESCO Dünya Mirası Listesi'nde yer almaktadır.",
        "Pamukkale, beyaz travertenleri ile ünlü doğal bir güzelliktir ve termal suları ile bilinir.",
        "Troya, Homeros'un İlyada destanında geçen ünlü antik kenttir ve Çanakkale'de yer alır.",
        "Galata Kulesi, İstanbul'un simgelerinden biridir ve 14. yüzyıldan kalma bir yapıdır.",
        "Ayasofya, Bizans döneminde kilise, Osmanlı döneminde cami, günümüzde ise müze olarak hizmet vermiştir.",
        "Topkapı Sarayı, Osmanlı sultanlarının 400 yıl boyunca yaşadığı saraydır ve şu an müzedir.",
        "Dolmabahçe Sarayı, Osmanlı'nın son döneminde inşa edilmiş görkemli bir saraydır."
    ]
    
    # Metinleri çoğaltarak daha büyük bir veri seti oluşturalım
    genisletilmis_metinler = []
    for metin in turkce_metinler:
        genisletilmis_metinler.append(metin)
        # Kelimeleri karıştırarak varyasyonlar oluşturalım
        kelimeler = metin.split()
        for i in range(3):
            karisik = np.random.permutation(kelimeler)
            genisletilmis_metinler.append(' '.join(karisik))
    
    # Dosyaya kaydet
    with open('turkce_ornek_veri.txt', 'w', encoding='utf-8') as f:
        for metin in genisletilmis_metinler:
            f.write(metin + '\n')
    
    return 'turkce_ornek_veri.txt'

# 3. Türkçe metin ön işleme fonksiyonları
def turkce_karakter_duzelt(metin):
    """
    Türkçe karakterleri düzgün formata çevirir
    """
    # Türkçe karakterleri küçük harfe çevir
    metin = metin.lower()
    
    # Noktalama işaretlerini kaldır
    noktalama = r'[^\w\sçğıöşü]'
    metin = re.sub(noktalama, ' ', metin)
    
    # Fazla boşlukları temizle
    metin = re.sub(r'\s+', ' ', metin).strip()
    
    return metin

def cumleleri_bol(metin):
    """
    Metni cümlelere böler
    """
    # Nokta, ünlem ve soru işaretlerine göre cümleleri ayır
    cumleler = re.split(r'[.!?]+', metin)
    # Boş cümleleri temizle
    cumleler = [cumle.strip() for cumle in cumleler if cumle.strip()]
    return cumleler

def kelimelere_bol(cumle):
    """
    Cümleyi kelimelere böler
    """
    # Basit boşluklara göre kelime ayırma
    kelimeler = cumle.split()
    # 2 harften kısa kelimeleri filtrele (çok kısa bağlaçlar vs.)
    kelimeler = [kelime for kelime in kelimeler if len(kelime) > 2]
    return kelimeler

def metin_on_isleme(dosya_yolu):
    """
    Tüm metin ön işleme adımlarını uygular
    """
    print("Metin ön işleme başlatılıyor...")
    
    # Dosyadan metinleri oku
    with open(dosya_yolu, 'r', encoding='utf-8') as f:
        metinler = f.readlines()[:1000]
    
    islenmis_cumleler = []
    
    for metin in metinler:
        # Türkçe karakter düzeltme
        temiz_metin = turkce_karakter_duzelt(metin)
        
        # Cümlelere böl
        cumleler = cumleleri_bol(temiz_metin)
        
        # Her cümleyi kelimelere böl
        for cumle in cumleler:
            kelimeler = kelimelere_bol(cumle)
            if len(kelimeler) > 2:  # En az 3 kelime olan cümleleri al
                islenmis_cumleler.append(kelimeler)
    
    print(f"Toplam {len(islenmis_cumleler)} cümle işlendi.")
    return islenmis_cumleler

# 4. Word embeddings modellerini eğitme fonksiyonları
def word2vec_egit(cumleler, boyut=100):
    """
    Word2Vec modelini eğitir
    """
    print("Word2Vec modeli eğitiliyor...")
    model = Word2Vec(
        sentences=cumleler,
        vector_size=boyut,  # Vektör boyutu
        window=5,  # Pencere boyutu
        min_count=2,  # Minimum kelime frekansı
        workers=4,  # Paralel işlem sayısı
        epochs=100,  # Eğitim epoch sayısı
        sg=1  # Skip-gram modeli (0=CBOW)
    )
    return model

def fasttext_egit(cumleler, boyut=100):
    """
    FastText modelini eğitir
    """
    print("FastText modeli eğitiliyor...")
    model = FastText(
        sentences=cumleler,
        vector_size=boyut,
        window=5,
        min_count=2,
        workers=4,
        epochs=100,
        sg=1
    )
    return model

# 5. Model değerlendirme ve karşılaştırma fonksiyonları
def benzer_kelimeleri_bul(model, kelime, n=5):
    """
    Verilen kelimeye en benzer kelimeleri bulur
    """
    try:
        benzerler = model.wv.most_similar(kelime, topn=n)
        return benzerler
    except KeyError:
        return [(f"'{kelime}' kelimesi kelime dağarcığında bulunamadı", 0)]

def kelime_analojisi(model, pozitif, negatif, n=3):
    """
    Kelime analojisi yapar: pozitif - negatif + ?
    Örnek: ['kadın', 'kral'] - ['erkek'] = 'kraliçe'
    """
    try:
        sonuc = model.wv.most_similar(
            positive=pozitif,
            negative=negatif,
            topn=n
        )
        return sonuc
    except KeyError as e:
        return [(f"Kelime bulunamadı: {e}", 0)]

def model_karsilastir(word2vec_model, fasttext_model, test_kelimeler):
    """
    Modelleri karşılaştırır ve sonuçları gösterir
    """
    print("\n=== MODEL KARŞILAŞTIRMASI ===")
    
    sonuclar = []
    
    for kelime in test_kelimeler:
        print(f"\n'{kelime}' kelimesi için benzerlik analizi:")
        
        # Word2Vec sonuçları
        w2v_benzer = benzer_kelimeleri_bul(word2vec_model, kelime)
        print(f"Word2Vec: {w2v_benzer}")
        
        # FastText sonuçları
        ft_benzer = benzer_kelimeleri_bul(fasttext_model, kelime)
        print(f"FastText: {ft_benzer}")
        
        sonuclar.append({
            'kelime': kelime,
            'word2vec': w2v_benzer,
            'fasttext': ft_benzer
        })
    
    return sonuclar

def vektor_gorsellestir(model, kelimeler, baslik):
    """
    Kelime vektörlerini 2D boyutta görselleştirir
    """
    # Kelime vektörlerini al
    vektorler = []
    etiketler = []
    
    for kelime in kelimeler:
        if kelime in model.wv:
            vektorler.append(model.wv[kelime])
            etiketler.append(kelime)
    
    if len(vektorler) == 0:
        print(f"Hiçbir kelime '{baslik}' modelinde bulunamadı.")
        return
    
    # PCA ile 2D'ye indirgeme
    pca = PCA(n_components=2)
    vektorler_2d = pca.fit_transform(vektorler)
    
    # Görselleştirme
    plt.figure(figsize=(10, 8))
    plt.scatter(vektorler_2d[:, 0], vektorler_2d[:, 1])
    
    for i, kelime in enumerate(etiketler):
        plt.annotate(kelime, (vektorler_2d[i, 0], vektorler_2d[i, 1]))
    
    plt.title(f'{baslik} - Kelime Vektörleri Görselleştirmesi')
    plt.xlabel('Bileşen 1')
    plt.ylabel('Bileşen 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{baslik.lower().replace(" ", "_")}_vektorler.png')
    plt.show()

# 6. Ana çalıştırma fonksiyonu
def main():
    """
    Tüm süreci çalıştıran ana fonksiyon
    """
    print("=== TÜRKÇE WORD EMBEDDINGS EĞİTİMİ BAŞLIYOR ===\n")
    
    # 1. Veri setini indir veya oluştur
    dosya_yolu = "NLP/trwiki-latest-pages-articles.xml"
    # dosya_yolu = olustur_ornek_turkce_veri()  # Wikipedia yerine örnek veri kullanıyoruz
    
    # 2. Metin ön işleme
    islenmis_cumleler = metin_on_isleme(dosya_yolu)
    
    # 3. Modelleri eğit
    word2vec_model = word2vec_egit(islenmis_cumleler)
    fasttext_model = fasttext_egit(islenmis_cumleler)
    
    # 4. Modelleri kaydet
    word2vec_model.save("turkce_word2vec.model")
    fasttext_model.save("turkce_fasttext.model")
    print("Modeller kaydedildi.")
    
    # 5. Model değerlendirmesi
    test_kelimeler = ['türkiye', 'istanbul', 'ankara', 'deniz', 'kültür']
    karsilastirma_sonuclari = model_karsilastir(word2vec_model, fasttext_model, test_kelimeler)
    
    # 6. Kelime analojisi testi
    print("\n=== KELİME ANALOJİSİ TESTİ ===")
    analoji_sonuc = kelime_analojisi(word2vec_model, ['kral', 'kadın'], ['erkek'])
    print(f"kral - erkek + kadın = {analoji_sonuc}")
    
    # 7. Vektör görselleştirmesi
    gorsellestirme_kelimeler = ['türkiye', 'istanbul', 'ankara', 'izmir', 'bursa', 'antalya']
    vektor_gorsellestir(word2vec_model, gorsellestirme_kelimeler, 'Word2Vec')
    vektor_gorsellestir(fasttext_model, gorsellestirme_kelimeler, 'FastText')
    
    # 8. Model boyutları ve istatistikleri
    print(f"\n=== MODEL İSTATİSTİKLERİ ===")
    print(f"Word2Vec kelime dağarcığı boyutu: {len(word2vec_model.wv)}")
    print(f"FastText kelime dağarcığı boyutu: {len(fasttext_model.wv)}")
    
    # 9. Örnek kelime vektörleri
    print(f"\n=== ÖRNEK KELİME VEKTÖRLERİ ===")
    ornek_kelime = 'türkiye'
    if ornek_kelime in word2vec_model.wv:
        print(f"'{ornek_kelime}' için Word2Vec vektörü (ilk 10 değer):")
        print(word2vec_model.wv[ornek_kelime][:10])
    
    if ornek_kelime in fasttext_model.wv:
        print(f"'{ornek_kelime}' için FastText vektörü (ilk 10 değer):")
        print(fasttext_model.wv[ornek_kelime][:10])

# Programı çalıştır
if __name__ == "__main__":
    main()