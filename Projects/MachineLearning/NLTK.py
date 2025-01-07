import nltk  # Doğal dil işleme kütüphanesi
import string  # Python'da string işlemleri ve karakter setleri için
from nltk.corpus import movie_reviews  # NLTK'nin film incelemeleri veri setini kullanmak için
import random  # Veri kümesini rastgele karıştırmak için
import os

# İndirme dizinini ayarlama
nltk_data_path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler"
os.makedirs(nltk_data_path, exist_ok=True)  # Klasörün var olduğundan emin oluyoruz

# NLTK'ya yeni dizini ekleme
nltk.data.path.append(nltk_data_path)

# Gutenberg verisini belirlenen konuma indirme
nltk.download('gutenberg', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('movie_reviews', download_dir=nltk_data_path)

# Shakespeare'in Hamlet metninin dosya yolu
path = "C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/corpora/gutenberg/shakespeare-hamlet.txt"

# Hamlet metnini sözcük bazında yükle
hamlet = nltk.corpus.gutenberg.words(path)
print(hamlet)  # Metindeki kelimeleri yazdır
print(len(hamlet))  # Metindeki kelime sayısını yazdır
print(hamlet[:10])  # Metindeki ilk 10 kelimeyi yazdır

# Hamlet metnini cümle bazında yükle
hamlet_sents = nltk.corpus.gutenberg.sents(path)
print(hamlet_sents[:5])  # İlk 5 cümleyi yazdır

# Hamlet metnini bir NLTK Text objesine çeviriyoruz
text = nltk.Text(hamlet)
text.concordance("Stage")  # 'Stage' kelimesinin geçtiği yerleri bul ve göster

# 'Stage' kelimesinin geçtiği bağlamları bul ve yazdır
text.common_contexts(['Stage'])

# Hamlet metnindeki kelimelerin frekans dağılımını alıyoruz
fd = nltk.FreqDist(hamlet)
# En sık geçen 10 kelimeyi al
swTop10 = fd.most_common(10)
print("En Çok 10:", swTop10)

# İngilizce stopwords (önemsiz kelimeleri) alıyoruz
sw = set(nltk.corpus.stopwords.words('english'))
print("Stopwords sayısı:", len(sw))  # Stopwords sayısını yazdır

# Stopwords'lerin ilk 10 tanesini yazdırıyoruz
swTop10 = list(sw)[:10]
print("Stop Words İlk 10:", swTop10)

# 1. Stop words'leri (önemsiz kelimeleri) metinden çıkarıyoruz
hamlet_filtered = [w for w in hamlet if w.lower() not in sw]
# Stop words çıkarılmış metnin frekans dağılımını alıyoruz
fd_hamlet_filtered = nltk.FreqDist(hamlet_filtered)

# Stop words çıkarılmış metindeki en sık geçen 10 kelime
swTop10Filtered = fd_hamlet_filtered.most_common(10)
print("Stop Words Filtreli İlk 10:", swTop10Filtered)

# 2. Noktalama işaretlerini de metinden çıkarıyoruz
p = set(string.punctuation)  # Noktalama işaretlerini bir set'e alıyoruz
hamlet_filtered_punct = [w for w in hamlet_filtered if w not in p]  # Hem stop words hem noktalama işaretlerini filtreliyoruz
# Filtrelenmiş metnin frekans dağılımı
fd_hamlet_filtered_punct = nltk.FreqDist(hamlet_filtered_punct)

# Hem stop words hem de noktalama işaretleri çıkarılmış metindeki en sık geçen 10 kelime
swTop10FilteredPunct = fd_hamlet_filtered_punct.most_common(10)
print("Stop Words ve Noktalama İşaretleri Filtreli İlk 10:", swTop10FilteredPunct)

# Bigram (iki kelimelik diziler) frekans dağılımını oluşturuyoruz
bgrms = nltk.FreqDist(nltk.bigrams(hamlet_filtered_punct))
# En çok geçen 15 bigram'ı yazdırıyoruz
encok15bgrms = bgrms.most_common(15)
print("En Çok Bigrams 15", encok15bgrms)

# Trigram (üç kelimelik diziler) frekans dağılımını oluşturuyoruz
tgrms = nltk.FreqDist(nltk.trigrams(hamlet_filtered_punct))
# En çok geçen 9 trigram'ı yazdırıyoruz
encok15tgrms = tgrms.most_common(9)
print("En Çok Trigrams 9", encok15tgrms)

# Film incelemelerinden kategorize edilmiş dokümanlar oluşturuyoruz
# (kelime listesi, kategori) şeklinde bir liste
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()  # Her kategori için
             for fileid in movie_reviews.fileids(category)]  # Her doküman için

# Dokümanları rastgele karıştırıyoruz
random.shuffle(documents)

# Tüm film incelemelerindeki kelimelerin frekans dağılımı
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
# En sık geçen 2000 kelimeyi alıyoruz
word_features = list(all_words.keys())[:2000]


# Belirli bir dokümandaki kelimelerin var olup olmadığını gösteren özellikler fonksiyonu
def document_features(document):
    document_words = set(document)  # Dokümandaki kelimeler seti
    features = {}  # Boş bir özellikler sözlüğü
    for word in word_features:  # En sık geçen 2000 kelimeyi döngüyle dolaş
        # 'contains(kelime)' şeklinde anahtar oluştur ve True/False değeri ata
        features['contains(%s)' % word] = (word in document_words)
    return features  # Özellikler sözlüğünü döndür


# Özellik setleri oluşturuyoruz: (özellikler, kategori)
featuresets = [(document_features(d), c) for (d, c) in documents]
print("Feature Set Sayısı:", len(featuresets))  # Toplam kaç özellik seti olduğunu yazdır

# Eğitim ve test setlerine ayırıyoruz
train_set, test_set = featuresets[1600:], featuresets[:400]
# Naive Bayes sınıflandırıcısını eğitiyoruz
classifier = nltk.NaiveBayesClassifier.train(train_set)
# Test setinde sınıflandırma doğruluğunu ölçüyoruz
sonuc = nltk.classify.accuracy(classifier, test_set)
print(sonuc)  # Doğruluk oranını yazdır

# En bilgilendirici 5 özelliği yazdır
iyilestirilmisSonuc = classifier.show_most_informative_features(5)
print(iyilestirilmisSonuc)
