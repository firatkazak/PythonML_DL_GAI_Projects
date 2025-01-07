import numpy as np  # NumPy kütüphanesini içe aktarıyor.
from sklearn.datasets import load_files  # Dosyaları yüklemek için gerekli fonksiyonu içe aktarıyor.
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # Metin verilerini sayısal verilere dönüştürmek için sınıfları içe aktarıyor.
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes sınıflandırıcısını içe aktarıyor.
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent (SGD) sınıflandırıcısını içe aktarıyor.
from sklearn.pipeline import Pipeline  # Birden fazla adımı bir araya getirmek için kullanılacak sınıfı içe aktarıyor.
from sklearn import metrics  # Model değerlendirme metriklerini içe aktarıyor.

# Her örnekte kullanılan kategorileri tutan değişken
categories = ['rec.motorcycles', 'rec.sport.baseball', 'comp.graphics', 'rec.sport.hockey']

# Öznitelikleri Oluşturma
twenty_train = load_files(
    container_path='C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/20news-bydate-train',  # Veri dosyalarının yolu
    categories=categories,  # Hangi kategorilerin yükleneceği
    shuffle=True,  # Verileri karıştır
    random_state=42,  # Tekrar edilebilirlik için rastgele durum
    encoding='utf-8',  # Dosya kodlaması
    decode_error='ignore'  # Kodlama hatalarını yoksay
)

# Yukarıdaki satırlarda yüklenen veri kümesine dair bilgiler
# print(type(twenty_train))  # <class 'sklearn.utils._bunch.Bunch'> veri türünü gösterir
# print(twenty_train.target_names)  # Hedef isimlerini yazdırır
# print(len(twenty_train.data))  # Veri kümesindeki örnek sayısını yazdırır
# print(twenty_train.target[:10])  # İlk 10 hedef etiketini yazdırır

count_vect = CountVectorizer()  # Metin verilerini kelime sayısına dönüştüren bir nesne oluşturuyor
X_train_counts = count_vect.fit_transform(twenty_train.data)  # Metin verilerini sayısal verilere dönüştürüyor
# print(X_train_counts.shape)  # (2379, 32550) boyutlarını gösterir
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)  # TF-IDF dönüşümü için nesne oluşturuyor (idf kullanılmıyor)
X_train_tf = tf_transformer.transform(X_train_counts)  # Sayısal verileri TF-IDF formatına dönüştürüyor
# print(X_train_tf.shape)  # (2379, 32550) boyutlarını gösterir

# Model Kurma
clf = MultinomialNB().fit(X_train_tf, twenty_train.target)  # Naive Bayes modelini oluşturuyor ve veriye fit ediyor
docs_new = ['brake-lamp is good', 'this computer is fast']  # Sınıflandırmak istediğimiz yeni belgeler
X_new_count = count_vect.transform(docs_new)  # Yeni belgeleri sayısal verilere dönüştürüyor
X_new_tf = tf_transformer.transform(X_new_count)  # Yeni sayısal verileri TF-IDF formatına dönüştürüyor
predicted = clf.predict(X_new_tf)  # Yeni belgeler için tahmin yapıyor

# Tahmin sonuçlarını yazdırma
for doc, category in zip(docs_new, predicted):
    print('%r=>%s' % (doc, twenty_train.target_names[category]))  # Tahmin edilen kategoriyi yazdırır

# Pipeline ile daha kolay şekilde model kurma
text_clf = Pipeline([  # Birden fazla işlemi tek bir nesnede toplamak için Pipeline oluşturma
    ('vect', CountVectorizer()),  # İlk adım: Metni kelime sayısına dönüştür
    ('tfidf', TfidfTransformer()),  # İkinci adım: TF-IDF dönüşümü
    ('clf', MultinomialNB())  # Üçüncü adım: Naive Bayes sınıflandırıcısı
])
text_clf.fit(twenty_train.data, twenty_train.target)  # Modeli eğitiyor

# Model Değerlendirme
twenty_test = load_files(
    container_path='C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/20news-bydate-test',  # Test veri dosyası yolu
    categories=categories,  # Hangi kategorilerin yükleneceği
    shuffle=True,  # Verileri karıştır
    random_state=42,  # Tekrar edilebilirlik için rastgele durum
    encoding='utf-8',  # Dosya kodlaması
    decode_error='ignore'  # Kodlama hatalarını yoksay
)

docs_test = twenty_test.data  # Test verisi
predicted = text_clf.predict(docs_test)  # Test verisi üzerinde tahmin yapıyor
print(np.mean(predicted == twenty_test.target))  # Doğru tahminlerin ortalamasını yazdırır

# Yeni bir Pipeline ile farklı bir model (SGDClassifier) oluşturma
text_clf = Pipeline([
    ('vect', CountVectorizer()),  # İlk adım: Metni kelime sayısına dönüştür
    ('tfidf', TfidfTransformer()),  # İkinci adım: TF-IDF dönüşümü
    ('clf', SGDClassifier(loss='hinge',  # Üçüncü adım: SGD sınıflandırıcısı
                          penalty='l2',  # L2 ceza terimi
                          alpha=1e-3,  # Öğrenme oranı
                          random_state=42,  # Tekrar edilebilirlik için rastgele durum
                          max_iter=5,  # Maksimum iterasyon sayısı
                          tol=None  # Durma kriteri
                          )
     )
])
text_clf.fit(twenty_train.data, twenty_train.target)  # Modeli eğitiyor
predicted = text_clf.predict(docs_test)  # Test verisi üzerinde tahmin yapıyor
print(np.mean(predicted == twenty_test.target))  # Doğru tahminlerin ortalamasını yazdırır

# Karışıklık matrisini oluşturma
sonuc = metrics.confusion_matrix(twenty_test.target, predicted)  # Gerçek ve tahmin edilen değerleri karşılaştırarak karışıklık matrisini oluşturuyor
print(sonuc)  # Karışıklık matrisini yazdırır
