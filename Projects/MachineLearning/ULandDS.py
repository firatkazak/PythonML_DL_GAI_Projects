# Unsupervised Learning and Data Scaling
from sklearn.datasets import load_breast_cancer  # Sklearn kütüphanesinden meme kanseri veri setini yükleyen fonksiyonu içe aktarır.
from sklearn.model_selection import train_test_split  # Eğitim ve test veri setlerine ayırmak için kullanılan fonksiyonu içe aktarır.
from sklearn.preprocessing import MinMaxScaler  # Özellikleri ölçeklendirmek için kullanılan sınıfı içe aktarır.
from sklearn.svm import SVC  # Destek vektör makineleri (SVM) sınıflandırıcısını içe aktarır.

kanser = load_breast_cancer()  # Meme kanseri veri setini yükler ve 'kanser' değişkenine atar.
X_egitim, X_test, y_egitim, y_test = train_test_split(kanser.data,  # Özellikleri ve etiketleri eğitim ve test setlerine böler.
                                                      kanser.target,  # Özellikler (X) ve etiketler (y) arasındaki ilişkiyi belirtir.
                                                      random_state=0  # Rastgelelik için sabit bir tohum değeri belirler (sonuçların tekrarlanabilirliğini sağlar).
                                                      )

svm = SVC(C=100,  # C: SVM modelinin düzenleme parametresi, modelin karmaşıklığını kontrol eder. Burada 100 olarak ayarlanmış.
          gamma="auto"  # gamma: SVM modelinde kernel fonksiyonunun genişliğini belirler. "auto" seçeneği, otomatik bir değer kullanır.
          )

svm.fit(X_egitim, y_egitim)  # SVM modelini eğitim verileri ile eğitir.
print(svm.score(X_test, y_test))  # Test verileri ile modelin başarısını değerlendirir ve skorunu ekrana yazdırır. (0.6293706293706294)

scaler = MinMaxScaler()  # Özellikleri [0, 1] aralığına ölçeklendirmek için bir scaler nesnesi oluşturur.
scaler.fit(X_egitim)  # Eğitim verileri ile scaler'ı eğitir (ölçekleme parametrelerini hesaplar).

X_egitim_olcekli = scaler.transform(X_egitim)  # Eğitim verilerini ölçeklendirir.
X_test_olcekli = scaler.transform(X_test)  # Test verilerini ölçeklendirir.

svm.fit(X_egitim_olcekli, y_egitim)  # Ölçeklenmiş eğitim verileri ile SVM modelini tekrar eğitir.
print(svm.score(X_test_olcekli, y_test))  # Ölçeklenmiş test verileri ile modelin başarısını değerlendirir ve skorunu ekrana yazdırır. (0.965034965034965)
