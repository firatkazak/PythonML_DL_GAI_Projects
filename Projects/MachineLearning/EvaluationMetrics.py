from sklearn.datasets import load_iris, load_digits  # Iris ve rakam verilerini yüklemek için.
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine bölmek için.
from sklearn.dummy import DummyClassifier  # Dummy sınıflandırıcı, basit referans model olarak kullanılır.
from sklearn.svm import SVC  # Destek Vektör Makineleri (SVC) sınıflandırma algoritması.
from sklearn.metrics import confusion_matrix  # Karışıklık matrisi hesaplama.
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon sınıflandırıcı.
from sklearn.metrics import classification_report, roc_curve, accuracy_score  # Sınıflandırma raporu, ROC eğrisi ve doğruluk skoru.
import seaborn as sns  # Veri görselleştirme için.
import pandas as pd  # Veri manipülasyonu için.
import mglearn  # Makine öğrenimi görselleştirme araçları için.
import matplotlib.pyplot as plt  # Grafik çizim kütüphanesi.

# Binary sınıflandırma örneği: Dengesi bozuk veriler üzerinde
iris = load_iris()  # Iris çiçeği veri setini yükler.
X, y = iris.data, iris.target  # Özellikler (X) ve hedef (y) değişkeni.
y[y != 1] = -1  # Veriyi iki sınıfa dönüştürüyoruz: Sınıf 1 ve diğerleri (-1 olarak ayarlanıyor).
X_train, X_test, y_train, y_test = train_test_split(X,  # Veriyi eğitim ve test setlerine böler.
                                                    y,
                                                    random_state=0  # Rastgeleliği sabitlemek için.
                                                    )

svc = SVC(kernel='linear',  # Destek Vektör Makineleri kullanılıyor, 'linear' kernel ile.
          C=1  # C parametresi: Regularizasyon parametresi (hata toleransı ayarı).
          ).fit(X=X_train, y=y_train)  # Eğitim setiyle modeli eğitir.

svcScore = svc.score(X_test, y_test)  # Test seti üzerinde modelin doğruluk skorunu hesaplar.
print(svcScore)  # Doğruluk skoru: 0.631578947368421

# DummyClassifier, basit bir referans model.
clf = DummyClassifier(strategy='most_frequent',  # En sık görülen sınıfı tahmin eder.
                      random_state=0  # Rastgeleliği sabitler.
                      ).fit(X_train, y_train)  # Eğitim setiyle modeli eğitir.

clfScore = clf.score(X_test, y_test)  # Dummy modelin doğruluk skorunu hesaplar.
print(clfScore)  # Doğruluk skoru: 0.5789473684210527

# SVC modeli bu sefer 'rbf' kernel ile.
svc = SVC(kernel='rbf',  # Radial Basis Function (RBF) kernel kullanılır.
          gamma='scale',  # Gamma parametresi: Kernel etki alanı ayarı (otomatik hesaplanır).
          C=1  # Regularizasyon parametresi.
          ).fit(X_train, y_train)  # Modeli eğitim setinde eğitir.

print(svc.score(X_test, y_test))  # Test seti doğruluk skoru: 0.9473684210526315

# Confusion Matrix (Karışıklık Matrisi)
digits = load_digits()  # Rakam veri setini yükler.
y = digits.target == 9  # Hedef değişken: Sadece rakam 9 olan ve olmayan şeklinde binary sınıflandırma.

X_train, X_test, y_train, y_test = train_test_split(digits.data,  # Veriyi eğitim ve test setlerine böler.
                                                    y,
                                                    random_state=0  # Rastgeleliği sabitler.
                                                    )

logreg = LogisticRegression(solver='liblinear',  # Lojistik regresyon modeli 'liblinear' çözücü ile.
                            C=0.1  # Regularizasyon parametresi.
                            ).fit(X_train, y_train)  # Eğitim seti ile modeli eğitir.

pred_logreg = logreg.predict(X_test)  # Test seti üzerindeki tahminler.
cm = confusion_matrix(y_true=y_test, y_pred=pred_logreg)  # Karışıklık matrisi.
print(cm)  # [[401   2] [  8  39]] - 401 doğru negatif, 2 yanlış pozitif, 8 yanlış negatif, 39 doğru pozitif.

# Karışıklık matrisi görselleştirmesi
mglearn.plots.plot_confusion_matrix_illustration()  # Karışıklık matrisi görselleştirme örneği.
plt.show()

mglearn.plots.plot_binary_confusion_matrix()  # İkili sınıflandırma için karışıklık matrisi.
plt.show()

# Sınıflandırma raporu
print(classification_report(y_true=y_test,  # Gerçek sınıf etiketleri.
                            y_pred=pred_logreg,  # Tahmin edilen sınıf etiketleri.
                            target_names=['not nine', 'nine']  # Sınıf isimleri.
                            ))

# ROC Curve (ROC Eğrisi)
fpr, tpr, thresholds = roc_curve(y_true=y_test,  # Gerçek sınıf etiketleri.
                                 y_score=logreg.decision_function(X_test))  # Lojistik regresyonun karar fonksiyonu skorları.
plt.plot(fpr,  # Yanlış pozitif oranı (False Positive Rate).
         tpr,  # Doğru pozitif oranı (True Positive Rate).
         label='ROC Curve'  # Eğriye etiket ekler.
         )

plt.xlabel('FPR')  # X ekseni: Yanlış pozitif oranı.
plt.ylabel('TPR(recall)')  # Y ekseni: Doğru pozitif oranı.
plt.show()

# Multiclass Classification (Çoklu Sınıf Sınıflandırma)
X_train, X_test, y_train, y_test = train_test_split(digits.data,  # Rakam veri setini eğitim ve test setlerine böler.
                                                    digits.target,  # Hedef değişken: Rakamlar (0-9).
                                                    random_state=0  # Rastgeleliği sabitler.
                                                    )

lr = LogisticRegression(solver='lbfgs',  # Lojistik regresyon 'lbfgs' çözücü ile.
                        max_iter=1000,  # Varsayılan 100'dür, artırıyoruz
                        random_state=42  # random_state parametresi, modelin rastgele süreçlerinde tutarlılık sağlamak için kullanılan bir tohum (seed) değeridir; aynı değer kullanıldığında, model her çalıştırıldığında aynı sonuçları üretir.
                        ).fit(X_train, y_train)

pred = lr.predict(X_test)  # Test seti üzerindeki tahminler.

# Doğruluk skoru hesaplama
ascore = accuracy_score(y_true=y_test,  # Gerçek sınıf etiketleri.
                        y_pred=pred  # Tahmin edilen sınıf etiketleri.
                        )
print(ascore)  # Doğruluk skoru.

# Karışıklık matrisi (Confusion Matrix)
cm = confusion_matrix(y_true=y_test,  # Gerçek sınıf etiketleri.
                      y_pred=pred  # Tahmin edilen sınıf etiketleri.
                      )
print(cm)

# Karışıklık matrisini bir pandas DataFrame olarak oluşturma ve görselleştirme.
df = pd.DataFrame(data=cm,  # Karışıklık matrisi.
                  index=digits.target_names,  # Satır etiketleri (gerçek sınıflar).
                  columns=digits.target_names  # Sütun etiketleri (tahmin edilen sınıflar).
                  )

# Seaborn ile karışıklık matrisini ısı haritası olarak çizdirme.
sns.heatmap(data=df,  # Karışıklık matrisi verisi.
            annot=True,  # Hücrelere sayı ekle.
            cbar=None,  # Renk skalasını gösterme.
            cmap='Blues'  # Renk haritası olarak 'Blues' kullanılır.
            )

plt.title('Confusion Matrix')  # Grafik başlığı.
plt.ylabel('True Class')  # Y ekseni etiketi.
plt.xlabel('Predicted Class')  # X ekseni etiketi.
plt.show()  # Grafiği göster.

# SVC(kernel='linear', C=1):
# kernel='linear': Lineer karar sınırları oluşturmak için kullanılır.
# C=1: Hata toleransı (daha düşük C, daha fazla hata payına izin verir).

# DummyClassifier(strategy='most_frequent'):
# strategy='most_frequent': En sık görülen sınıfı tahmin eder, bu basit bir referans modeldir.

# confusion_matrix(y_true, y_pred):
# y_true: Gerçek sınıf etiketleri.
# y_pred: Modelin tahmin ettiği sınıf etiketleri.
# Çıktı: Doğru/yanlış tahminleri gösteren bir karışıklık matrisi.

# roc_curve(y_true, y_score):
# y_true: Gerçek sınıf etiketleri.
# y_score: Modelin karar fonksiyonu skorları.
# Çıktı: ROC eğrisi için yanlış pozitif ve doğru pozitif oranlarını hesaplar.

# accuracy_score(y_true, y_pred):
# Doğruluk skorunu hesaplar: Gerçek etiketlerle tahmin edilen etiketlerin ne kadar uyuştuğunu gösterir.

# Seaborn ve sns.heatmap ile karışıklık matrisi görselleştirme: Matrisin ısı haritasını oluşturur.

# Bu kod parçasında, Destek Vektör Makineleri (SVC), DummyClassifier, ve Lojistik Regresyon kullanılarak sınıflandırma modelleri eğitiliyor.
# Ayrıca, confusion matrix (karışıklık matrisi), ROC eğrisi, ve diğer sınıflandırma metrikleri hesaplanıyor.
