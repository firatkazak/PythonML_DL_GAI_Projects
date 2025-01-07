import pandas as pd  # Veri işleme ve analiz için pandas kütüphanesi.
import numpy as np  # Sayısal işlemler için numpy kütüphanesi.
from sklearn.model_selection import train_test_split  # Veri kümesini eğitim ve test setlerine ayırmak için kullanılır.
from sklearn.preprocessing import MinMaxScaler  # Veriyi minimum ve maksimum değerler arasında ölçeklendirmek için kullanılır.
from sklearn.preprocessing import StandardScaler  # Veriyi standartlaştırmak için kullanılır (ortalama 0, standart sapma 1).
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon modeli kurmak için kullanılır.
from sklearn.ensemble import RandomForestClassifier  # Rastgele orman sınıflandırıcı modelini kullanmak için.
from sklearn.feature_selection import SelectFromModel  # Özellik seçiminde kullanılacak model.

# Veriyi CSV dosyasından okur, header parametresi None çünkü sütun başlıkları yok.
df = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/wine/wine.data", header=None)

# Sütun isimlerini atar.
df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
              'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
              'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# İlk 5 satırı yazdırır.
print(df.head())

# 'Class label' sütunundaki benzersiz etiketleri yazdırır (3 farklı sınıf var).
print(np.unique(df["Class label"]))

# Bağımsız değişkenleri (X) ve bağımlı değişkeni (y) ayırır. X, tüm sütunlar dışında ilk sütun.
X = df.iloc[:, 1:].values  # Özellikler
y = df.iloc[:, 0].values  # Etiketler (Class label)

# Veriyi %75 eğitim ve %25 test olarak ikiye böler, stratify parametresi ile sınıf dengesini korur.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# **Veri Ölçeklendirme** (Normalizasyon ve Standartlaştırma)

# MinMaxScaler ile veriyi 0 ve 1 aralığına ölçeklendirir.
mmscaler = MinMaxScaler()

# Eğitim verisini ölçeklendirir ve dönüştürür.
X_train_norm = mmscaler.fit_transform(X_train)

# Test verisini de aynı ölçekle dönüştürür.
X_test_norm = mmscaler.transform(X_test)

# StandardScaler ile veriyi standartlaştırır (ortalama 0, standart sapma 1).
stdscaler = StandardScaler()

# Eğitim verisini standartlaştırır.
X_train_std = stdscaler.fit_transform(X_train)

# Test verisini de aynı ölçekle standartlaştırır.
X_test_std = stdscaler.transform(X_test)

# **Lojistik Regresyon (Regularization)**

# Lojistik regresyon modeli oluşturur. L1 cezası (Lasso) ile parametrelerin bazılarını sıfıra yakınlaştırır.
# C: Düzenlileştirme katsayısı, daha büyük C değerleri daha az düzenleme anlamına gelir.
# solver: 'liblinear', küçük veri setleri için kullanışlıdır. multi_class: 'ovr' sınıflandırmayı bire bir yapar (one-vs-rest).
lr = LogisticRegression(penalty="l1", C=1, solver="liblinear", multi_class="ovr")

# Modeli standartlaştırılmış eğitim verisi ile eğitir.
print(lr.fit(X_train_std, y_train))

# Eğitim verisi üzerinde modelin doğruluğunu yazdırır.
print(lr.score(X_train_std, y_train))

# Test verisi üzerinde modelin doğruluğunu yazdırır.
print(lr.score(X_test_std, y_test))

# Modelin kesişim terimlerini (bias) yazdırır.
print(lr.intercept_)

# Modelin her bir özelliğe ait katsayılarını (coef) yazdırır.
print(lr.coef_)

# Katsayıları sıfır olmayan özelliklerin sayısını yazdırır (L1 ile bazı özelliklerin katsayıları sıfırlanmış olabilir).
print(lr.coef_[lr.coef_ != 0].shape)

# Örnek bir veri noktası için sınıf tahminini yazdırır (ilk veri noktası).
X_data = X_train_std[:1, ]  # İlk eğitim verisi örneği
print(lr.predict(X_data))  # Modelin tahmini
print(y_train[:1])  # Gerçek sınıf etiketi

# **Özellik Seçimi (Feature Engineering)**

# Rastgele orman sınıflandırıcı modeli oluşturur, 500 ağaç (estimators) kullanır.
forest = RandomForestClassifier(n_estimators=500, random_state=1)

# Özellik isimlerini (etiketlerini) alır.
feat_labels = df.columns[1:]

# Modeli eğitim verisi ile eğitir.
forest.fit(X_train, y_train)

# Her özelliğin önem derecesini hesaplar.
importances = forest.feature_importances_

# Özellik önem derecelerini sıralar (küçükten büyüğe).
indices = np.argsort(importances)

# Her bir özellik için sırayla isim ve önem derecesini yazdırır.
for f in range(X_train.shape[1]):
    print(f"{f + 1}", feat_labels[indices[f]], importances[indices[f]])

# En önemli özellikleri seçmek için SelectFromModel kullanır.
# threshold=0.1, öneme göre %10'dan yüksek olan özellikleri seçer.
selector = SelectFromModel(forest, threshold=0.1, prefit=True)

# Eğitim verisinden seçilen özellikleri dönüştürür (özellik sayısını azaltır).
X_selected = selector.transform(X_train)

# Seçilen özelliklerin isimlerini ve önem derecelerini yazdırır.
for f in range(X_selected.shape[1]):
    print(f"{f + 1}", feat_labels[indices[f]], importances[indices[f]])

# AÇIKLAMA
# Veri Ölçeklendirme: MinMaxScaler veriyi 0-1 aralığında ölçeklendirir, StandardScaler ise veriyi ortalama 0, standart sapma 1 olacak şekilde standartlaştırır.
# Bu, bazı algoritmaların daha iyi performans göstermesini sağlar.

# Lojistik Regresyon (Logistic Regression): L1 düzenlileştirme ile bazı özellikler sıfıra yakınlaştırılır. C katsayısı, düzenlileştirmenin gücünü kontrol eder.
# Özellik Seçimi (Feature Selection): Rastgele Orman (Random Forest) modeli ile her özelliğin önemi hesaplanır ve SelectFromModel ile en önemli özellikler seçilir.
