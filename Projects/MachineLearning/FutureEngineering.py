from sklearn.feature_extraction import DictVectorizer  # Sözlük veri yapısını sayısal verilere dönüştüren sınıfı içe aktarır.
from sklearn.feature_extraction.text import CountVectorizer  # Metin verilerini kelime sıklığına göre vektörize eden sınıfı içe aktarır.
from sklearn.linear_model import LinearRegression  # Lineer regresyon modelini oluşturmak için sınıfı içe aktarır.
from sklearn.preprocessing import PolynomialFeatures  # Polinom özellikleri oluşturan sınıfı içe aktarır.
from sklearn.impute import SimpleImputer  # Eksik verileri tamamlayan sınıfı içe aktarır.
from sklearn.pipeline import make_pipeline  # Bir dizi işleme ardışık olarak boru hattı (pipeline) oluşturmak için sınıfı içe aktarır.
from numpy import nan  # NumPy'den eksik veri için kullanılan 'NaN' değeri içe aktarılır.
import pandas as pd  # Veri işleme için pandas kütüphanesi içe aktarılır.
import numpy as np  # Sayısal işlemler için NumPy kütüphanesi içe aktarılır.
import matplotlib.pyplot as plt  # Grafik çizimi için Matplotlib kütüphanesi içe aktarılır.

data = [
    {'not': 85, 'kardes': 4, 'ders': 'mat'},
    {'not': 70, 'kardes': 3, 'ders': 'ing'},
    {'not': 65, 'kardes': 3, 'ders': 'mat'},
    {'not': 60, 'kardes': 2, 'ders': 'fiz'}
]  # Öğrenci notları, kardeş sayısı ve ders adları içeren bir sözlük listesi oluşturur.

vek1 = DictVectorizer(sparse=True, dtype=int)  # DictVectorizer: Sözlük formatındaki veriyi sayısal verilere dönüştürür.
# Parametreler:
# sparse=True: Sonuç sparse (seyrek) bir matris olur, belleği daha az kullanır.
# dtype=int: Çıktıdaki sayılar tam sayı formatında olur.
sonuc1 = vek1.fit_transform(data)  # Veriyi sayısal forma dönüştürür ve sparse matris elde eder.
print(sonuc1)  # Dönüştürülmüş veriyi yazdırır.
sonuc2 = vek1.get_feature_names_out()  # Dönüştürmede kullanılan özelliklerin isimlerini alır.
print(sonuc2)  # Özellik isimlerini yazdırır.

veri = [
    'hava iyi',
    'iyi insan',
    'hava bozuk',
]  # Metin verileri içeren bir liste oluşturur.

vek2 = CountVectorizer()  # CountVectorizer: Metni kelime sıklığına dayalı sayısal vektörlere dönüştürür.
X = vek2.fit_transform(veri)  # Metin verisini vektörleştirir (kelime frekansına dayalı).
print(X)  # Seyrek matris olarak vektörleştirilmiş veriyi yazdırır.

sonuc3 = pd.DataFrame(X.toarray(), columns=vek2.get_feature_names_out())  # Kelime frekanslarını bir DataFrame'e dönüştürür.
print(sonuc3)  # Vektörleştirilmiş veriyi tablo formatında yazdırır.

x = np.array([1, 2, 3, 4, 5])  # NumPy dizisi olarak bağımsız değişkenler oluşturur.
y = np.array([5, 3, 1, 2, 7])  # NumPy dizisi olarak bağımlı değişkenler oluşturur.

plt.scatter(x, y)  # x ve y değişkenlerinin dağılım grafiğini çizer.
plt.show()  # Grafiği ekranda gösterir.

X = x[:, np.newaxis]  # x dizisini sütun vektörüne çevirir.
model = LinearRegression().fit(X, y)  # Lineer regresyon modelini oluşturur ve veriye uyarlar (fit eder).
y_fit = model.predict(X)  # Modelin tahmin ettiği y değerlerini alır.
plt.scatter(x, y)  # Orijinal veriyi dağılım grafiği olarak çizer.
plt.plot(x, y_fit)  # Tahmin edilen doğrusal eğrinin grafiğini çizer.
plt.show()  # Grafiği gösterir.

pol = PolynomialFeatures(degree=3, include_bias=False)  # Polinom özellikleri oluşturur.
# Parametreler:
# degree=3: 3. dereceden polinom özellikler oluşturur.
# include_bias=False: Sabit terim (bias) eklenmez, sadece özellikler dönüşür.
X2 = pol.fit_transform(X)  # x verisine 3. dereceden polinom dönüşümü uygular.
print(X2)  # Dönüşüm sonucu elde edilen polinom özelliklerini yazdırır.

model = LinearRegression().fit(X2, y)  # Polinom özelliklerle lineer regresyon modelini oluşturur ve eğitir.
y_fit = model.predict(X2)  # Modelin tahmin ettiği y değerlerini alır.
plt.scatter(x, y)  # Orijinal veriyi dağılım grafiği ile çizer.
plt.plot(x, y_fit)  # Tahmin edilen polinom eğrinin grafiğini çizer.
plt.show()  # Grafiği gösterir.

X = np.array([[1, nan, 3], [5, 6, 9], [4, 5, 2], [4, 6, nan], [9, 8, 1]])  # Eksik veriler (NaN) içeren bir NumPy dizisi oluşturur.
y = np.array([10, 13, -2, 7, -6])  # Bağımlı değişkenleri içeren NumPy dizisi oluşturur.
imp = SimpleImputer(strategy='mean')  # Eksik değerleri doldurur.
# Parametre:
# strategy='mean': Eksik değerler sütunların ortalama değeri ile doldurulur.
X2 = imp.fit_transform(X)  # Eksik değerler ortalama ile doldurulmuş veriyi oluşturur.

model = LinearRegression().fit(X2, y)  # Eksik değerler doldurulduktan sonra lineer regresyon modelini eğitir.
sonuc4 = model.predict(X2)  # Modelin tahmin ettiği y değerlerini alır.
print(sonuc4)  # Tahmin edilen y değerlerini yazdırır.

# Pipeline;
model = make_pipeline(SimpleImputer(strategy='mean'), PolynomialFeatures(degree=2),
                      LinearRegression())  # Pipeline: Eksik verileri doldurup polinom özellik oluşturur ve regresyonu uygular.
# Parametreler:
# strategy='mean': Eksik veriler ortalama ile doldurulur.
# degree=2: 2. dereceden polinom özellikler oluşturulur.
model.fit(X, y)  # Pipeline'ı veriye uyarlar (fit eder).
print(y)  # Bağımlı değişken y'yi yazdırır.
print(model.predict(X))  # Pipeline ile tahmin edilen y değerlerini yazdırır.
# Feature Engineering konusunu işledik.
