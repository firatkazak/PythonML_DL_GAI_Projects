import pandas as pd  # Veri işleme için kullanılıyor.
import seaborn as sns  # Veri görselleştirme kütüphanesi.
import matplotlib.pyplot as plt  # Grafik çizmek için kullanılıyor.
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak ayırmak için.
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon modeli oluşturmak için.
from sklearn.metrics import confusion_matrix  # Karışıklık matrisi hesaplamak için.

# Titanic veri setini CSV dosyasından okuma
veri = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/titanic.csv")

# 'survived' sütununa göre bir sayım grafiği oluşturuyor.
sns.countplot(x="survived", data=veri, hue="survived", palette={0: "red", 1: "blue"}, legend=False)
plt.show()

# 'survived' ve 'pclass' sütunlarına göre bir sayım grafiği oluşturuyor.
sns.countplot(x="survived",
              data=veri,
              hue="pclass",
              palette={1: "gold", 2: "silver", 3: "brown"})
plt.show()

# 'age' sütunundaki verilerin histogramını çiziyor.
veri['age'].plot.hist()
plt.show()

# 'fare' sütunundaki verilerin histogramını, 20 aralığa bölerek çiziyor.
veri['fare'].plot.hist(bins=20, figsize=(10, 5))
plt.show()

# Eksik veri analizi için ısı haritası çiziyor.
sns.heatmap(veri.isnull(), yticklabels=False, cmap='viridis')
plt.show()

# Gereksiz sütunlar listesi.
silinecek_sutunlar = ["cabin", "boat", "body", "home.dest"]

# Mevcut veri setinde olan ve silinecek sütunları kontrol ediyor.
mevcut_sutunlar = [sutun for sutun in silinecek_sutunlar if sutun in veri.columns]

# Bu sütunları veri setinden kaldırıyor.
veri.drop(mevcut_sutunlar, axis=1, inplace=True)

# Eksik verileri veri setinden kaldırıyor.
veri.dropna(inplace=True)

# Kategorik değişkenleri dummy değişkenlere çeviriyor ve ilk sütunu düşürüyor.
veri = pd.get_dummies(veri, drop_first=True)

# Bağımsız değişkenleri X'e, bağımlı değişkeni y'ye atıyor.
X = veri.drop("survived", axis=1)  # "survived" dışındaki tüm sütunlar.
y = veri["survived"]  # Hayatta kalma durumu (0 veya 1).

# X'in sütun isimlerini string yapıyor, hata almamak için.
X.columns = X.columns.astype(str)

# Veriyi eğitim ve test seti olarak böler (test seti %25, eğitim seti %75).
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=100)

# Lojistik regresyon modeli oluşturuyor, max_iter parametresi iterasyon sayısını belirtir.
lg_model = LogisticRegression(max_iter=1000)
lg_model.fit(X_train, y_train)  # Modeli eğitim verisiyle eğitiyor.

# Eğitim ve test seti üzerindeki doğruluk skorlarını yazdırıyor.
print("Test set skoru:", lg_model.score(X_test, y_test))  # Test setindeki başarı oranı. Test set skoru: 0.8304093567251462
print("Train set skoru:", lg_model.score(X_train, y_train))  # Eğitim setindeki başarı oranı. Train set skoru: 0.9473684210526315

# Regularization gücünü düşüren (C=0.1) bir model oluşturuyor ve eğitiyor.
lg_model = LogisticRegression(C=0.1, max_iter=1000)  # C, regularization gücünü ayarlar.
lg_model.fit(X_train, y_train)

# Test seti üzerindeki skoru yazdırıyor.
print("Test set skoru (C=0.1):", lg_model.score(X_test, y_test))  # Test set skoru (C=0.1): 0.8128654970760234

# Model ile test seti üzerinde tahmin yapıyor.
tahmin = lg_model.predict(X_test)

# Tahminler ile gerçek sonuçlar arasındaki karışıklık matrisini yazdırıyor.
print(confusion_matrix(y_test, tahmin))  # confusion_matrix, doğru ve yanlış sınıflandırmaları gösterir.
# [[80 10]
#  [22 59]]
