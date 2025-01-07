import pandas as pd  # Veri manipülasyonu ve analiz için kullanılır.
import numpy as np  # Sayısal işlemler için kullanılır.
from sklearn.impute import SimpleImputer  # Eksik verilerin yerine belirli stratejilerle veri doldurmak için kullanılır.
from sklearn.preprocessing import LabelEncoder  # Kategorik verileri sayısal verilere çevirmek için kullanılır.
from sklearn.preprocessing import OneHotEncoder  # Kategorik verileri One-Hot encoding ile sayısal verilere çevirmek için kullanılır.
from sklearn.compose import ColumnTransformer  # Birden fazla sütuna farklı dönüşümler uygulamak için kullanılır.

# NumPy dizisi oluşturuluyor. "A", "B", "C", "D" sütun başlıklarına sahip bir DataFrame yaratılıyor.
data = np.array([[1, 2, 3, 4], [5, 6, 7, np.nan], [9, 10, np.nan, 11]])
df = pd.DataFrame(data, columns=["A", "B", "C", "D"])

# DataFrame yazdırılıyor.
print(df)

# Hangi hücrelerde eksik veri olduğunu kontrol eder (True eksik, False eksik değil).
print(df.isnull())

# Her sütundaki eksik veri sayısını hesaplar.
print(df.isnull().sum())

# Eksik verisi olan satırları siler (axis=0 satır bazında işlem yapar).
print(df.dropna(axis=0))

# Eksik verisi olan sütunları siler (axis=1 sütun bazında işlem yapar).
print(df.dropna(axis=1))

# Tüm hücreleri eksik olan satırları siler (how="all" tüm değerlerin NaN olduğu satırları siler).
print(df.dropna(how="all"))

# En az 4 dolu hücresi olan satırları tutar (thresh=4).
print(df.dropna(thresh=4))

# Belirtilen sütunda eksik veri bulunan satırları siler (D sütunu eksik olanları siler).
print(df.dropna(subset=["D"]))

# Eksik değerleri sütunların ortalaması ile doldurmak için SimpleImputer oluşturuluyor (strategy="mean" ortalamayı kullan).
imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")

# SimpleImputer modelini DataFrame'deki verilere uydurur (fit işlemi).
imp_mean = imp_mean.fit(df.values)

# Eksik değerleri doldurup yeni bir veri kümesi oluşturur (transform işlemi).
imputed_data = imp_mean.transform(df.values)

# Eksik değerler doldurulmuş yeni veri yazdırılıyor.
print(imputed_data)

# Eksik verileri sütun ortalaması ile doldurur (fillna metodu).
print(df.fillna(df.mean()))

# Yeni bir DataFrame oluşturuluyor. Renk, beden, yaş ve sınıf etiketleri.
df = pd.DataFrame([["red", "L", 5.0, "class1"],
                   ["blue", "XL", 2.0, "class2"],
                   ["black", "M", 3.0, "class1"]])

# DataFrame'e sütun isimleri atanıyor.
df.columns = ["color", "size", "age", "classlabel"]

# DataFrame yazdırılıyor.
print(df)

# Sütunların veri tipleri yazdırılıyor.
print(df.dtypes)

# Bedenlerin sayısal değerlere dönüştürülmesi için bir haritalama yapılıyor.
size_mapping = {"XL": 3, "L": 2, "M": 1}

# "size" sütunu sayısal değerlere dönüştürülüyor (map metodu).
df["size"] = df["size"].map(size_mapping)
print(df)

# Sınıf etiketlerini (classlabel) sayısal değerlere çevirmek için bir haritalama oluşturuluyor.
class_mapping = {label: idx for idx, label in enumerate(np.unique(df["classlabel"]))}
print(class_mapping)

# "classlabel" sütunu bu haritaya göre güncelleniyor.
df["classlabel"] = df["classlabel"].map(class_mapping)
print(df)

# LabelEncoder sınıfı ile sınıf etiketleri sayısal değerlere çevriliyor.
class_le = LabelEncoder()

# Sınıf etiketleri sayısal hale getiriliyor (fit_transform metodu).
y = class_le.fit_transform(df["classlabel"].values)
print(y)

# Modelde kullanılacak bağımsız değişkenler (X) seçiliyor (color, size, age sütunları).
X = df[["color", "size", "age"]].values

# "color" sütunu LabelEncoder ile sayısal değerlere çevriliyor.
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# OneHotEncoder ile "color" sütunu One-Hot kodlaması yapılıyor.
color_ohe = OneHotEncoder()

# OneHotEncoder için reshape yapılarak "color" sütunu dönüştürülüyor.
X[:, 0].reshape(-1, 1)
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

# ColumnTransformer ile ilk sütuna (color) One-Hot kodlaması, diğerlerine herhangi bir işlem uygulanmıyor.
c_transf = ColumnTransformer([
    ("onehot", OneHotEncoder(), [0]),  # color sütununa One-Hot kodlama
    ("nothing", "passthrough", [1, 2])  # diğer sütunlara bir işlem yapılmıyor (passthrough).
])
print(X)

# ColumnTransformer ile dönüştürülmüş veri yazdırılıyor (float veri tipine çevriliyor).
print(c_transf.fit_transform(X).astype(float))

# Pandas get_dummies fonksiyonu ile "color", "size", "age" sütunlarına One-Hot kodlama uygulanıyor.
print(pd.get_dummies(df[["color", "size", "age"]]))

# İlk kategori dışındaki sütunlara One-Hot kodlama uygulanıyor (drop_first=True ile ilk kategori düşürülüyor).
print(pd.get_dummies(df[["color", "size", "age"]], drop_first=True))

# OneHotEncoder ile ilk kategori düşürülerek One-Hot kodlama yapılacak şekilde yeniden tanımlanıyor.
color_ohe = OneHotEncoder(categories="auto", drop="first")

# ColumnTransformer ile yeniden dönüştürme yapılıyor (ilk sütun "color" One-Hot, diğer sütunlar olduğu gibi bırakılıyor).
c_transf = ColumnTransformer([
    ("onehot", color_ohe, [0]),  # color sütununa One-Hot kodlama (ilk kategori hariç).
    ("nothing", "passthrough", [1, 2])  # diğer sütunlara bir işlem uygulanmıyor (passthrough).
])

# Dönüştürülmüş veri yazdırılıyor (float veri tipine çevriliyor).
print(c_transf.fit_transform(X).astype(float))
