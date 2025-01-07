import pandas as pd  # Veri analizi ve manipülasyonu için pandas kütüphanesini içe aktar.
from sklearn.impute import SimpleImputer  # Eksik verileri doldurmak için kullanılır.
from sklearn.pipeline import Pipeline  # İşlem adımlarını bir araya getirip yönetmek için kullanılır.
from sklearn.preprocessing import OneHotEncoder  # Kategorik verileri One-Hot kodlama ile sayısal verilere çevirir.
from sklearn.compose import ColumnTransformer  # Farklı sütunlara farklı dönüşümler uygular.
from sklearn.model_selection import train_test_split  # Veri kümesini eğitim ve test setlerine ayırmak için kullanılır.
from sklearn.ensemble import RandomForestRegressor  # Rastgele orman algoritması ile regresyon modeli oluşturur.
from sklearn.metrics import mean_absolute_error  # Model performansını değerlendirmek için mutlak hata hesaplar.
import matplotlib.pyplot as plt  # Veri görselleştirmesi için matplotlib kütüphanesi.

# Eğitim verilerini dosyadan okur ve "Id" sütununu indeks olarak kullanır.
train_data = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/HousePrices/train.csv",
                         index_col="Id")

# İlk 5 satırı yazdırır.
print(train_data.head())

# DataFrame'in boyutunu (satır, sütun) yazdırır.
print(train_data.shape)

# "SalePrice" sütunu hariç tüm sütunları X değişkenine atar (bağımsız değişkenler).
X = train_data.drop(["SalePrice"], axis=1)

# "SalePrice" sütununu y değişkenine atar (bağımlı değişken).
y = train_data.SalePrice

# X ve y'nin boyutlarını yazdırır.
print(X.shape)
print(y.shape)

# X'in yapısını ve her sütunun veri tipini yazdırır.
print(X.info())

# Her sütundaki eksik değer sayısını yazdırır.
print(X.isnull().sum())

# Eksik değer içeren sütun sayısını yazdırır.
print((X.isnull().sum() != 0).sum())

# Sayısal veri tipine sahip sütunları seçer.
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]]

# Kategorik veri tipine sahip, 10'dan az benzersiz değere sahip sütunları seçer.
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and X[cname].dtype == "object"]

# Seçilen sütunları birleştirir.
my_cols = categorical_cols + numerical_cols

# Seçilen sütunlardan yeni bir DataFrame oluşturur.
X_data = X[my_cols].copy()

# Yeni DataFrame'in boyutunu yazdırır.
print(X_data.shape)

# Sayısal veriler için ortalama stratejisi ile eksik değerleri doldurmak üzere SimpleImputer oluşturur.
numirerical_transformer = SimpleImputer(strategy="median")

# Kategorik veriler için bir işlem hattı (Pipeline) oluşturur.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # En sık görülen değer ile eksik verileri doldur.
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # Kategorik verileri One-Hot kodlaması yap.
])

# Sayısal ve kategorik dönüşümleri bir araya getiren ColumnTransformer oluşturur.
preprocessor = ColumnTransformer(transformers=[
    ("num", numirerical_transformer, numerical_cols),  # Sayısal sütunlar için dönüşüm.
    ("cat", categorical_transformer, categorical_cols)  # Kategorik sütunlar için dönüşüm.
])

# X_data'yı ön işleme tabi tutarak dönüştürülmüş veri kümesini oluşturur.
X_data_pre = preprocessor.fit_transform(X_data)

# Dönüştürülmüş veri kümesinin boyutunu yazdırır.
print(X_data_pre.shape)

# RandomForestRegressor modeli oluşturur, 100 ağaç kullanır.
model = RandomForestRegressor(n_estimators=100, random_state=1)

# İşlem hattı oluşturur, ön işleme ve model adımlarını bir araya getirir.
my_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),  # Ön işleme adımı.
    ("model", model)  # Model adımı.
])

# Eğitim ve test setlerine ayırır (%20 test seti).
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2)

# Pipeline'ı eğitim verisi ile eğitir.
print(my_pipeline.fit(X_train, y_train))

# Test setindeki tahminleri yapar.
preds_test = my_pipeline.predict(X_test)

# Test seti için mutlak hata değerini yazdırır.
print(mean_absolute_error(preds_test, y_test))

# Eğitim setindeki tahminleri yapar.
preds_train = my_pipeline.predict(X_train)

# Eğitim seti için mutlak hata değerini yazdırır.
print(mean_absolute_error(preds_train, y_train))

from sklearn.model_selection import cross_val_score  # Çapraz doğrulama için kullanılır.


# Belirtilen sayıda ağaç ile modelin ortalama hatasını döndüren bir fonksiyon.
def get_score(n_estimators):
    my_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),  # Ön işleme adımı.
        ("model", RandomForestRegressor(n_estimators, random_state=1))  # Model adımı (ağaç sayısı belirtilir).
    ])
    scores = -1 * cross_val_score(my_pipeline, X_data, y, cv=5,  # 5 katlı çapraz doğrulama yapar.
                                  scoring="neg_mean_absolute_error")  # Negatif mutlak hata hesaplar.
    return scores.mean()  # Ortalama hatayı döndürür.


# Ağaç sayısına göre sonuçları saklamak için bir sözlük oluşturur.
result = {}

# 2 ile 8 arasında döngü başlatır ve her döngüde 50'nin katlarını kullanır.
for i in range(2, 8):
    result[50 * i] = get_score(50 * i)  # Ağaç sayısını belirler ve hata değerini alır.

# Sonuçları çizgi grafiği olarak görselleştirir.
plt.plot(list(result.keys()), list(result.values()))
plt.show()

# RandomForestRegressor modeli, 350 ağaç kullanarak yeniden oluşturuluyor.
model = RandomForestRegressor(n_estimators=350, random_state=1)

# Pipeline oluşturuluyor.
my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# Tüm veriyi (X_data) ile modeli eğitir.
print(my_pipeline.fit(X_data, y))

# Test verisini dosyadan okur.
test_data = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/HousePrices/test.csv")

# Test verisinden seçilen sütunları alır.
X_test = test_data[my_cols].copy()

# Test verisi için tahminleri yapar.
preds_test = my_pipeline.predict(X_test)

# Tahminleri yazdırır.
print(preds_test)
