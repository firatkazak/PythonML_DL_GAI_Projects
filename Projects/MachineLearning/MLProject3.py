import pandas as pd  # Veri işleme ve analiz için kullanılır
import matplotlib.pyplot as plt  # Veri görselleştirme için kullanılır
import seaborn as sns  # İstatistiksel veri görselleştirme için kullanılır
import numpy as np  # Sayısal hesaplamalar ve matris işlemleri için kullanılır

# CSV dosyasını okur ve 'data' isimli bir DataFrame'e dönüştürür
data = pd.read_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/data.csv")

# İlk 5 satırı gösterir
print(data.head())

# Veri setinin boyutlarını (satır, sütun) gösterir
print(data.shape)

# Veri setindeki sütun isimlerini gösterir
print(data.columns)

# Sütun isimlerini küçük harfe çevirir ve aralarındaki boşlukları "_" ile değiştirir
data.columns = data.columns.str.lower().str.replace(" ", "_")
print(data.columns)

# Sütunlardaki veri türlerini gösterir
print(data.dtypes)

# Metin (object) veri türüne sahip sütunların isimlerini alır ve listeye dönüştürür
str_columns = list(data.dtypes[data.dtypes == "object"].index)
print(str_columns)

# Metin sütunlarındaki verileri küçük harfe çevirir ve boşlukları "_" ile değiştirir
for col in str_columns:
    data[col] = data[col].str.lower().str.replace(" ", "_")
data.head()

# Seaborn tema stilini ayarlar
sns.set_style("whitegrid")

# MSRP (aracın fiyatı) dağılımını histogram olarak çizer (40 aralık)
sns.histplot(data.msrp, bins=40)
plt.show()

# MSRP < 100000 olanları filtreleyerek histogram çizer (40 aralık)
sns.histplot(data.msrp[data.msrp < 100000], bins=40)
plt.show()

# MSRP sütununa log1p (log(1+x)) dönüşümü uygular ve histogramını çizer
log_data = np.log1p(data.msrp)
sns.histplot(log_data)
plt.show()

# Eksik (NaN) değerlerin sütun başına sayısını gösterir
print(data.isnull().sum())

# Veri setindeki satır sayısını alır
n = len(data)

# %15 doğrulama ve %15 test için ayırma işlemi yapar, geri kalan %70 eğitim verisidir
n_val = int(0.15 * n)
n_test = int(0.15 * n)
n_train = n - (n_val + n_test)

# Rastgele veri karıştırma için tohum belirler
np.random.seed(42)

# Satır indekslerini oluşturur ve karıştırır
idx = np.arange(n)
np.random.shuffle(idx)
print(idx)

# Karışık veri setini oluşturur
data_shuffled = data.iloc[idx]
print(data_shuffled.head())

# Eğitim, doğrulama ve test veri setlerini oluşturur
data_train = data_shuffled.iloc[:n_train].copy()
data_val = data_shuffled.iloc[n_train:n_train + n_val].copy()
data_test = data_shuffled.iloc[n_train + n_val:].copy()
print(data_train.shape)

# MSRP sütununa log1p uygulayarak hedef (y) değerlerini oluşturur
y_train = np.log1p(data_train.msrp.values)
y_val = np.log1p(data_val.msrp.values)
y_test = np.log1p(data_test.msrp.values)

# Eğitim, doğrulama ve test veri setlerinden 'msrp' sütununu kaldırır
del data_train["msrp"]
del data_test["msrp"]
del data_val["msrp"]
print(data_train.columns)


# Basit doğrusal regresyon eğitimi yapan fonksiyon
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])  # Sabit birimler ekler
    X = np.column_stack([ones, X])  # Birim terimi X'in başına ekler
    XTX = X.T.dot(X)  # X^T * X hesaplar (çarpım)
    XTX_inv = np.linalg.inv(XTX)  # (X^T * X)^-1 alır (ters matris)
    w = XTX_inv.dot(X.T).dot(y)  # Ağırlık vektörünü hesaplar
    return w[0], w[1:]  # w[0] sabit terim, w[1:] regresyon katsayıları


# Kullanılacak temel özellikler
base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg", "popularity"]

# Temel özellikler seçilir, eksik değerler 0 ile doldurulur
data_num = data_train[base]
data_num = data_num.fillna(0)
print(data_num.isnull().sum())

# Özellikleri matrise dönüştürür ve model eğitimi yapar
X_train = data_num.values
w_0, w = train_linear_regression(X_train, y_train)

# Model tahminlerini hesaplar
y_pred = w_0 + X_train.dot(w)


# Kök Ortalama Kare Hatası (RMSE) hesaplayan fonksiyon
def rmse(y, y_pred):
    error = y_pred - y  # Gerçek ve tahmin değerleri arasındaki farkı alır
    mse = (error ** 2).mean()  # Hataların karelerinin ortalamasını alır
    return np.sqrt(mse)  # Kare kökünü alır ve RMSE'yi döndürür


# RMSE değeri yazdırılır
print(rmse(y_train, y_pred))

# Doğrulama verisinde model tahminleri hesaplanır
data_num = data_val[base]
data_num = data_num.fillna(0)
X_val = data_num.values
y_pred = w_0 + X_val.dot(w)
print(rmse(y_val, y_pred))


# Özellikleri hazırlayan fonksiyon
def prepare_X(data):
    data_num = data[base]  # Temel özellikler seçilir
    data_num = data_num.fillna(0)  # Eksik değerler 0 ile doldurulur
    X = data_num.values  # Numpy dizisine dönüştürülür
    return X


# Eğitim ve doğrulama verileri üzerinde model eğitimi ve RMSE hesaplama işlemleri tekrarlanır
X_train = prepare_X(data_train)
w_0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(data_val)
y_pred = w_0 + X_val.dot(w)
print("Validation:", rmse(y_val, y_pred))


# Yaş (age) özelliğini ekleyerek veri hazırlayan fonksiyon
def prepare_X(data):
    data = data.copy()
    feature = base.copy()  # Temel özellikler kopyalanır
    data["age"] = 2017 - data.year  # Yaş hesaplanır ve yeni bir sütun eklenir
    feature.append("age")  # Yaş özelliği listeye eklenir
    data_num = data[feature]
    data_num = data_num.fillna(0)
    X = data_num.values
    return X


# Yaş özelliği eklendikten sonra model eğitimi ve doğrulaması yapılır
X_train = prepare_X(data_train)
w_0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(data_val)
y_pred = w_0 + X_val.dot(w)
print("Validation:", rmse(y_val, y_pred))

# 'number_of_doors' (kapı sayısı) değerlerinin frekansını gösterir
data.number_of_doors.value_counts()


# Kapı sayısına ve bazı markalara göre özellikler oluşturan fonksiyon
def prepare_X(data):
    data = data.copy()
    features = base.copy()  # Temel özellikler kopyalanır
    data["age"] = 2017 - data.year  # Yaş hesaplanır ve yeni bir sütun eklenir
    features.append("age")  # Yaş özelliği eklenir

    # Kapı sayısına göre yeni özellikler ekler
    for v in [2, 3, 4]:
        feature = "num_doors_%s" % v
        data[feature] = (data["number_of_doors"] == v).astype(int)
        features.append(feature)

    # Belirli markalara göre yeni özellikler ekler
    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        data[feature] = (data['make'] == v).astype(int)
        features.append(feature)

    data_num = data[features]
    data_num = data_num.fillna(0)
    X = data_num.values
    return X


# Yeni özellikler eklendikten sonra model eğitimi ve doğrulama yapılır
X_train = prepare_X(data_train)
w_0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(data_val)
y_pred = w_0 + X_val.dot(w)
print("Validation:", rmse(y_val, y_pred))


# Daha fazla kategorik değişkene göre veri hazırlayan fonksiyon
def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)',
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)

    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)

    for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheens_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)

    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)

    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)

    for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


# Yeni özellikler ile model eğitimi ve doğrulama yapılır
X_train = prepare_X(data_train)
w_0, w = train_linear_regression(X_train, y_train)
X_val = prepare_X(data_val)
y_pred = w_0 + X_val.dot(w)
print("Validation:", rmse(y_val, y_pred))

# Eğitim veri setinin boyutlarını gösterir
print(X_train.shape)


# L2 regularizasyonlu doğrusal regresyon fonksiyonu
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])  # Sabit birimler ekler
    X = np.column_stack([ones, X])  # Birim terimi X'e ekler
    XTX = X.T.dot(X)  # X^T * X hesaplar
    reg = r * np.eye(XTX.shape[0])  # L2 regularizasyon matrisi ekler
    XTX = XTX + reg  # Regularizasyon eklenir
    XTX_inv = np.linalg.inv(XTX)  # Ters matris alınır
    w = XTX_inv.dot(X.T).dot(y)  # Ağırlık vektörünü hesaplar
    return w[0], w[1:]  # Sabit terim ve katsayılar döndürülür


# Farklı regularizasyon parametreleriyle modeli dener ve sonuçları yazdırır
for r in [0.001, 0.01, 0.1, 1, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    print("%5s, %.2f, %.2f, %.2f" % (r, w_0, w[13], w[21]))

# Son regularizasyon değeri ile tahmin yapılır
X_train = prepare_X(data_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.001)
X_val = prepare_X(data_val)
y_pred = w_0 + X_val.dot(w)
print("Validation:", rmse(y_val, y_pred))

# Farklı regularizasyon değerleriyle tahmin yapar ve RMSE'yi gösterir
X_train = prepare_X(data_train)
X_val = prepare_X(data_val)
for r in [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w_0 + X_val.dot(w)
    print('%6s' % r, rmse(y_val, y_pred))

# Regularizasyon ile eğitilen modelin eğitim verisindeki RMSE'sini hesaplar
X_train = prepare_X(data_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)
y_pred = w_0 + X_train.dot(w)
print("Validation:", rmse(y_train, y_pred))

# Yeni bir veri noktası oluşturur ve fiyat tahmini yapar
ad = {
    'city_mpg': 18,
    'driven_wheels': 'all_wheel_drive',
    'engine_cylinders': 6.0,
    'engine_fuel_type': 'regular_unleaded',
    'engine_hp': 268.0,
    'highway_mpg': 25,
    'make': 'BMV',
    'market_category': 'crossover,performance',
    'model': 'venza',
    'number_of_doors': 4.0,
    'popularity': 2031,
    'transmission_type': 'automatic',
    'vehicle_size': 'midsize',
    'vehicle_style': 'wagon',
    'year': 2017
}

# Yeni veriyi hazırlayıp tahmin yapar ve sonucu yazdırır
data_test = pd.DataFrame([ad])
X_test = prepare_X(data_test)
y_pred = w_0 + X_test.dot(w)
predict = np.expm1(y_pred)  # Log1p dönüşümünün tersini alır
print(predict)
