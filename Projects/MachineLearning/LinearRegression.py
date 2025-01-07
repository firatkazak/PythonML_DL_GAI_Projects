import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# CSV dosyasını okuyup DataFrame'e aktarıyor
veri = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/student-mat.csv", sep=';')

# İlgili sütunları seçiyor
veri = veri[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences', 'age']]

# Sütun adlarını Türkçeleştiriyor
veri.rename(columns={
    'G1': 'Not 1',  # G1 yerine 'Not 1' olarak yeniden adlandırılıyor
    'G2': 'Not 2',  # G2 yerine 'Not 2' olarak yeniden adlandırılıyor
    'G3': 'Final',  # G3 yerine 'Final' olarak yeniden adlandırılıyor
    'studytime': 'Çalışma Süresi',  # studytime yerine 'Çalışma Süresi' olarak yeniden adlandırılıyor
    'failures': 'Sınıf Tekrarı',  # failures yerine 'Sınıf Tekrarı' olarak yeniden adlandırılıyor
    'absences': 'Devamsızlık',  # absences yerine 'Devamsızlık' olarak yeniden adlandırılıyor
    'age': 'Yaş',  # age yerine 'Yaş' olarak yeniden adlandırılıyor
}, inplace=True)

# Hedef değişken olan 'Final' sütununu numpy array'e dönüştürüyor
y = np.array(veri['Final'])

# 'Final' dışındaki sütunları bağımsız değişkenler (X) olarak alıyor
X = np.array(veri.drop('Final', axis=1))

# Veriyi eğitim ve test setlerine bölüyor
X_train, X_test, y_train, y_test = train_test_split(
    X,  # Bağımsız değişkenler (özellikler)
    y,  # Bağımlı değişken (hedef değişken)
    test_size=0.2,  # Verinin %20'si test seti olarak ayrılıyor
    random_state=2  # Rastgele bölünmeyi sabitlemek için sabit bir seed değeri
)

# Doğrusal regresyon modeli oluşturuluyor
linear = LinearRegression()

# Model, eğitim verisi üzerinde eğitiliyor (fit işlemi)
linear.fit(X_train, y_train)

# Modelin test seti üzerindeki doğruluk skorunu hesaplıyor (R-kare)
trainScore = linear.score(X_train, y_train)
testScore = linear.score(X_test, y_test)

# Train ve Test setindeki skor yazdırılıyor
print(trainScore)  # 0.8261275475197141
print(testScore)  # 0.8325898318712226
# Eğitim verisinin skoru Test verisinden yüksekse Overfitting, alçaksa Underfitting olur. Bu istenmeyen bir durumdur. İkisinin birbine yakın olması beklenir.

# Modelin katsayıları (özelliklerin her biri için ağırlık) yazdırılıyor
print('Katsayılar: ', linear.coef_)  # Katsayılar:  [ 0.19575962  0.95558174 -0.24215827  0.12730705  0.03566925 -0.23224281]

# Modelin sabit terimi (intercept) yazdırılıyor
print('Sabit: ', linear.intercept_)  # Sabit:  2.1549097406751443

# Yeni bir veri üzerinde tahmin yapılıyor (Tahmin edilen Not 20 üzerinden)
yeni_veri = np.array([[20, 20, 10, 0, 0, 14]])
# Yukarıda sırayla 1. not, 2. not, Çalışma süresi, Sınıf tekrarı, Devamsızlık ve Yaş verileri ile tahmin yapıldı.

# Yeni verinin tahmin sonucu yazdırılıyor
sonuc = linear.predict(yeni_veri)
print(sonuc)  # [19.50875499] 20 üstünden 19 küsur not alması bekleniyor bu öğrencinin.
