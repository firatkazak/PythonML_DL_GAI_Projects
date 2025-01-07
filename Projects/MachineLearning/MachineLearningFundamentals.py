import pandas as pd  # Veri işleme ve analiz için pandas kütüphanesi içe aktarılır.
import seaborn as sns  # Veri görselleştirme için seaborn kütüphanesi içe aktarılır.
import matplotlib.pyplot as plt  # Grafik çizimi için matplotlib kütüphanesi içe aktarılır.
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için fonksiyon içe aktarılır.
from sklearn.linear_model import LinearRegression  # Lineer regresyon modeli oluşturmak için sınıf içe aktarılır.
from sklearn.metrics import mean_squared_error  # Ortalama kare hatasını hesaplamak için fonksiyon içe aktarılır.
import math  # Matematiksel işlemler (karekök gibi) için math kütüphanesi içe aktarılır.

data = pd.read_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/insurance.csv")
# CSV dosyasını pandas DataFrame'e yükler. Bu veri sigorta bilgilerini içeriyor olabilir.

print(data.head())  # Verinin ilk 5 satırını yazdırır.
print(data.shape)  # Verinin satır ve sütun sayısını (boyutunu) yazdırır.
print(data.info())  # Verinin genel bilgilerini (sütun türleri, boş değerler, boyutlar) yazdırır.
print(data.isnull().sum())  # Her sütunda kaç tane eksik (NaN) veri olduğunu yazdırır.
print(data.dtypes)  # Sütunların veri tiplerini yazdırır.

data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
# Sigara içme durumunu sayısal veriye dönüştürür: 'yes' -> 1, 'no' -> 0.

data['sex'] = data['sex'].map({'male': 1, 'female': 0})
# Cinsiyet bilgisini sayısal veriye dönüştürür: 'male' -> 1, 'female' -> 0.

data['region'] = data['region'].astype('category')
# Bölge bilgisini kategorik bir veri tipine dönüştürür, çünkü bölge sayısal değil sınıflandırılabilir bir değerdir.

print(data.dtypes)  # Veri türlerini tekrar yazdırır, yapılan dönüşümleri kontrol eder.
print(data.describe())  # Sayısal veriler için özet istatistikleri yazdırır (ortalama, standart sapma vb.).
print(data.describe().T)  # Özet istatistiklerin transpozesini alarak satır ve sütunları ters çevirir ve daha okunaklı hale getirir.

numeric_data = data.select_dtypes(include=['number'])
# Sadece sayısal sütunları seçer, çünkü daha sonra bunlarla işlemler yapılacak.

smoke_data = numeric_data.groupby(data['smoker']).mean().round(3)
# Sigara içen ve içmeyenler için sayısal sütunların ortalamalarını hesaplar ve 3 ondalık basamağa yuvarlar.
print(smoke_data)  # Sigara durumuna göre gruplandırılmış verileri yazdırır.

sns.set_style("whitegrid")  # Seaborn ile çizilecek grafiklerin arka plan stilini beyaz kareli olarak ayarlar.

sns.pairplot(
    data=data[["age", "bmi", 'expenses', "smoker"]],
    hue="smoker",  # Renkleri 'smoker' sütununa göre ayırır.
    height=3,  # Her bir grafiğin boyutunu 3 birim olarak ayarlar.
    palette="Set1"  # Renk paletini 'Set1' olarak ayarlar.
)  # Verilen sütunlar arasında çiftli grafikler (pairplot) oluşturur ve sigara içme durumuna göre renklendirir.

plt.show()  # Oluşturulan grafikleri ekranda gösterir.

numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Sayısal veri türlerini seçer.
corr_matrix = numeric_data.corr()  # Sayısal verilerin korelasyon matrisini hesaplar.

sns.heatmap(
    data=corr_matrix,  # Korelasyon matrisini görselleştirir.
    annot=True,  # Hücrelerin içinde korelasyon değerlerini gösterir.
    cmap='coolwarm'  # 'coolwarm' renk paletini kullanarak grafiği renklendirir.
)  # Korelasyon matrisini ısı haritası olarak çizer.

plt.show()  # Çizilen ısı haritasını ekranda gösterir.

data = pd.get_dummies(data)  # Kategorik değişkenleri one-hot encoding ile sayısal verilere çevirir.
print(data.columns)  # Yeni oluşturulan sütunların isimlerini yazdırır.

y = data['expenses']  # Hedef değişken olarak 'expenses' (harcamalar) sütununu ayırır.

X = data.drop(
    labels='expenses',  # 'expenses' sütununu bağımsız değişkenlerden çıkarır.
    axis=1  # 1: sütunu ifade eder.
)  # Bağımsız değişkenler olarak kalan tüm sütunları seçer.

X_train, X_test, y_train, y_test = train_test_split(
    X,  # Bağımsız değişkenler (özellikler).
    y,  # Bağımlı değişken (hedef).
    train_size=0.8,  # Verinin %80'i eğitim için, %20'si test için ayrılır.
    random_state=1  # Rastgele bölmeyi kontrol eder, böylece her çalışmada aynı bölme elde edilir.
)

lr = LinearRegression()  # Lineer regresyon modelini oluşturur.
lr.fit(X_train, y_train)  # Modeli eğitim verileri üzerinde eğitir (fit eder).
sonuc1 = lr.score(X_test, y_test).__round__(3)  # Test setinde modelin başarımını (R² skoru) alır ve 3 basamaklı yuvarlar.
print(sonuc1)  # Modelin R² skorunu yazdırır.
lr.score(X_train, y_train).__round__(3)  # Eğitim setindeki model başarımını (R² skoru) hesaplar.

y_pred = lr.predict(X_test)  # Test setindeki bağımlı değişkenlerin tahminlerini yapar.

sonuc2 = math.sqrt(mean_squared_error(y_test, y_pred))  # Test setindeki tahminlerin ortalama kare hatasının karekökünü hesaplar.
print(sonuc2)  # RMSE (root mean squared error) değerini yazdırır.

data_new = X_train[:1]  # Eğitim setindeki ilk satırı alır.
print(data_new)  # İlk satırı yazdırır.

print(lr.predict(data_new))  # Modelin ilk satır için yaptığı tahmini yazdırır.
print(y_train[:1])  # Gerçek 'expenses' değerini (ilk satır) yazdırır.
# ML Temellerini öğrendik.
