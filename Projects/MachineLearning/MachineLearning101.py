import pandas as pd  # Verilerle çalışmak için Pandas kütüphanesini içe aktarır.
import matplotlib.pyplot as plt  # Grafik oluşturmak için Matplotlib kütüphanesini içe aktarır.
import seaborn as sns  # Veri görselleştirme için Seaborn kütüphanesini içe aktarır.
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine bölmek için kullanılır.
from sklearn.feature_extraction import DictVectorizer  # Kategorik özellikleri sayısal değerlere dönüştürmek için kullanılır.
from sklearn.linear_model import LogisticRegression  # Lojistik regresyon modeli için kullanılır.

# Veri yükleme
df = pd.read_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.head())  # İlk 5 satırı ekrana yazdırır.

print(df.shape)  # Veri setinin boyutunu (satır ve sütun sayısını) yazdırır.
print(df.dtypes)  # Her bir sütunun veri türünü yazdırır.

# Seaborn ayarları ve tema
sns.set_theme()  # Seaborn için varsayılan tema ayarı.
sns.set(rc={"figure.figsize": (10, 6), "figure.dpi": 300})  # Grafik boyutu ve çözünürlük ayarları.

# Cinsiyet dağılımı (Gender)
x = round(df["gender"].value_counts() / df.shape[0] * 100, 2)  # Cinsiyetin yüzdesel dağılımını hesaplar.
plt.pie(  # Pie chart (pasta grafiği) oluşturur.
    x=x,
    labels=["Male", "Female"],  # Etiketler.
    explode=[0.1, 0],  # "Male" dilimini biraz ayırarak vurgular.
    autopct='%.2f%%'  # Yüzdeleri iki ondalık basamakla gösterir.
)
plt.legend()  # Grafik için bir gösterge ekler.
plt.show()  # Grafiği gösterir.

# Yaşlı vatandaş (SeniorCitizen) dağılımı
x = round(df["SeniorCitizen"].value_counts() / df.shape[0] * 100, 2)  # SeniorCitizen yüzdesini hesaplar.
plt.pie(
    x=x,
    labels=["Yes", "No"],  # Etiketler.
    explode=[0.1, 0],  # "Yes" dilimini biraz ayırarak vurgular.
    autopct='%.2f%%'
)
plt.legend()  # Gösterge ekler.
plt.show()  # Grafiği gösterir.

# Partner dağılımı
x = round(df["Partner"].value_counts() / df.shape[0] * 100, 2)  # Partner sütununun yüzdesini hesaplar.
plt.pie(
    x=x,
    labels=["Yes", "No"],  # Etiketler.
    explode=[0.1, 0],  # "Yes" dilimini biraz ayırarak vurgular.
    autopct='%.2f%%'
)
plt.legend()  # Gösterge ekler.
plt.show()  # Grafiği gösterir.

# Bağımlılar (Dependents) dağılımı
x = round(df["Dependents"].value_counts() / df.shape[0] * 100, 2)  # Dependents yüzdesini hesaplar.
plt.pie(
    x=x,
    labels=["Yes", "No"],  # Etiketler.
    explode=[0.1, 0],  # "Yes" dilimini biraz ayırarak vurgular.
    autopct='%.2f%%'
)
plt.legend()  # Gösterge ekler.
plt.show()  # Grafiği gösterir.

# MultipleLines (Birden fazla hat) dağılımı
x = round(df["MultipleLines"].value_counts() / df.shape[0] * 100, 2)  # MultipleLines sütununun yüzdesini hesaplar.
plt.pie(
    x=x,
    labels=["Yes", "No", "No phone service"],  # Etiketler.
    explode=[0.05, 0.05, 0.05],  # Tüm dilimleri hafifçe ayırır.
    autopct='%.2f%%'
)
plt.legend(loc='lower right')  # Göstergeyi grafiğin sağ alt köşesine koyar.
plt.show()  # Grafiği gösterir.

# Ödeme yöntemi (PaymentMethod) dağılımı
plt.figure(figsize=(10, 6))  # Grafik boyutunu belirler.
sns.countplot(  # Çubuk grafik oluşturur.
    x="PaymentMethod",  # Ödeme yöntemi sütununu kullanır.
    data=df  # Veri çerçevesi olarak df'yi kullanır.
)
plt.show()  # Grafiği gösterir.

# TotalCharges sütununu sayısal değere dönüştürme
df.TotalCharges = pd.to_numeric(  # TotalCharges sütununu sayısal bir türe çevirir.
    arg=df.TotalCharges,
    errors="coerce"  # Hata durumunda, sayısal olmayan verileri NaN yapar.
)
print(df.isnull().sum())  # Veri setindeki eksik değerlerin sayısını yazdırır.

# Eksik değerleri 0 ile doldurur
df.TotalCharges = df.TotalCharges.fillna(0)
print(df.isnull().sum().sum())  # Veri setindeki tüm eksik değerlerin toplam sayısını yazdırır (0 beklenir).

# Sütun adlarını küçük harflere ve alt çizgilere dönüştürme
df.columns = df.columns.str.lower().str.replace(" ", "_")  # Tüm sütun adlarını düzenler.
string_columns = list(df.dtypes[df.dtypes == "object"].index)  # Nesne türündeki sütunları listeler.

# Metinsel sütunlardaki boşlukları alt çizgiyle değiştirir ve küçük harfe çevirir
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(" ", "_")

print(df.head())  # Düzenlenmiş veri çerçevesinin ilk 5 satırını yazdırır.

# 'churn' sütununu bool (True/False) yapar
df.churn = (df.churn == "yes").astype(bool)
print(df.head())  # Güncellenmiş churn sütununu içeren ilk 5 satırı yazdırır.

# Veriyi eğitim ve test setlerine böler
df_train_full, df_test = train_test_split(  # Eğitim ve test setlerini böler.
    df,
    test_size=0.2,  # %20'lik kısmı test seti olarak ayırır.
    random_state=42  # Rastgelelik için sabit bir değer belirler.
)
df_train, df_val = train_test_split(  # Eğitim setini yeniden eğitim ve doğrulama setlerine böler.
    df_train_full,
    test_size=0.25,  # %25'ini doğrulama seti olarak ayırır.
    random_state=42  # Rastgelelik için sabit bir değer belirler.
)
y_train = df_train.churn.values  # Eğitim setindeki hedef değişkeni ayırır.
y_val = df_val.churn.values  # Doğrulama setindeki hedef değişkeni ayırır.
del df_train["churn"]  # Eğitim setinden hedef değişkeni kaldırır.
del df_val["churn"]  # Doğrulama setinden hedef değişkeni kaldırır.

# Kategorik ve sayısal sütunlar
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']  # Sayısal sütunlar.

# Eğitim verilerini sözlük formatına dönüştürme
train_dict = df_train[categorical + numerical].to_dict(orient="records")  # Eğitim verilerini sözlük formatına dönüştürür.
print(train_dict[:1])  # İlk satırı yazdırır.

# DictVectorizer ile kategorik veriyi sayısala çevirme
dv = DictVectorizer(sparse=False)  # DictVectorizer nesnesi oluşturur, sparse=False ile sonuç matrisini yoğun yapar.
dv.fit(train_dict)  # Eğitim verilerini vektörize etmek için fit eder.
X_train = dv.transform(train_dict)  # Eğitim verilerini sayısal değerlere dönüştürür.
print(X_train[0])  # İlk örneğin dönüştürülmüş değerlerini yazdırır.

# Özellik isimlerini yazdırma
print(dv.get_feature_names_out())  # Vektörleştirilmiş verinin özellik isimlerini yazdırır.

# Lojistik regresyon modeli
model = LogisticRegression(  # Lojistik regresyon modelini oluşturur.
    solver="liblinear",  # Küçük veri setleri için uygun olan solver.
    random_state=42  # Rastgelelik için sabit bir değer belirler.
)
model.fit(X_train, y_train)  # Modeli eğitir.

# Doğrulama seti ile aynı işlemleri tekrarlar
val_dict = df_val[categorical + numerical].to_dict(orient="records")  # Doğrulama setini sözlük formatına dönüştürür.
X_val = dv.transform(val_dict)  # Doğrulama setini dönüştürür.
y_pred = model.predict_proba(X_val)  # Modelin doğrulama seti üzerindeki olasılık tahminlerini yapar.
print(y_pred[:5])  # İlk 5 tahmini yazdırır.

# Modelin doğrulama ve eğitim setindeki performansını ölçer
print("The performance of the model on the validation dataset: ", model.score(X_val, y_val))  # Doğrulama seti üzerindeki doğruluk.
print("The performance of the model on the training dataset: ", model.score(X_train, y_train))  # Eğitim seti üzerindeki doğruluk.

# Modelin sapma ve katsayılarını yazdırır
print("Bias: ", model.intercept_[0])  # Modelin bias (sapma) değeri.
print(dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3))))  # Katsayıları özellik isimleriyle eşleştirip yazdırır.

# Yeni müşteri için tahmin yapma
customer = {  # Örnek müşteri verisi.
    'customerid': '8879-zkjof',
    'gender': 'male',
    'seniorcitizen': 1,
    'partner': 'no',
    'dependents': 'no',
    'tenure': 41,
    'phoneservice': 'yes',
    'multiplelines': 'no',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'no',
    'deviceprotection': 'yes',
    'techsupport': 'yes',
    'streamingtv': 'yes',
    'streamingmovies': 'yes',
    'contract': 'one_year',
    'paperlessbilling': 'yes',
    'paymentmethod': 'bank_transfer_(automatic)',
    'monthlycharges': 79.85,
    'totalcharges': 2990.75,
}

x_new = dv.transform([customer])  # Yeni müşteri verisini dönüştürür.
print(model.predict_proba(x_new))  # Modelin olasılık tahminini yazdırır.

# İkinci bir müşteri için tahmin yapma
customer2 = {  # İkinci örnek müşteri verisi.
    'gender': 'female',
    'seniorcitizen': 1,
    'partner': 'no',
    'dependents': 'no',
    'phoneservice': 'yes',
    'multiplelines': 'yes',
    'internetservice': 'fiber_optic',
    'onlinesecurity': 'no',
    'onlinebackup': 'no',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'yes',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 85.7,
    'totalcharges': 85.7
}
x_new2 = dv.transform([customer2])  # İkinci müşteri verisini dönüştürür.
print(model.predict_proba(x_new2))  # İkinci müşteri için olasılık tahmini.
