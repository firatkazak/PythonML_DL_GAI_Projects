import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV dosyasını okuyor ve bir DataFrame'e yüklüyor
df = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/googleplaystore.csv")

print(df.head())  # İlk 5 satırı yazdırır (veri setinin hızlı bir ön izlemesi)
print(df.columns)  # Tüm sütun isimlerini yazdırır
df.columns = df.columns.str.replace(" ", "_")  # Sütun isimlerindeki boşlukları alt çizgi ile değiştirir (kullanımı kolaylaştırmak için)
print(df.columns)  # Tüm sütun isimlerini yazdırır
print(df.shape)  # Veri setinin boyutunu (satır, sütun sayısını) yazdırır
print(df.dtypes)  # Sütunların veri tiplerini yazdırır
print(df.isnull().sum())  # Veri setindeki eksik (NaN) değer sayısını sütun başına yazdırır

sns.set_theme()  # Seaborn tema ayarlarını yapar ve görselleştirme kalitesini artırır
sns.set(rc={"figure.dpi": 90, "figure.figsize": (15, 10)})  # Grafik çözünürlüğünü (DPI) ve grafik boyutunu ayarlar
sns.heatmap(df.isnull(), cbar=False)  # Veri setindeki eksik değerleri ısı haritasında gösterir
plt.show()  # 1. Grafiği gösterir.

rating_median = df["Rating"].median()  # 'Rating' sütununun medyan değerini bulur
print(rating_median)  # 'Rating' sütununun medyan değerini ekrana yazdırır.
df["Rating"].fillna(rating_median, inplace=True)  # Eksik 'Rating' değerlerini medyan ile doldurur

df.dropna(inplace=True)  # Kalan tüm satırları (içinde NaN olan satırları) siler
df.isnull().sum().sum()  # Kalan eksik değerlerin toplamını yazdırır (hepsinin silindiğini kontrol etmek için)
print(df.info())  # Veri seti hakkında özet bilgi verir

df["Reviews"].describe()  # 'Reviews' sütunundaki istatistiksel özet bilgiyi verir
df["Reviews"] = df["Reviews"].astype("int64")  # 'Reviews' sütununu tam sayı ('int64') veri tipine çevirir
df["Reviews"].describe().round()  # 'Reviews' sütunundaki istatistiksel özet bilgiyi yuvarlayarak tekrar yazdırır
print(len(df["Size"].unique()))  # 'Size' sütunundaki benzersiz değer sayısını yazdırır

df["Size"].unique()  # 'Size' sütunundaki benzersiz değerleri yazdırır
df["Size"].replace("M", "", regex=True, inplace=True)  # 'Size' sütunundaki 'M' harflerini kaldırır (MB yerine sadece sayı olacak)
df["Size"].replace("k", "", regex=True, inplace=True)  # 'Size' sütunundaki 'k' harflerini kaldırır (kB yerine sadece sayı olacak)
df["Size"].unique()  # 'Size' sütunundaki benzersiz değerleri tekrar yazdırır
size_median = df[df["Size"] != "Varies with device"]["Size"].astype(float).median()  # 'Size' sütununda "Varies with device" olmayan değerlerin medyanını bulur
print(size_median)  # 'Size' sütununda "Varies with device" olmayan değerlerin medyanını yazdırır.

df["Size"].replace("Varies with device", size_median, inplace=True)  # 'Size' sütunundaki "Varies with device" değerlerini medyan ile değiştirir
df.Size = pd.to_numeric(df.Size)  # 'Size' sütununu sayısal ('float') veri tipine çevirir
print(df.Size.head())  # 'Size' sütununu sayısal ('float') veri tipi olarak yazdırır.

df.Size.describe().round()  # 'Size' sütunundaki istatistiksel özet bilgiyi yuvarlayarak yazdırır
df["Installs"].unique()  # 'Installs' sütunundaki benzersiz değerleri yazdırır
df.Installs = df.Installs.apply(lambda x: x.replace("+", ""))  # 'Installs' sütunundaki '+' işaretini kaldırır
df.Installs = df.Installs.apply(lambda x: x.replace(",", ""))  # 'Installs' sütunundaki virgülleri kaldırır
df.Installs = df.Installs.apply(lambda x: int(x))  # 'Installs' sütunundaki değerleri tam sayıya çevirir
df["Installs"].unique()  # 'Installs' sütunundaki benzersiz değerleri tekrar yazdırır
df["Price"].unique()  # 'Price' sütunundaki benzersiz değerleri yazdırır
df.Price = df.Price.apply(lambda x: x.replace("$", ""))  # 'Price' sütunundaki '$' işaretini kaldırır
df.Price = df.Price.apply(lambda x: float(x))  # 'Price' sütunundaki değerleri float veri tipine çevirir
df["Price"].unique()  # 'Price' sütunundaki benzersiz değerleri tekrar yazdırır
len(df["Genres"].unique())  # 'Genres' sütunundaki benzersiz tür sayısını yazdırır
df["Genres"].head(10)  # 'Genres' sütununun ilk 10 satırını yazdırır
df["Genres"] = df["Genres"].str.split(";").str[0]  # 'Genres' sütunundaki değerleri ";" ile ayırır ve ilk kısmı alır (ö; "Action;Adventure" yerine sadece "Action" kalır)
len(df["Genres"].unique())  # 'Genres' sütunundaki benzersiz tür sayısını tekrar yazdırır
df["Genres"].unique()  # 'Genres' sütunundaki benzersiz değerleri yazdırır
df["Genres"].value_counts()  # 'Genres' sütunundaki türlerin sayısını yazdırır
df["Genres"].replace("Music & Audio", "Music", inplace=True)  # 'Music & Audio' türünü 'Music' olarak değiştirir
df["Last_Updated"].head()  # 'Last Updated' sütununun ilk 5 satırını yazdırır
df["Last_Updated"] = pd.to_datetime(df["Last_Updated"])  # 'Last Updated' sütununu tarih ('datetime') formatına çevirir
print(df.head())  # Veri setinin ilk 5 satırını tekrar yazdırır
print(df.dtypes)  # Sütunların veri tiplerini tekrar yazdırır

df["Type"].value_counts().plot(kind="bar", color="red")  # 'Type' sütunundaki değerlerin sıklığını gösteren bir çubuk grafik çizer
plt.title("Free & Paid")  # Başlığa Free & Paid yazar.
plt.show()  # 2. Grafiği gösterir.

sns.boxplot(x="Type", y="Rating", data=df)  # 'Type' ve 'Rating' arasındaki ilişkiyi kutu grafiği ile gösterir
plt.title("Content rating with their counts")  # Başlığa Content rating with their counts yazar.
plt.show()  # 3. Grafiği gösterir.

sns.countplot(y="Content_Rating", data=df)  # 'Content Rating' sütunundaki değerlerin sıklığını gösteren bir sayım grafiği çizer
plt.title("Content rating with their counts")  # Başlığa Content rating with their counts yazar.
plt.show()  # 4. Grafiği gösterir.

sns.boxplot(x="Content_Rating", y="Rating", data=df)  # # 'Content Rating' ve 'Rating' arasındaki ilişkiyi kutu grafiği ile gösterir
plt.title("The content rating & rating", size=20)  # Başlığa The content rating & rating yazar.
plt.show()  # 5. Grafiği gösterir.

cat_num = df["Category"].value_counts()  # Kategorilerin sayısını gösteren bir çubuk grafik çizer
sns.barplot(x=cat_num.values, y=cat_num.index, orient='h')  # 'h' yatay çubuk grafiği oluşturur
plt.title("The number of categories", size=20)  # Başlığa The number of categories yazar.
plt.show()  # 6. Grafiği gösterir.

sns.scatterplot(data=df, y="Category", x="Price")  # Fiyat ile Kategori arasındaki ilişkiyi gösteren bir saçılım grafiği çizer
plt.title("Category & Price", size=20)  # Başlığa Category & Price yazar.
plt.show()  # 7. Grafiği gösterir.

numeric_df = df.select_dtypes(include=['float64', 'int64'])  # Sayısal sütunlar arasındaki korelasyonu gösteren bir ısı haritası çizer
sns.heatmap(numeric_df.corr(), annot=True, linewidths=.5, fmt=".2f")  # Korelasyon matrisini çizer
plt.title("Heatmap for numerical columns", size=20)  # Başlığa Heatmap for numerical columns yazar.
plt.show()  # 8. Grafiği gösterir.

sns.histplot(df["Rating"], kde=True)  # 'Rating' sütunu için KDE (yoğunluk fonksiyonu) ile birlikte bir histogram çizer
plt.title("Histogram with the kde for the rating column ", size=20)  # Başlığa Histogram with the kde for the rating column yazar.
plt.show()  # 9. Grafiği gösterir.
