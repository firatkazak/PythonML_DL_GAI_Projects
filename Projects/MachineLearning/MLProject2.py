import pandas as pd  # Pandas kütüphanesi, veri analizi ve veri manipülasyonu için kullanılır
import matplotlib.pyplot as plt  # Matplotlib kütüphanesi, grafik çizimi ve veri görselleştirme için kullanılır
import seaborn as sns  # Seaborn, Matplotlib üzerine kurulu, istatistiksel veri görselleştirmesi sağlar
import warnings  # Uyarıları yönetmek ve baskılamak için kullanılır

# CSV dosyasını okur ve bir DataFrame'e dönüştürür
df = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/forbes_2022_billionaires.csv")

# İlk 5 satırı ekrana yazdırır (veri setinin ilk bakışta nasıl göründüğünü görmek için)
print(df.head())

# Veri setinin boyutunu (satır ve sütun sayısını) yazdırır
print(df.shape)

# Sütunlardaki veri tiplerini ekrana yazdırır
print(df.dtypes)

# İlgili sütunları seçer ve veri setine yeniden atar (veri setini temizlemek için)
df = df.loc[:, ["rank", "personName", "age", "finalWorth", "category", "country", "gender"]]

# İlk 5 satırı tekrar yazdırır (yeni veri setini görmek için)
print(df.head())

# 'rank' sütununu indeks olarak ayarlar
df = df.set_index("rank")
print(df.head())  # İndekslenmiş ilk 5 satırı yazdırır

# Eksik (NaN) değerlerin sayısını sütun başına yazdırır
print(df.isnull().sum())

# Eksik verileri siler (inplace=True ile değişiklik veri setine uygulanır)
print(df.dropna(inplace=True))

# Veri seti hakkında özet bilgi verir (sütun türleri, bellek kullanımı vb.)
print(df.info())

# 'gender' sütunundaki değerlerin frekansını (kaç kere tekrarlandığını) verir
df["gender"].value_counts()

# 'gender' sütunundaki değerlerin yüzde olarak dağılımını gösterir
df["gender"].value_counts(normalize=True)

# 'gender' sütununa göre gruplama yapar
df_gender = df.groupby(["gender"])

# Gruplar için ortalama yaşları hesaplar
df_gender["age"].mean()

# Seaborn tema ayarlarını yapar ve genel grafik görünümünü düzenler
sns.set_theme()

# Grafik çözünürlüğünü ve boyutunu ayarlar
sns.set(rc={"figure.dpi": 90, "figure.figsize": (15, 10)})

# Uyarıları göz ardı eder (örneğin, olası veri tipi uyarılarını baskılar)
warnings.filterwarnings("ignore")

# Cinsiyete göre kişi sayısını gösteren bir çubuk grafik çizer
df_gender.size().plot(kind="bar")
plt.title('Average ages of men and women', fontsize=20)  # Başlık ekler
plt.show()  # 1. grafiği gösterir

# İlk 10 kişinin servetini çubuk grafik ile gösterir
sns.barplot(x=df["personName"][:10], y=df["finalWorth"][:10])
plt.title('Top 10 richest', fontsize=20)  # Başlık ekler
plt.show()  # 2. grafiği gösterir

# 'country' sütunundaki benzersiz ülke sayısını yazdırır
print(len(df["country"].unique()))

# Ülkelere göre gruplama yapar
df_country = df.groupby("country")

# Her ülkenin milyarder sayısını hesaplar, azalan sıraya göre sıralar ve DataFrame'e çevirir
df_country_count = pd.DataFrame(df_country.size().sort_values(ascending=False), columns=["Count"])

# İlk 5 ülkeyi yazdırır
df_country_count.head()

# İlk 10 ülkenin milyarder sayısını çubuk grafik ile gösterir
sns.barplot(x=df_country_count["Count"][:10].values, y=df_country_count.index[:10])
plt.title('Top 10 countries', fontsize=20)  # Başlık ekler
plt.show()  # 3. grafiği gösterir

# 'category' sütunundaki benzersiz kategorileri gösterir
df["category"].unique()

# 'category' sütunundaki boşlukları kaldırır ve '&' işaretini '_' ile değiştirir
df["category"] = df["category"].apply(lambda x: x.replace(" ", "")).apply(lambda x: x.replace("&", "_"))

# Düzenlenmiş benzersiz kategorileri tekrar gösterir
df["category"].unique()

# Kategorilere göre gruplanmış veri sayısını hesaplar
df_category = df.groupby("category").size()
print(df_category.head())  # İlk 5 kategoriyi yazdırır

# Kategorileri DataFrame formatına çevirir
df_category = df_category.to_frame()
print(df_category.head())  # İlk 5 kategoriyi yazdırır

# Yeni sütun adı "Count" olacak şekilde yeniden adlandırır ve en yüksek sayıya göre sıralar
df_category = df_category.rename(columns={0: "Count"}).sort_values(by="Count", ascending=False)
print(df_category.head())  # En popüler 5 kategoriyi yazdırır

# İlk 10 kategorinin dağılımını çubuk grafik ile gösterir
sns.barplot(x=df_category["Count"][:10].values, y=df_category.index[:10])
plt.title('Top 10 categories', fontsize=20)  # Başlık ekler
plt.show()  # 4. grafiği gösterir

# Yaş ve servet arasındaki ilişkiyi saçılım grafiği ile gösterir
sns.scatterplot(x=df["age"], y=df["finalWorth"])
plt.title('The relationship between money and age', fontsize=20)  # Başlık ekler
plt.show()  # 5. grafiği gösterir

# Yaş dağılımını histogram ile gösterir
sns.histplot(df["age"])
plt.title('The distribution of age', fontsize=20)  # Başlık ekler
plt.show()  # 6. grafiği gösterir
