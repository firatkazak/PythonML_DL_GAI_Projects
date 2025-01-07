import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
iris = sns.load_dataset('iris')

X_iris = iris.drop(labels='species', axis=1)  # species hariç tüm sütunları alarak özellik matrisini oluşturur.
# axis=1 parametresi, sütunları (yani veri setinin genişliğini) ifade eder.
# iris.drop(labels='species', axis=1) kodu, species sütununu veri setinden çıkartarak sadece özellik sütunlarını içeren bir veri çerçevesi oluşturur.
# Eğer axis=0 verilseydi, bu durumda satırları (yani veri setinin yüksekliğini) ifade ederdi ve belirli bir satırı çıkartmaya çalışırdı.
# Ancak burada amacımız sütunları çıkarmak olduğu için axis=1 kullanılmıştır.
y_iris = iris['species']  # species sütununu hedef değişken (yani hangi tür olduğu) olarak ayırıyor.
# Bu sütun, sınıflandırma için kullanılacak olan iris çiçek türlerini (setosa, versicolor, virginica) içerir.

# Rastgele veri oluştur
rng = np.random.RandomState(42)  # Sabit bir rastgelelik sağlamak için rastgele sayı üreteci oluşturuluyor. 42 sabit bir seed değeri olarak kullanılır.
x = 10 * rng.rand(50)  # 0 ile 10 arasında rastgele 50 sayı üretiyor ve x adlı değişkene atıyor.
y = 2 * x - 1 + rng.randn(50)  # x değerlerinin 2 katı eksi 1 formülü ile y değerlerini hesaplıyor ve üzerine rastgele gürültü ekliyor.

# Veriyi çiz
plt.scatter(x, y)  # x ve y değerlerini kullanarak bir saçılım grafiği oluşturuyor. Bu, verilerin dağılımını görselleştirir.

# Lineer model oluştur ve eğit
model = LinearRegression(fit_intercept=True)  # Doğrusal regresyon modelini oluşturuyor.
# fit_intercept=True parametresi, modelin kesişim (intercept) terimini hesaplamasını sağlar.
X = x[:, np.newaxis]  # x vektörünü, tek sütunlu bir matris (X) haline getiriyor. Bu, LinearRegression modelinin beklentisi olan şekli sağlar.
model.fit(X, y)  # Modeli X ve y verileri ile eğitiyor. Bu, doğrusal ilişkileri öğrenir ve bir regresyon doğrusunu uyumlu hale getirir.

# Model parametrelerini yazdır
print("Katsayı:", model.coef_)  # Katsayı: [1.9776566]
print("Kesişen:", model.intercept_)  # Kesişen: -0.903310725531111
print("Iris veri seti şekli:", X_iris.shape)  # Iris veri seti şekli: (150, 4)

# Tahmin yap ve çiz
x_fit = np.linspace(start=-1, stop=11)  # -1 ile 11 arasında eşit aralıklı 50 değer içeren bir dizi oluşturur.
# Bu dizi, modelin tahminlerini göstermek için kullanılacak x değerlerini içerir.
X_fit = x_fit[:, np.newaxis]  # x_fit dizisini, tek sütunlu bir matris (X_fit) haline getirir. Bu, modelin tahmin yapmak için beklediği şekli sağlar.
y_fit = model.predict(X_fit)  # Eğitim sırasında öğrenilen doğrusal regresyon modelini kullanarak X_fit değerleri için tahminler (y_fit) üretir.
plt.plot(x_fit, y_fit, color='red')  # x_fit ve y_fit değerlerini kullanarak bir doğrusal çizgi çizer. Çizginin rengi kırmızı olarak ayarlanmıştır.

# Grafikleri göster
plt.xlabel('x')  # X ekseninin etiketini 'x' olarak ayarlar. Bu, grafikte X ekseninin neyi temsil ettiğini belirtir.
plt.ylabel('y')  # Y ekseninin etiketini 'y' olarak ayarlar. Bu, grafikte Y ekseninin neyi temsil ettiğini belirtir.
plt.title('Lineer Regresyon')  # Grafiğin başlığını 'Lineer Regresyon' olarak ayarlar. Bu başlık, grafiğin içeriğini açıklamak için kullanılır.
plt.show()  # Grafiği ekranda gösterir.

# Modelin Eğitimi ve Performans değerlendirme
X_egitim, X_test, y_egitim, y_test = train_test_split(X_iris, y_iris, random_state=1)
# X_egitim ve y_egitim: Eğitim seti için özellikler ve etiketler.
# X_test ve y_test: Test seti için özellikler ve etiketler.
# X_iris (özellikler) ve y_iris (etiketler) veri setini rastgele olarak eğitim ve test setlerine böler.
# random_state=1 sabit bir tohum değeri sağlar, böylece sonuçlar tekrarlanabilir.

model = GaussianNB()  # Gaussian Naive Bayes sınıflandırıcısını oluşturur. Bu, verilerin normal dağıldığı varsayımıyla çalışan bir modeldir.
model.fit(X_egitim, y_egitim)  # Eğitim verileri (X_egitim ve y_egitim) ile modelin eğitimini yapar.
y_model = model.predict(X_test)  # Test setindeki (X_test) özelliklere dayanarak tahminlerde bulunur ve bu tahminleri y_model değişkenine atar.
sonuc = accuracy_score(y_test, y_model)  # Test setindeki gerçek etiketler (y_test) ile modelin tahminleri (y_model) arasındaki doğruluk oranını hesaplar.
print("Doğruluk:", sonuc)  # Accuracy: 0.9736842105263158
# Projenin Ana Fikri: Regresyon ve Sınıflandırma ile yeni bir iris çiçeğinin etiketini %97 ihtimal ile doğru tahmin edeceğiz.
