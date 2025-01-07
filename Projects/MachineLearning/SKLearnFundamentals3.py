from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Model Kurma;
clf = RandomForestClassifier(random_state=0)
# RandomForestClassifier(): Bu fonksiyon, rastgele orman algoritmasını kullanarak bir sınıflandırıcı (classifier) oluşturur.
# Random Forest, birçok karar ağacının (decision tree) bir araya gelmesiyle oluşturulan bir topluluk öğrenme yöntemidir.
# Her ağaç, eğitim verisinde farklı bölümler üzerinde eğitilir ve daha sonra bu ağaçların tahminlerinin çoğunluğu alınarak sınıflandırma yapılır.
# random_state=0: Modelin sonuçlarının tekrar edilebilir olmasını sağlar. Aynı rastgelelik kullanıldığı sürece her çalıştırmada aynı sonuçları alırsın.

X = [[1, 2, 3], [11, 12, 13]]  # Bu, iki örnekten oluşan bir girdi veri setidir. Her örnek, üç özelliğe (feature) sahiptir
y = [0, 1]  # Bu etiketler, veri setindeki her örneğin sınıfını belirtir. İlk örnek sınıfı 0 ve ikinci örnek sınıfı 1'dir.
clf.fit(X, y)  # Modeli, girdi verisi X ve etiketler y ile eğitirsin.
# Bu adımda model, örneklere bakarak karar ağaçlarını oluşturur ve her ağacın nasıl sınıflandırma yapacağını öğrenir.
print(clf.predict(X))  # [0 1] Eğitim verisi olan X üzerinde modelin tahmin yapmasını sağlar. Model, daha önce gördüğü bu verilere göre sınıfları tahmin eder:
# İlk örnek için tahmin: 0
# İkinci örnek için tahmin: 1
# Bu tahminler, eğitim verisi etiketleri y = [0, 1] ile uyumludur, yani model doğru tahmin yapmıştır.

print(clf.predict([[4, 5, 6], [14, 15, 16]]))  # [0 1] Bu satırda, model daha önce görmediği iki yeni örnek üzerinde tahmin yapar:
# [4, 5, 6] için tahmin: 0
# [14, 15, 16] için tahmin: 1
# Bu sonuçlar, modelin veriye göre yeni girdilerde de nasıl sınıflandırma yapacağını gösterir. Model, 4-5-6 örneğini sınıf 0'a, 14-15-16 örneğini sınıf 1'e atamıştır.

# Veri Ön İşleme;
X = [[0, 15], [1, -10]]  # X: İki örnekten oluşan bir veri setidir:
# İlk örnek: [0, 15]
# İkinci örnek: [1, -10]
# Her bir örnek iki özellikten (sütundan) oluşur.

sonuc1 = StandardScaler().fit(X).transform(X)
# StandardScaler(): Bu, veri standardizasyonu için kullanılan bir sınıf. Standardizasyon, her bir özelliğin (sütunun) ortalamasını 0, standart sapmasını 1 yapar.
# fit(X): Veriye bakarak her bir özellik için ortalama (μ) ve standart sapmayı (σ) hesaplar.
# transform(X): Veriyi standartlaştırır, yani her bir özelliğin ortalamasını 0 ve standart sapmasını 1 olacak şekilde dönüştürür.
print(sonuc1)  # [[-1.  1.] [ 1. -1.]]
# İlk örnek: 0, 15 -> -1, 1
# İkinci örnek: 1, -10 -> 1, -1

# Pipeline oluşturma;
pipe = make_pipeline(StandardScaler(), LogisticRegression())  # Bu fonksiyon, ardışık işlem adımlarını bir araya getirir. Burada iki adım vardır;
# StandardScaler(): Veriyi standartlaştırarak her özelliğin (sütunun) ortalamasını 0 ve standart sapmasını 1 yapar.
# LogisticRegression(): Sınıflandırma problemlerinde kullanılan bir modeldir ve iki veya daha fazla sınıfa ayırma görevini gerçekleştirir.
X, y = load_iris(return_X_y=True)  # İris veri setini yükler.
# return_X_y=True: Bu parametre, veriyi ve etiketleri ayrı iki değişken olarak döndürür:
# X: Girdi verileri (özellikler).
# y: Hedef sınıflar (etiketler), yani hangi çiçek türüne ait olduğu bilgisi.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Veriyi eğitim ve test setlerine böler. Eğitim seti modeli eğitmek için, test seti ise modeli test etmek için kullanılır.
# X_train: Eğitim için kullanılacak giriş verileri.
# X_test: Test için kullanılacak giriş verileri.
# y_train: Eğitim için kullanılacak etiketler.
# y_test: Test için kullanılacak etiketler.
# random_state=0: Verinin bölünme şeklini sabit tutarak her çalıştırmada aynı sonuçları elde etmemizi sağlar.
pipe.fit(X_train, y_train)
# pipe.fit(X_train, y_train): Pipeline'ı kullanarak modelin eğitim verisi üzerinde öğrenme yapmasını sağlar. İşlem şu adımlarda gerçekleşir:
# İlk olarak, eğitim verisi StandardScaler ile standartlaştırılır.
# Standartlaştırılan veri üzerinde LogisticRegression modeli eğitilir.
a_score = accuracy_score(pipe.predict(X_test), y_test)
# Test verisi üzerinde tahmin yapılır. Bu işlem de pipeline üzerinden gerçekleştirilir, yani:
# İlk olarak, test verisi standartlaştırılır.
# Standartlaştırılmış veri üzerinde tahmin yapılır.
# accuracy_score(): Bu fonksiyon, modelin tahmin ettiği etiketler (sınıflar) ile gerçek etiketleri karşılaştırarak doğruluk oranını hesaplar.
# Doğruluk oranı, modelin doğru sınıflandırdığı örneklerin yüzdesidir.
# Örneğin, doğruluk oranı 0.97 ise, model test verisinin %97'sini doğru tahmin etmiştir.
print(a_score)  # 0.9736842105263158 %97 doğruluk ile tahmin etmiş.

# Model Değerlendirme;
X, y = make_regression(n_samples=1000, random_state=0)  # make_regression(): Rastgele bir regresyon veri seti oluşturur.
# Bu veri seti, sürekli bir hedef değişkenle (etiketle) ilişkilendirilmiş bir dizi girdi özelliğinden oluşur.
# n_samples=1000: 1000 örnek (veri noktası) oluşturur.
# random_state=0: Rastgele veri oluşturma sürecini sabitler, böylece her çalıştırmada aynı veri seti üretilir.
# X: Girdi verilerini (özellikler) temsil eder. Bu özellikler regresyon modeline girdi olarak verilir.
# y: Hedef değişkeni (çıktılar), yani tahmin etmeye çalıştığımız değerlerdir.
lr = LinearRegression()  # Lineer regresyon, hedef değişken (y) ile bağımsız değişkenler (X) arasındaki doğrusal ilişkiyi öğrenmeye çalışan bir regresyon modelidir.
result = cross_validate(lr, X, y)
# cross_validate(): Bu fonksiyon, modeli belirli sayıda alt küme üzerinde eğitip test ederek çapraz doğrulama yapar.
# Modelin her bir alt küme üzerindeki performansını ölçer.
# Cross-validation, overfitting'i (aşırı öğrenmeyi) önlemek ve modelin genel performansını daha iyi anlamak için kullanılır.
# lr: Lineer regresyon modelidir.
# X: Girdi verileridir.
# y: Hedef değişkenlerdir.
# NOT: Varsayılan olarak, cross-validation 5 katlı (5-fold) olarak gerçekleştirilir. Yani veri seti 5 alt kümeye bölünür ve model 5 kez eğitilir/test edilir:
# Her bir aşamada, veri setinin %80'i eğitim için, %20'si test için kullanılır.
# Bu işlem 5 kez farklı alt kümelerle tekrarlanır ve her bir testin doğruluk skoru kaydedilir.
print(result['test_score'])  # [1. 1. 1. 1. 1.]
# result['test_score']: Bu, her bir çapraz doğrulama adımında modelin test seti üzerindeki skorlarını döndürür.
# Lineer regresyon için, bu skorlar R² (determinasyon katsayısı) ile ölçülür.
# R² skoru, modelin veriye ne kadar iyi uyduğunu gösterir. 1.0 değeri, modelin mükemmel bir uyum sağladığını gösterir.
# Bu örnekte, tüm çapraz doğrulama adımlarında test skoru 1.0 olarak döndü. Bu, modelin tüm alt kümelerde mükemmel bir şekilde tahmin yaptığını gösterir.

# Otomatik Parametre Seçme;
X, y = fetch_california_housing(return_X_y=True)
# fetch_california_housing(): Kaliforniya'daki konut fiyatlarıyla ilgili bir veri setidir.
# Her satır, bir bölgedeki evlerin ortalama fiyatını içerir ve bu fiyat, çeşitli demografik ve coğrafi özelliklerle (X) ilişkilendirilir.
# X: Konut fiyatlarını tahmin etmek için kullanılan özelliklerdir (örneğin, nüfus, ortalama ev yaşı vb.).
# y: Tahmin edilmek istenen hedef değişken (Konut fiyatı).
# return_X_y=True: X ve y'nin ayrı iki değişken olarak döndürülmesini sağlar.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# train_test_split(): Veri setini eğitim ve test setlerine böler.
# X_train: Eğitim verisi (özellikler).
# X_test: Test verisi (özellikler).
# y_train: Eğitim verisi için hedef değişkenler (konut fiyatları).
# y_test: Test verisi için hedef değişkenler.
# random_state=0: Veri bölünmesinde sonuçların tekrarlanabilir olmasını sağlar.

param_districutions = {"n_estimators": randint(1, 5), "max_depth": randint(5, 10)}
# Bu adımda, RandomForestRegressor modelinin hiperparametreleri için olasılık dağılımları tanımlanır:
# n_estimators: Rastgele Orman algoritmasında kullanılacak ağaç sayısını belirtir. Burada, 1 ile 5 arasında rastgele bir değer seçilecektir.
# max_depth: Her bir ağacın maksimum derinliğini belirtir. Derinlik, ağacın kaç seviyeye kadar büyüyebileceğini belirler. Bu, 5 ile 10 arasında rastgele seçilecektir.

search = RandomizedSearchCV(  # RandomizedSearchCV(): hiperparametre aralığından rastgele seçimler yaparak en iyi hiperparametreleri bulur.
    estimator=RandomForestRegressor(random_state=0),  # Kullanılacak tahmin modeli olarak RandomForestRegressor seçilmiştir.
    n_iter=5,  # Hiperparametre araması 5 farklı parametre kombinasyonuyla yapılacak.
    param_distributions=param_districutions,  # Aranacak hiperparametrelerin aralıkları.
    random_state=0  # Rastgele arama sonuçlarının tekrarlanabilir olmasını sağlar.
)

search.fit(X_train, y_train)
# fit(): Hiperparametre araması yaparak modeli eğitim verisiyle eğitir.
# Her bir iterasyonda farklı bir hiperparametre kombinasyonu denenir ve modelin performansı değerlendirilir. Sonuç olarak, en iyi parametre seti seçilir.
print(search.best_params_)  # {'max_depth': 9, 'n_estimators': 4}
print(search.score(X_test, y_test))  # 0.735363411343253
# search.best_params_: RandomizedSearchCV tarafından seçilen en iyi hiperparametreleri gösterir.
# Örneğin, seçilen en iyi n_estimators ve max_depth değerleri bu çıktı ile alınır.

# search.score(X_test, y_test): En iyi hiperparametrelerle eğitilmiş modelin test seti üzerindeki performansını ölçer.
# Bu fonksiyon, R² skoru (determinasyon katsayısı) döndürür. R² skoru, modelin veriye ne kadar iyi uyduğunu gösterir.
# 1.0'a yakın değerler, modelin iyi bir performans gösterdiğini ifade eder.

# Sonuç: Bu derste sklearn kütüphanesi ile Model kurma, Veri ön işleme, Pipeline oluşturma, Model değerlendirme ve Otomatik parametre seçme konularını öğrendik.
