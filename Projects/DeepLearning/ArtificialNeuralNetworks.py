from sklearn.datasets import load_iris  # sklearn'den Iris veri setini yüklüyor.
from sklearn.linear_model import Perceptron  # sklearn'den Perceptron sınıfını yüklüyor, bu sınıf basit bir yapay sinir ağı modelidir.
from sklearn.datasets import fetch_california_housing  # sklearn'den California konut veri setini yüklüyor.
from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test setlerine ayırmak için kullanılan fonksiyon.
from sklearn.neural_network import MLPRegressor  # Çok katmanlı yapay sinir ağı modelini sağlayan sınıf.
from sklearn.pipeline import make_pipeline  # İşleme adımlarını sırayla yürütmek için bir pipeline (boru hattı) oluşturmaya yarar.
from sklearn.preprocessing import StandardScaler  # Verileri standartlaştırmak için kullanılan sınıf.
from sklearn.metrics import mean_squared_error  # Ortalama karesel hata (RMSE) metriğini hesaplar.

# 1. Perceptron
iris = load_iris(as_frame=True)  # Iris çiçeği veri setini yüklüyor, as_frame=True ile veri bir pandas DataFrame formatında döndürülüyor.
X = iris.data[["petal length (cm)", "petal width (cm)"]].values  # Iris veri setinden sadece taç yaprağı uzunluğu ve genişliği sütunlarını seçiyor.
y = (iris.target == 0)  # Iris setosa sınıfını seçiyor, 0'a eşit olanları True (setosa), diğerlerini False yapıyor.
print(y[:5])  # İlk 5 hedef değeri yazdırıyor, True veya False olarak.

per_clf = Perceptron(random_state=42)  # Perceptron modeli oluşturuluyor, random_state=42 ile aynı sonuçları elde etmek için rastgelelik sabitleniyor.
sonuc1 = per_clf.fit(X, y)  # Model, X özelliklerine ve y etiketlerine göre eğitiliyor.
print(sonuc1)  # Eğitim sonucunu yazdırıyor, bu sonuç modelin kendisidir.

X_new = [[2, 0.5], [3, 1]]  # Yeni tahmin için 2 tane örnek oluşturuluyor: [[2, 0.5], [3, 1]]
y_pred = per_clf.predict(X_new)  # Yeni örneklerin tahminleri yapılıyor. Tahmin edilen sınıf True (setosa) veya False (setosa değil).
print(y_pred)  # Tahmin sonuçları yazdırılıyor, [True, False] gibi olabilir.

# 2. Multilayer Perceptron (MLP: Çok Katmanlı Algılayıcı)
# 2.1. Loading dataset
housing = fetch_california_housing()  # California konut veri setini yüklüyor, konut fiyatlarını tahmin etmek için kullanılır.
# Spliting into dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)  # Veri setini eğitim ve test setlerine ayırıyor, random_state=42 ile rastgelelik sabitleniyor.
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)  # Eğitim setini bir de validasyon (doğrulama) ve eğitim seti olarak ayırıyor.

# Modeling
mlp_reg = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)  # Çok katmanlı sinir ağı modeli oluşturuluyor. 3 gizli katman var ve her katmanda 50 nöron var.
pipeline = make_pipeline(StandardScaler(), mlp_reg)  # Pipeline oluşturuluyor, önce veriler StandardScaler ile ölçekleniyor, sonra model uygulanıyor.
sonuc2 = pipeline.fit(X_train, y_train)  # Model eğitiliyor. Veriler önce ölçekleniyor, ardından sinir ağı modeli ile eğitiliyor.
print(sonuc2)  # Eğitim sonucu yazdırılıyor, pipeline nesnesini döndürüyor.

# Prediction
y_pred = pipeline.predict(X_valid)  # Validasyon setinde model tahmin yapıyor.
rmse = mean_squared_error(y_valid, y_pred)  # Tahminlerle gerçek değerler arasındaki hata hesaplanıyor (MSE).
print(rmse)  # Hata değeri yazdırılıyor. Bu, modelin tahmin performansını gösterir.

# Perceptron: Yapay sinir ağının en temel hali. İkili sınıflandırma problemleri için kullanılır. Girdi verilerine göre (X) hedef (y) tahmin edilir.
# MLPRegressor: Birden fazla katmanı olan sinir ağı modelidir. Regresyon problemlerinde (sürekli değerler tahmin etme) kullanılır.
# Gizli katmanlar ve nöron sayıları modelin karmaşıklığını belirler.
# StandardScaler: Verileri standartlaştırır, yani ortalaması 0, standart sapması 1 olacak şekilde ölçekler.
# Pipeline: Adımları sırayla uygulayan bir yapı. Burada önce veri standartlaştırılıyor, sonra model eğitiliyor veya tahmin yapılıyor.

