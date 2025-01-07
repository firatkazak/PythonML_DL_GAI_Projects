from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import numpy as np

# Kaliforniya konut fiyatlarını içeren veri setini yüklüyor
housing = fetch_california_housing()

# Özellikler (X) ve hedef değişkeni (y) ayırıyor
X = housing.data
y = housing.target

# Veriyi eğitim ve test setlerine ayırıyor
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, random_state=0)

# Ridge regresyon modeli oluşturuluyor ve eğitiliyor
ridge = Ridge().fit(X_egitim, y_egitim)

# Ridge modeli eğitim setindeki skorunu hesaplıyor (R-kare)
print(ridge.score(X_egitim, y_egitim))  # 0.610963329310342

# Ridge modeli test setindeki skorunu hesaplıyor (R-kare)
print(ridge.score(X_test, y_test))  # 0.5911615930747929

# Alpha=10 parametresi ile Ridge modeli oluşturulup eğitiliyor (daha fazla düzenleme)
ridge10 = Ridge(alpha=10).fit(X_egitim, y_egitim)

# Ridge10 modeli eğitim setindeki skorunu hesaplıyor (R-kare)
print(ridge10.score(X_egitim, y_egitim))  # 0.6109592860146297

# Lasso regresyon modeli oluşturuluyor ve eğitiliyor
lasso = Lasso().fit(X_egitim, y_egitim)

# Lasso modeli eğitim setindeki skorunu hesaplıyor (R-kare)
print(lasso.score(X_egitim, y_egitim))  # 0.2923162004273605

# Lasso modeli test setindeki skorunu hesaplıyor (R-kare)
print(lasso.score(X_test, y_test))  # 0.28490402733386166

# Lasso modelinde sıfır olmayan katsayıların sayısını yazdırıyor
print(np.sum(lasso.coef_ != 0))  # 3

# Lasso modeli alpha=0.01 ve daha fazla iterasyonla oluşturuluyor
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_egitim, y_egitim)

# Lasso001 modeli eğitim setindeki skorunu hesaplıyor (R-kare)
print(lasso001.score(X_egitim, y_egitim))  # 0.6072415448334423

# Lasso001 modeli test setindeki skorunu hesaplıyor (R-kare)
print(lasso001.score(X_test, y_test))  # 0.5855078217958598

# Lasso modeli alpha=0.0001 ve daha fazla iterasyonla oluşturuluyor
lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_egitim, y_egitim)

# Lasso00001 modeli eğitim setindeki skorunu hesaplıyor (R-kare)
print(lasso00001.score(X_egitim, y_egitim))  # 0.6109629994745134

# Lasso00001 modeli test setindeki skorunu hesaplıyor (R-kare)
print(lasso00001.score(X_test, y_test))  # 0.5911471829114463

# Elastic Net modeli oluşturuluyor ve eğitiliyor
elastic_net = ElasticNet().fit(X_egitim, y_egitim)

# Elastic Net modeli eğitim setindeki skorunu hesaplıyor (R-kare)
print(elastic_net.score(X_egitim, y_egitim))  # 0.43015600428508627

# Elastic Net modeli test setindeki skorunu hesaplıyor (R-kare)
print(elastic_net.score(X_test, y_test))  # 0.4151980682495934

# Elastic Net modelindeki sıfır olmayan katsayıların sayısını yazdırıyor
print(np.sum(elastic_net.coef_ != 0))  # Sıfır olmayan katsayıların sayısı: 3

# Elastic Net modeli alpha=0.01 ve l1_ratio=0.5 ile eğitiliyor (L1 ve L2 ceza oranı)
elastic_net01 = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=100000).fit(X_egitim, y_egitim)

# ElasticNet01 modeli eğitim setindeki skorunu hesaplıyor (R-kare)
print(elastic_net01.score(X_egitim, y_egitim))  # 0.609298050866643

# ElasticNet01 modeli test setindeki skorunu hesaplıyor (R-kare)
print(elastic_net01.score(X_test, y_test))  # 0.5881202190030845

# Ridge, Lasso ve ElasticNet modelleri, veriye düzenleme (regularization) uygulayan regresyon modelleridir.
# Ridge: L2 düzenlemesi yapar, yani büyük katsayıları küçültür.
# Lasso: L1 düzenlemesi yapar, bazı katsayıları tamamen sıfırlayarak özellik seçimi yapar.
# ElasticNet: L1 ve L2 düzenlemelerini birlikte kullanır.
# alpha parametresi: Modelin düzenleme miktarını kontrol eder. Yüksek alpha, daha fazla düzenleme anlamına gelir.
