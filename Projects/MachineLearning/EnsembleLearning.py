from sklearn.datasets import make_moons, make_blobs, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Ay veri seti (make_moons) oluşturuluyor, 100 örnek ve %25 gürültü ekleniyor.
X, y = make_moons(n_samples=100,
                  noise=0.25,  # Veriye rastgele gürültü eklemek için.
                  random_state=3  # Sonuçları tekrar edilebilir yapmak için sabit bir rastgelelik.
                  )

# Eğitim ve test verisi %75 eğitim, %25 test olacak şekilde ayrılıyor.
X_train, X_test, y_train, y_test = train_test_split(X,  # Özellikler
                                                    y,  # Hedef değişken
                                                    stratify=y  # Hedef değişkenin dengesini korumak için.
                                                    )

# Lojistik regresyon modeli oluşturuluyor ve eğitiliyor.
log = LogisticRegression(solver='lbfgs').fit(X_train, y_train)  # lbfgs algoritması ile çözülüyor.

# 10 karar ağacından oluşan bir Random Forest modeli oluşturuluyor ve eğitiliyor.
rnd = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)  # 10 ağaç ile Random Forest modeli.

# Support Vector Classifier (SVC) modeli oluşturuluyor ve eğitiliyor.
svm = SVC(gamma="auto").fit(X_train, y_train)  # gamma="auto" ile SVM kernel parametresi ayarlanıyor.

# Birleştirilmiş oy çokluğu sınıflandırıcı (voting classifier) oluşturuluyor.
voting = VotingClassifier(estimators=[('lr', log), ('rf', rnd), ('svc', svm)],  # Farklı modelleri birleştiriyoruz.
                          voting='hard'  # Her modelin oyu ile sonuç belirleniyor (hard voting).
                          ).fit(X_train, y_train)

# Farklı modellerin test setindeki doğruluğunu yazdırıyoruz.
print(log.score(X_test, y_test))  # Lojistik regresyon doğruluğu: 0.84
print(rnd.score(X_test, y_test))  # Random Forest doğruluğu: 0.96
print(svm.score(X_test, y_test))  # SVM doğruluğu: 0.92
print(voting.score(X_test, y_test))  # Voting Classifier doğruluğu: 0.92

# Bagging & Pasting
# make_blobs ile yapay veri kümesi oluşturuluyor, 4 merkezli ve 300 örnek.
X, y = make_blobs(n_samples=300,
                  centers=4,  # 4 farklı küme merkezi.
                  random_state=0,  # Sonuçları tekrar edilebilir yapmak için.
                  cluster_std=1  # Küme genişlikleri standart sapması.
                  )

# Eğitim ve test setine ayırıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y  # Hedef değişkenin dengesini korumak için.
                                                    )

# Karar ağacı modeli oluşturuluyor ve eğitiliyor.
tree = DecisionTreeClassifier().fit(X_train, y_train)

# Bagging yöntemiyle 100 karar ağacını birleştiriyoruz.
bag = BaggingClassifier(tree,  # Temel model olarak karar ağacı kullanılıyor.
                        n_estimators=100,  # 100 farklı karar ağacı oluşturulacak.
                        max_samples=0.8,  # Eğitim verisinin %80'i kullanılacak.
                        n_jobs=-1,  # Paralel işlem için tüm CPU çekirdekleri kullanılıyor.
                        random_state=1  # Rastgelelik sabitleniyor.
                        ).fit(X_train, y_train)

# Karar ağacı ve bagging modelinin test doğruluklarını yazdırıyoruz.
print(tree.score(X_test, y_test))  # Karar ağacının doğruluğu: 0.87
print(bag.score(X_test, y_test))  # Bagging modelinin doğruluğu: 0.89

# Random Forests;
# Göğüs kanseri veri seti yükleniyor.
kanser = load_breast_cancer()  # Göğüs kanseri veri setini yüklüyoruz.
X_train, X_test, y_train, y_test = train_test_split(kanser.data,  # Eğitim ve test verisi.
                                                    kanser.target,  # Hedef değişken.
                                                    random_state=0  # Sonuçları tekrar edilebilir yapmak için.
                                                    )

# 100 ağaçtan oluşan bir Random Forest modeli oluşturuluyor ve eğitiliyor.
forest = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)  # 100 ağaçlı Random Forest.
print(forest.score(X_test, y_test))  # Test setindeki doğruluk: 0.972

estimator = forest.estimators_[0]

# Ağaç görselleştirme
plt.figure(figsize=(20, 10))
plot_tree(estimator, feature_names=kanser.feature_names, filled=True)
plt.show()

# Gradient Boosted
# Gradient Boosting Classifier modeli oluşturuluyor ve eğitiliyor.
gbrt = GradientBoostingClassifier(learning_rate=0.1,  # Öğrenme oranı.
                                  random_state=0  # Rastgelelik sabitleniyor.
                                  ).fit(X_train, y_train)

# Gradient Boosting modelinin eğitim ve test doğruluğunu yazdırıyoruz.
print(gbrt.score(X_train, y_train))  # Eğitim seti doğruluğu: 1.0 (overfitting).
print(gbrt.score(X_test, y_test))  # Test seti doğruluğu: 0.965
