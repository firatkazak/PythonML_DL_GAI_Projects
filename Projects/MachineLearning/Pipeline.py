from sklearn.datasets import _samples_generator, load_breast_cancer, load_wine  # Veri setleri ve veri oluşturucu fonksiyonlar.
from sklearn.model_selection import train_test_split, GridSearchCV  # Eğitim-test bölme ve GridSearchCV (model optimizasyonu).
from sklearn.feature_selection import SelectKBest, f_regression  # Özellik seçimi için SelectKBest ve f_regression (ANOVA F-testi).
from sklearn.svm import LinearSVC  # Lineer SVM sınıflandırıcı.
from sklearn.pipeline import make_pipeline  # Adımlardan oluşan bir pipeline oluşturmak için.
from sklearn.metrics import classification_report  # Sınıflandırma raporu (precision, recall, f1-score vb.).
from sklearn.svm import SVC  # Destek Vektör Makineleri (SVC).
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures  # Veri ön işleme için scaler ve polinom dönüşümü.
from sklearn.linear_model import Ridge  # Ridge (L2 regularizasyonlu) regresyon modeli.

# Pipeline Oluşturma
X, y = _samples_generator.make_classification(n_features=20,  # Toplam 20 özellikli veri seti oluştur.
                                              n_informative=3,  # Sadece 3 özellik bilgilendirici (informative).
                                              n_redundant=0,  # Hiçbir özellik gereksiz (redundant) değil.
                                              n_classes=4,  # 4 farklı sınıf oluştur.
                                              n_clusters_per_class=2  # Her sınıfın 2 kümesi olacak.
                                              )

# Veriyi eğitim ve test setine böler.
X_train, X_test, y_train, y_test = train_test_split(X,  # Özellik matrisi (X).
                                                    y,  # Hedef değişken (y).
                                                    random_state=42  # Rastgeleliği sabitler.
                                                    )

# Özellik seçimi: 3 en iyi özelliği seçmek için ANOVA F-testi kullanılır.
anova_filter = SelectKBest(f_regression, k=3)  # f_regression: ANOVA F-testi, k=3 en iyi özellik sayısı.

# Destek vektör makineleri (SVM) sınıflandırıcısı (Lineer kernel ile).
clf = LinearSVC(dual=True)
# dual Parametresi:
# dual=True: Dual formül kullanılır (sadece n_samples > n_features olduğunda kullanılır).
# dual=False: Primal formül kullanılır (genellikle daha fazla örnek varsa daha hızlıdır).

# Pipeline oluşturma: önce özellik seçimi yapılır, ardından SVM modeli uygulanır.
anova_svm = make_pipeline(anova_filter, clf)  # make_pipeline: Adım adım işlemleri zincirler (önce SelectKBest, sonra LinearSVC).

# Pipeline modeli eğitme.
anova_svm.fit(X_train, y_train)  # Pipeline içerisindeki tüm adımlar sırasıyla uygulanır (önce özellik seçimi, sonra sınıflandırma).

# Test seti üzerinde tahminler yap.
y_pred = anova_svm.predict(X_test)  # Test setindeki veriler için sınıf tahminleri yapar.
print(y_pred)  # Modelin tahmin ettiği sınıflar (0, 1, 2, 3).

# Modelin doğruluğunu hesaplar.
print(anova_svm.score(X_test, y_test))  # Doğruluk skoru: Test setindeki tahminlerin gerçek etiketlerle uyuşma oranı.

# Sınıflandırma raporu: Precision, Recall, F1-Score ve Destek (Support) gibi metrikleri gösterir.
print(classification_report(y_test, y_pred))

# Örnek: Meme kanseri veri seti ile pipeline
cancer = load_breast_cancer()  # Meme kanseri veri setini yükler.

# Eğitim ve test setlerine ayırır.
X_train, X_test, y_train, y_test = train_test_split(cancer.data,  # Özellik matrisi (X).
                                                    cancer.target,  # Hedef değişken (y).
                                                    random_state=0  # Rastgeleliği sabitler.
                                                    )

# MinMaxScaler ile verileri ölçekler, ardından SVC modeli uygular.
pp = make_pipeline(MinMaxScaler(),  # Verileri 0 ile 1 arasında ölçekler.
                   SVC(gamma='auto')  # Destek Vektör Makineleri (RBF kernel ile).
                   )

# Pipeline'ı eğitir.
pp.fit(X_train, y_train)  # Eğitim seti ile modeli eğitir (ölçeklendirme + SVC).

# Modelin doğruluğunu hesaplar.
print(pp.score(X_test, y_test))  # Test setinde doğruluk skoru: 0.951048951048951

# Örnek: Şarap veri seti ile pipeline
wine = load_wine()  # Şarap veri setini yükler.

# Eğitim ve test setlerine ayırır.
X_train, X_test, y_train, y_test = train_test_split(wine.data,  # Özellik matrisi (X).
                                                    wine.target,  # Hedef değişken (y).
                                                    random_state=0  # Rastgeleliği sabitler.
                                                    )

# StandardScaler ile veri ölçeklendirme, PolynomialFeatures ile polinomal dönüşüm, ardından Ridge regresyonu.
pp = make_pipeline(StandardScaler(),  # Veriyi standartlaştırır (ortalama=0, standart sapma=1).
                   PolynomialFeatures(),  # Özelliklere polinomal dönüşüm uygular (varsayılan olarak 2. derece).
                   Ridge()  # Ridge regresyonu (L2 regularizasyonu ile).
                   )

# Hyperparametre araması için GridSearchCV (polinom derecesi ve Ridge alpha parametreleri).
param_grid = {'polynomialfeatures__degree': [1, 2, 3],  # Polinomal dönüşümün derecesi.
              'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]  # Ridge regularizasyon parametresi (alpha).
              }

# GridSearchCV: En iyi parametre kombinasyonunu bulmak için grid araması.
grid = GridSearchCV(estimator=pp,  # Pipeline modeli.
                    param_grid=param_grid,  # Denenecek parametreler.
                    cv=5,  # 5 katlı çapraz doğrulama.
                    n_jobs=-1  # Paralel çalıştırma (işlemci çekirdeklerini maksimum kullanma).
                    )

# Grid araması ile en iyi modeli eğitir.
grid.fit(X_train, y_train)  # Eğitim seti üzerinde en iyi parametre kombinasyonunu bulur ve modeli eğitir.

# En iyi parametreleri gösterir.
print(grid.best_params_)  # En iyi parametre kombinasyonu: {'polynomialfeatures__degree': 1, 'ridge__alpha': 1}

# En iyi modelin test seti üzerindeki doğruluk skoru.
print(grid.score(X_test, y_test))  # Test setinde doğruluk skoru: 0.8109962898503273
