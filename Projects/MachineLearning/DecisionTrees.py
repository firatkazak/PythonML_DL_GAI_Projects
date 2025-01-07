from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
import graphviz

# NOT: Sadece pip install graphviz ile çalışmaz!
# https://graphviz.org/download/ buradan exe dosyasını indirip kurmamız gerekiyor.

# 1. Örnek: Burada Modelin Train ve Test çıktılarını iyileştirmeyi öğreniyoruz.
# 1. örnekte modelin eğitim verisinin doğruluğu 1'den 0.98'e düşerken test verisinin doğruluğu 0.94'ten 0.95'e çıktı. İstenen de bu.

kanser = load_breast_cancer()  # load_breast_cancer fonksiyonu ile göğüs kanseri veri setini yüklüyoruz.
X_train, X_test, y_train, y_test = train_test_split(kanser.data,  # Kanser veri setini eğitim ve test olarak ayırıyoruz.
                                                    kanser.target,  # Hedef değişkenleri eğitim ve teste ayırıyoruz.
                                                    stratify=kanser.target  # Veri dengesini korumak için stratify kullanıyoruz.
                                                    )

tree = DecisionTreeClassifier()  # Karar ağacı sınıflandırıcısı modeli oluşturuluyor.
tree.fit(X_train, y_train)  # Modeli eğitim verileri ile eğitiyoruz.

print(tree.score(X_train, y_train))  # Eğitim setindeki doğruluğu yazdırıyoruz. %100 doğruluk.
print(tree.score(X_test, y_test))  # Test setindeki doğruluğu yazdırıyoruz. %94 doğruluk.

# Maksimum derinlik sınırlaması olan yeni bir model oluşturuyoruz.
tree = DecisionTreeClassifier(max_depth=4)  # Karar ağacının derinliği 4 ile sınırlandırılıyor, overfitting'i önlemek için.
tree.fit(X_train, y_train)  # Yeni modeli eğitim verisi ile tekrar eğitiyoruz.

print(tree.score(X_train, y_train))  # Eğitim setindeki yeni doğruluğu yazdırıyoruz. %98 doğruluk.
print(tree.score(X_test, y_test))  # Test setindeki yeni doğruluğu yazdırıyoruz. %95 doğruluk.

# 2. Örnek:
iris = load_iris()  # Iris veri setini yüklüyoruz.
X = iris.data[:, 2:]  # Eğitim için sadece sepal uzunluğu ve genişliğini kullanıyoruz (2. sütundan itibaren).
y = iris.target  # Hedef değişken, yani iris çiçeğinin türü.
tree = DecisionTreeClassifier()  # Karar ağacı sınıflandırıcısı oluşturuluyor.
tree.fit(X, y)  # Model, iris veri seti ile eğitiliyor.

dot_data = export_graphviz(decision_tree=tree,  # Karar ağacını görselleştirmek için graphviz formatında çıktı alıyoruz.
                           out_file=None,  # Çıktıyı dosya yerine bellekte tutmak için None ayarlandı.
                           feature_names=iris.feature_names[2:],  # Kullanılan özelliklerin isimlerini belirtiyoruz.
                           class_names=iris.target_names,  # Sınıfların isimlerini belirtiyoruz.
                           filled=True,  # Düğümleri sınıflara göre renkli hale getiriyoruz.
                           rounded=True,  # Düğümler yuvarlatılmış şekilde çiziliyor.
                           special_characters=True  # Özellik isimlerinde özel karakterleri desteklemek için.
                           )

graph = graphviz.Source(dot_data)  # Karar ağacının graphviz formatındaki verisini kaynak olarak alıyoruz.
graph.render("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/decision_tree")  # Karar ağacını "decision_tree" adlı dosyaya kaydediyoruz.
graph.view()  # Karar ağacını görüntülüyoruz.

# 3. Örnek:
tree_reg = DecisionTreeRegressor(max_depth=2)  # Karar ağacı regresyon modeli oluşturuyoruz, maksimum derinlik 2.
tree_reg.fit(X, y)  # Regresyon modeli iris verisiyle eğitiliyor.

dot_data = export_graphviz(tree_reg,  # Regresyon ağacını görselleştirmek için export_graphviz kullanılıyor.
                           out_file=None,  # Çıktıyı dosyaya değil, bellekte tutuyoruz.
                           feature_names=iris.feature_names[2:],  # Kullanılan özelliklerin isimleri.
                           filled=True,  # Düğümleri dolu ve renkli hale getiriyoruz.
                           rounded=True,  # Düğümleri yuvarlatıyoruz.
                           special_characters=True)  # Özellik isimlerinde özel karakterleri destekliyoruz.

graph = graphviz.Source(dot_data)  # Regresyon ağacının graphviz verisini kaynak olarak alıyoruz.
graph.render("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/decision_tree_regressor")  # Regresyon ağacını "decision_tree_regressor" adlı dosyaya kaydediyoruz.
graph.view()  # Regresyon ağacını görüntülüyoruz.
