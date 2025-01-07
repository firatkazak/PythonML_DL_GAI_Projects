from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Veriyi oluşturma
X, y = make_moons(n_samples=100,
                  noise=0.25,
                  random_state=3
                  )

# Eğitim ve test setlerine ayırma
X_egitim, X_test, y_egitim, y_test = train_test_split(X,
                                                      y,
                                                      stratify=y,
                                                      random_state=42
                                                      )

# MLPClassifier modelini oluşturma ve eğitim
mlp = MLPClassifier(alpha=1,
                    hidden_layer_sizes=(100, 50),
                    max_iter=10000,
                    learning_rate_init=0.001,  # öğrenme oranı
                    momentum=0.9,  # momentum
                    random_state=0
                    ).fit(X_egitim, y_egitim)


# Karar sınırlarını çizmek için bir fonksiyon tanımlama
def plot_decision_boundary(X, y, model, ax):
    h = .02  # mesh grid adımı
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')


# Grafik hazırlama
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_boundary(X, y, mlp, ax)
ax.set_title('MLPClassifier Karar Sınırları')
plt.show()

# Örnek

kanser = load_breast_cancer()
X_egitimKanser, X_testKanser, y_egitimKanser, y_testKanser = train_test_split(kanser.data,
                                                                              kanser.target,
                                                                              random_state=0
                                                                              )

mlp.fit(X_egitimKanser, y_egitimKanser)
print("Ölçeklenmemiş Eğitim:", mlp.score(X_egitimKanser, y_egitimKanser))  # 0.92018779342723
print("Ölçeklenmemiş Test", mlp.score(X_testKanser, y_testKanser))  # 0.9090909090909091

#
scaler = StandardScaler()
scaler.fit(X_egitimKanser)
X_egitimKanser_Scaled = scaler.transform(X_egitimKanser)
X_testKanser_Scaled = scaler.transform(X_testKanser)

mlp.fit(X_egitimKanser_Scaled, y_egitimKanser)
print("Ölçeklenmiş Eğitim", mlp.score(X_egitimKanser_Scaled, y_egitimKanser))  # 0.9906103286384976
print("Ölçeklenmiş Test", mlp.score(X_testKanser_Scaled, y_testKanser))  # 0.951048951048951
# Görüldüğü üzere değerler 0.92 ve 0.90'dan 0.99 ve 0.95'e yükseldi. İstediğimiz de buydu.
