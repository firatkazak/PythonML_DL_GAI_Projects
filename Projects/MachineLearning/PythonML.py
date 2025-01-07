import pandas as pd  # Veri işleme ve analiz için kullanılır.
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test setlerine ayırmak için kullanılır.
from sklearn.pipeline import Pipeline  # İş akışlarını tanımlamak için kullanılır.
from sklearn.impute import SimpleImputer  # Eksik verileri doldurmak için kullanılır.
from sklearn.preprocessing import StandardScaler  # Veriyi standartlaştırmak (ölçeklemek) için kullanılır.
from sklearn.preprocessing import OneHotEncoder  # Kategorik verileri sayısal değerlere çevirmek için kullanılır.
from sklearn.compose import ColumnTransformer  # Farklı veri türlerine göre ayrı ayrı ön işleme yapılmasını sağlar.
from sklearn.ensemble import RandomForestRegressor  # Rastgele orman regresyon modeli.
from sklearn.metrics import mean_absolute_error  # Hata metriği olarak MAE kullanılır.
from sklearn.model_selection import cross_val_score  # Çapraz doğrulama yapmak için kullanılır.
from sklearn.model_selection import GridSearchCV  # Parametre araması ve en iyi modeli bulmak için kullanılır.

df_train = pd.read_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/house_train.csv")  # Eğitim verisini okur.
df_test = pd.read_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/house_test.csv")  # Test verisini okur.

print(df_train.head())  # Eğitim verisinin ilk 5 satırını görüntüler.
print("The shape of train set: ", df_train.shape)  # Eğitim setinin boyutunu görüntüler.
print("The shape of test set: ", df_test.shape)  # Test setinin boyutunu görüntüler.
print(df_train.dtypes)  # Veri tiplerini görüntüler.
print(df_train.info())  # Verinin genel bilgisini verir.
print(df_train.describe().T)  # Verinin istatistiksel özetini verir.

df_train.set_index(
    keys="Id",  # "Id" sütununu indeks olarak ayarlar.
    inplace=True  # Mevcut DataFrame'de işlemi uygular.
)

df_test.set_index(
    keys="Id",  # "Id" sütununu indeks olarak ayarlar.
    inplace=True  # Mevcut DataFrame'de işlemi uygular.
)

df_train.head()  # İlk birkaç satırı görüntüler.
df_train.isnull().sum()  # Her sütundaki eksik veri sayısını görüntüler.
cols_with_null = df_train.isnull().sum().sort_values(ascending=False)  # Eksik veri içeren sütunları azalan sırayla sıralar.
cols_with_null.head(20)  # En çok eksik veri içeren ilk 20 sütunu görüntüler.
print("Total number of missing data in the dataset: ", df_train.isnull().sum().sum())  # Toplam eksik veri sayısını görüntüler.
df_train["SalePrice"].isnull().sum()  # "SalePrice" sütunundaki eksik veri sayısını görüntüler.

cols_to_drop = cols_with_null.head(6).index.tolist()  # Eksik veri içeren ilk 6 sütunu listeler.
df_train.drop(
    labels=cols_to_drop,  # Belirtilen sütunları düşürür.
    axis=1,  # Sütun ekseninde işlem yapar.
    inplace=True  # Mevcut DataFrame üzerinde işlem yapar.
)
df_test.drop(
    labels=cols_to_drop,  # Belirtilen sütunları test setinden de düşürür.
    axis=1,  # Sütun ekseninde işlem yapar.
    inplace=True  # Mevcut DataFrame üzerinde işlem yapar.
)

y = df_train.SalePrice  # Hedef değişken olan "SalePrice" sütununu alır.
X = df_train.drop(labels=["SalePrice"], axis=1)  # Geriye kalan bağımsız değişkenleri alır.

X_train, X_val, y_train, y_val = train_test_split(
    X,  # Bağımsız değişkenler (özellikler).
    y,  # Bağımlı değişken (hedef).
    train_size=0.8,  # Verinin %80'i eğitim, %20'si doğrulama için ayrılır.
    random_state=0  # Rastgele bölmeyi sabitler, böylece her çalışmada aynı bölme elde edilir.
)

categorical_cols = [cname for cname in X_train.columns
                    if X_train[cname].nunique() < 10 and X_train[cname].dtype == "object"]  # 10'dan az benzersiz değeri olan kategorik sütunları seçer.

numerical_cols = [cname for cname in X_train.columns
                  if X_train[cname].dtype in ["int64", "float64"]]  # Sayısal veri türüne sahip sütunları seçer.

print("The number of categorical columns: ", len(categorical_cols))  # Kategorik sütunların sayısını görüntüler.
print("The number of numerical columns: ", len(numerical_cols))  # Sayısal sütunların sayısını görüntüler.

my_cols = categorical_cols + numerical_cols  # Kategorik ve sayısal sütunları birleştirir.
X_train = X_train[my_cols]  # Eğitim setinde yalnızca bu sütunları tutar.
X_val = X_val[my_cols]  # Doğrulama setinde yalnızca bu sütunları tutar.
X_test = df_test[my_cols]  # Test setinde yalnızca bu sütunları tutar.

numerical_transformer = Pipeline(steps=[
    ("imputer_num", SimpleImputer(strategy="median")),  # Sayısal verilerdeki eksik değerleri ortanca ile doldurur.
    ("scaler", StandardScaler())  # Veriyi standartlaştırır (ortalama 0, standart sapma 1).
])

categorical_transformer = Pipeline(steps=[
    ("imputer_cal", SimpleImputer(strategy="most_frequent")),  # Kategorik verilerdeki eksik değerleri en sık görülen değerle doldurur.
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # Kategorik verileri one-hot encoding ile sayısal değerlere çevirir.
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols),  # Sayısal sütunlara sayısal dönüştürücüyü uygular.
    ("cat", categorical_transformer, categorical_cols)  # Kategorik sütunlara kategorik dönüştürücüyü uygular.
])

rf = RandomForestRegressor(random_state=0)  # Rastgele orman regresyon modeli, rastgelelik için `random_state=0`.
my_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", rf)])  # Ön işlemciyi ve modeli bir pipeline'da birleştirir.
my_pipeline.fit(X_train, y_train)  # Modeli eğitir.

val_preds = my_pipeline.predict(X_val)  # Doğrulama seti için tahminler yapar.
print("Validation MAE: ", mean_absolute_error(y_val, val_preds))  # Doğrulama seti üzerindeki MAE'yi hesaplar.

scores = -1 * cross_val_score(
    estimator=my_pipeline,  # Pipeline modelini kullanır.
    X=X,  # Bağımsız değişkenler.
    y=y,  # Bağımlı değişken.
    cv=5,  # 5 katlı çapraz doğrulama yapar.
    scoring="neg_mean_absolute_error"  # MAE'yi negatif olarak kullanır, çünkü daha küçük değerler daha iyidir.
)
print("Mean Cross Validation Score: ", scores.mean())  # Çapraz doğrulama ortalama MAE değerini yazdırır.

param_grid = {
    'model__n_estimators': [500, 600, 700],  # Rastgele ormandaki ağaç sayısını arar.
    'model__max_features': ['sqrt', 'log2'],  # Her bölünme için değerlendirilecek özelliklerin sayısını arar.
    'model__max_depth': [5, 6, 7],  # Ağacın maksimum derinliğini arar.
    'model__criterion': ['squared_error', 'absolute_error', 'poisson']  # Bölünme kriterini arar.
}

GridCV = GridSearchCV(
    estimator=my_pipeline,  # Pipeline'ı kullanır.
    param_grid=param_grid,  # Parametre ızgarasını kullanarak en iyi parametreleri arar.
    n_jobs=-1  # Tüm işlemcileri kullanarak paralel çalışmayı sağlar.
)
GridCV.fit(X_train, y_train)  # GridSearchCV ile en iyi modeli bulur.

print(GridCV.best_params_)  # En iyi parametre kombinasyonunu yazdırır.
print(GridCV.best_score_)  # En iyi modelin skorunu yazdırır.

preds_test = GridCV.predict(X_test)  # Test seti üzerinde tahminler yapar.
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': preds_test})  # Test tahminlerini bir DataFrame'e koyar.
output.head()  # İlk birkaç tahmini görüntüler.
output.to_csv(path_or_buf='C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/submission.csv', index=False)  # Sonuçları CSV dosyasına kaydeder.
