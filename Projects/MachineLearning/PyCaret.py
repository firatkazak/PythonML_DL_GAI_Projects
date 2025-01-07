import pandas as pd
from pycaret.classification import *
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/heart.csv")
print(df.shape)

data = df.sample(frac=0.95, random_state=0)
print(data.head())

data_unseen = df.drop(data.index)
data.reset_index(inplace=True, drop=True)
print(data.head())

data_unseen.reset_index(inplace=True, drop=True)
print(data["DEATH"].value_counts())

model = setup(data=data,
              target="DEATH",
              normalize=True,
              normalize_method="minmax",
              train_size=0.8,
              fix_imbalance=True,
              fix_imbalance_method=RandomOverSampler(),
              session_id=0)

knn = create_model("knn")
tuned_knn = tune_model(knn)

plot_model(tuned_knn, plot="auc", save=True)  # AUC diye kaydediliyor.
plot_model(tuned_knn, plot="pr", save=True)  # Precision Recall diye kaydediliyor. Bunu Precision Recall_First.png olarak değiştirdim.

predict_model(tuned_knn)
best = compare_models()
tuned_best = tune_model(best)

plot_model(tuned_best, plot="pr", save=True)  # Precision Recall diye kaydediliyor. Bunu Precision Recall_Tuned.png olarak değiştirdim.

final_best = finalize_model(tuned_best)
predict_model(final_best)

result_predict_model = predict_model(final_best, data=data_unseen)
print(result_predict_model)
#    AGE_50  MD_50  SBP_50  ...  DEATH  prediction_label  prediction_score
# 0      40      3     120  ...      1                 0            0.7347
# 1      54      1     141  ...      1                 1            0.7443
# 2      52      4     145  ...      0                 1            0.5978
# 3      34      2     130  ...      0                 0            0.7665
# 4      51      3     138  ...      0                 1            0.5348
# 5      43      1     120  ...      1                 0            0.8122
# 6      25      1     110  ...      1                 0            0.9735
# 7      61      1     150  ...      1                 0            0.5411
# 8      36      3     135  ...      0                 0            0.9677
# 9      49      2     110  ...      0                 0            0.6532
