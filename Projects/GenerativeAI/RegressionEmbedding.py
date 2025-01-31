import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(filepath_or_buffer="C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/fine_food_reviews_with_embeddings_1k.csv", index_col=0)

df.head(2)

df["embedding"] = df.embedding.apply(literal_eval).apply(np.array)
X_train, X_test, y_train, y_test = train_test_split(list(df.embedding.values), df.Score, random_state=42)
rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)

preds = rfr.predict(X_test)
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"ada-002 embedding performance on 1k Amazon reviews: \
    mse={mse:.2f}, mae={mae:.2f}")

bmse = mean_squared_error(y_test, np.repeat(y_test.mean(), len(y_test)))
bmae = mean_absolute_error(y_test, np.repeat(y_test.mean(), len(y_test)))
print(f"Dummy mean prediction performance on 1k Amazon reviews: \
    bmse={bmse:.2f}, bmae={bmae:.2f}")

pred1 = rfr.predict(X_test[:1])
print("Tahmin: ", pred1)
