import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "6"

df = pd.read_csv("C:/Users/firat/OneDrive/Belgeler/Projects/PythonMLProjects/Projects/Gerekliler/fine_food_reviews_with_embeddings_1k.csv")

print(df.head(2))
print(df.dtypes)

matrix = np.array(df.embedding.apply(literal_eval).to_list())
print(matrix)
print(matrix.shape)

tsne = TSNE(n_components=2,
            perplexity=15,
            random_state=42,
            init="random",
            learning_rate=200)

vis_dims = tsne.fit_transform(matrix)
print(vis_dims.shape)

colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
x = [x for x, y in vis_dims]
y = [y for x, y in vis_dims]
print(df.Score.values[:10])

color_indices = df.Score.values - 1
colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
plt.title("Amazon rating visualized in language using t-SNE")
plt.show()
