import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/firat/source/repos/PythonMLDersleri/Gerekliler/googleplaystore.csv")

df.columns = df.columns.str.replace(" ", "_")

sns.set_theme()
sns.set(rc={"figure.dpi": 100, "figure.figsize": (15, 10)})
sns.heatmap(df.isnull(), cbar=False)
plt.show()
