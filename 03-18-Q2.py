import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel('data/A01-data0318.xlsx')
data.fillna(0, inplace=True)
print(data.corr())
sns.heatmap(
    data.corr(),
    annot=False,
    cmap="coolwarm",
    fmt='.4f',
)
plt.show()