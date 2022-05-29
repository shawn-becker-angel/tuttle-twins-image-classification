
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
sns.set_theme(style="dark")

print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("seaborn version:", sns.__version__)
print("matplotlib version:", matplotlib.__version__)


# install seaborn csv datasets using:
# cd ~/ && git clone git@github.com:mwaskom/seaborn-data.git
print("loading data...")
penguins = pd.read_csv("~/seaborn-data/penguins.csv")
print(penguins.head(3))

print("loading seaborn and matplotlib modules...")

def show_with_title(title):
    '''Plots title and prompts to close window'''
    fig = plt.figure(1)
    fig.canvas.manager.set_window_title(title)
    print(f"waiting for you close '{title}'")
    plt.show()

title = "DataFrame HIST of univariate"
plt.figure()
penguins['flipper_length_mm'].hist(
    bins=50, 
    color='black', 
    alpha=0.5)
plt.xlabel("Flipper Length mm")
plt.ylabel("Frequency")
show_with_title(title)

title = "DISPLOT of DataFrame univariate with KDE"
sns.displot(
    penguins, 
    bins=50,
    x="flipper_length_mm",
    kde=True)
# smoothed curve only
# sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=1.0)
show_with_title(title)

# Simulate data from a bivariate Gaussian
n = 10000
mean = [0, 0]
cov = [(2, .4), (.4, .2)]
rng = np.random.RandomState(0)
x, y = rng.multivariate_normal(mean, cov, n).T

title = "combo HISTOGRAM and SCATTERPLOT with DENSITY CONTOURS"
f, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=x, y=y, s=5, color=".15")
sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
show_with_title(title)

title = "PAIR PLOT of 4 attributes"
sns.pairplot(penguins)
show_with_title(title)

title = "PAIR GRID of 4 attributes using HISTPLOT, KDEPLOT, and HISTPLOT+KDE"
g = sns.PairGrid(penguins)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=True)
g.map_diag(sns.histplot, kde=True)
show_with_title(title)

title = "DISPLOT of bivariate with KDE"
sns.displot(
    data=penguins, 
    x="bill_length_mm", 
    y="bill_depth_mm", 
    hue="species", 
    kind="kde")
show_with_title(title)

title = "JOINTPLOT with MARGINAL DISTRIBUTIONS"
sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
show_with_title(title)

title = "JOINTPLOT with MARGINAL KDE DISTRIBUTIONS"
sns.jointplot(
    data=penguins,
    x="bill_length_mm", 
    y="bill_depth_mm", 
    hue="species",
    kind="kde")
show_with_title(title)

print("done")

