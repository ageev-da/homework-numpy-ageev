# Домашнее задание 15
# Выполнил: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

### Задание № 1
# Обучение с учителем (классификация). Выбрать ДВА ЛЮБЫХ СОРТА и для них реализовать.
# 1. Метод опорных векторов
# 2. Метод главных компонент
# Обучение без учителя (классификация).
# 3. Метод k средних


## 1. Реализация метода опорных векторов

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

iris = sns.load_dataset('iris')

data = iris[['sepal_length','sepal_width','petal_width','species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length','sepal_width','petal_width']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

model = SVC(kernel='linear', C=10000)
model.fit(X, y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(data_df_virginica['sepal_length'],data_df_virginica['sepal_width'],data_df_virginica['petal_width'],c='red',label='virginica')
ax.scatter(data_df_versicolor['sepal_length'],data_df_versicolor['sepal_width'],data_df_versicolor['petal_width'],c='green',label='versicolor')

x_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
y_p = np.linspace(min(data_df['sepal_width']), max(data_df['sepal_width']), 100)
z_p = np.linspace(min(data_df['petal_width']), max(data_df['petal_width']), 100)

X, Y = np.meshgrid(x_p,y_p)

w = model.coef_[0]
b = model.intercept_[0]
Z = (-w[0] * X - w[1] * Y - b) / w[2]
ax.plot_surface(X, Y, Z, alpha=0.7)
ax.set_xlabel('sepal_length')
ax.set_ylabel('sepal_width')
ax.set_zlabel('petal_width')
ax.legend()
ax.view_init(15, 35)
plt.tight_layout()
plt.savefig(
    'SVC_image.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.3
)
plt.show()

### 2. Реализация метода главных компонент
data = iris[['sepal_length', 'sepal_width', 'petal_width', 'species']]

def plot_pca_3d(data_subset, ax, title):
    p_full = PCA(n_components=3)
    p_full.fit(data_subset)
    
    p1 = PCA(n_components=1)
    X_p1 = p1.fit_transform(data_subset)
    X_p1_new = p1.inverse_transform(X_p1)
    
    ax.scatter(
        data_subset['sepal_length'],
        data_subset['sepal_width'],
        data_subset['petal_width'],
        c='red', label='Original'
    )
    
    # Векторы главных компонент (PC1, PC2, PC3)
    for i in range(3):
        vec = p_full.components_[i]
        length = np.sqrt(p_full.explained_variance_[i]) * 3
        start = p_full.mean_
        end = start + vec * length
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            linewidth=2,
            label=f'PC{i+1}' if i == 0 else None  # метка только для первой, чтобы не дублировать в легенде
        )
    
    ax.scatter(
        X_p1_new[:, 0],
        X_p1_new[:, 1],
        X_p1_new[:, 2],
        c='black',
        alpha=0.4,
        label='Projection to PC1'
    )
    
    ax.set_xlabel('sepal_length')
    ax.set_ylabel('sepal_width')
    ax.set_zlabel('petal_width')
    ax.set_title(title)
    ax.legend()

data_virginica  = data[data['species'] == 'virginica'].drop(columns='species')
data_versicolor = data[data['species'] == 'versicolor'].drop(columns='species')

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_pca_3d(data_virginica, ax1, 'Virginica')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_pca_3d(data_versicolor, ax2, 'Versicolor')

plt.tight_layout()
plt.savefig(
    'PCA_image.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.3, 
)
plt.show()

### 3. Реализация метода k-средних

# Функция для отрисовки 3D-кластеров
def plot_kmeans_3d(data_subset, ax, labels=None, centers=None, title=''):
    # Рисуем либо по меткам, либо по цветам из data_subset
    if labels is not None:
        ax.scatter(
            data_subset[:, 0], data_subset[:, 1], data_subset[:, 2],
            c=labels, s=30, cmap='rainbow', edgecolor='k'
        )
    else:
        Xsub, ysub, color = data_subset
        ax.scatter(
            Xsub[:, 0], Xsub[:, 1], Xsub[:, 2],
            c=color, s=30, label=ysub, edgecolor='k'
        )
    # Рисуем центры, если указаны
    if centers is not None:
        ax.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            c='black', s=200, alpha=0.5, label='centers'
        )
    ax.set_xlabel('sepal_length')
    ax.set_ylabel('sepal_width')
    ax.set_zlabel('petal_width')
    ax.set_title(title)
    ax.legend()

data = iris[['sepal_length','sepal_width','petal_width','species']]
data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length','sepal_width','petal_width']].values
y = data_df['species'].values

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
pred = kmeans.predict(X)
centers = kmeans.cluster_centers_

data_vir = data_df[data_df['species']=='virginica'][['sepal_length','sepal_width','petal_width']].values
data_ver = data_df[data_df['species']=='versicolor'][['sepal_length','sepal_width','petal_width']].values

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plot_kmeans_3d(
    data_subset=(data_vir, 'virginica', 'red'),
    ax=ax1,
    title='Original data'
)
plot_kmeans_3d(
    data_subset=(data_ver, 'versicolor', 'purple'),
    ax=ax1
)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plot_kmeans_3d(
    data_subset=X,
    ax=ax2,
    labels=pred,
    centers=centers,
    title='KMeans prediction'
)

plt.tight_layout()
plt.savefig(
    'Kmeans_image.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.3
)
plt.show()

