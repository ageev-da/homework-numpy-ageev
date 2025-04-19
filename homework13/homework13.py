# Домашнее задание 13
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

# Задание: Убрать из данных iris часть точек (на которых обучаемся) и убедиться, 
# что на предсказание влияют только опорные вектора

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')
data = iris[['sepal_length', 'petal_length', 'species']]

# Максимальный шаг пропуска данных
max_step = 2

fig,ax = plt.subplots(max_step , sharex='col')

for i in range(max_step):
    data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')][::i+1]
    X = data_df[['sepal_length','petal_length']]
    y = data_df['species']

    data_df_setosa = data_df[data_df['species'] == 'setosa']
    data_df_versicolor = data_df[data_df['species'] == 'versicolor']

    model = SVC(kernel='linear', C=10000)
    model.fit(X, y)

    ax[i].scatter(data_df_setosa['sepal_length'],data_df_setosa['petal_length'])
    ax[i].scatter(data_df_versicolor['sepal_length'],data_df_versicolor['petal_length'])

    ax[i].scatter(model.support_vectors_[:,0],
                model.support_vectors_[:,1], 
                s=400, 
                facecolor='none',
                edgecolors='black'
    )

    if not i:
        x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
        x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)
        X1_p, X2_p = np.meshgrid(x1_p, x2_p)

    X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
        columns = ['sepal_length','petal_length']
    )

    y_p = model.predict(X_p)

    X_p['species'] = y_p

    X_p_setosa = X_p[X_p['species'] == 'setosa']
    X_p_versicolor = X_p[X_p['species'] == 'versicolor']

    ax[i].scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha = 0.05)
    ax[i].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha = 0.05)