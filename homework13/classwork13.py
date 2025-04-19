# Конспект лекции 13
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

# ## Метод опорных векторов (SVM - support vector machine)
# Классификация и регрессия
# 
# **Разделяющая классификация** - разделение классов данных (с помощью линий).
# 
# Выбирается линия с максимальным отступом. Точки, через которые проходят параллельные прямые, - опорные векторы.

iris = sns.load_dataset('iris')

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]

X = data_df[['sepal_length','petal_length']]
y = data_df['species']

data_df_setosa = data_df[data_df['species'] == 'setosa']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']


# Если C большое, то отступ задается "жестко".
# Чем меньше С, тем отступ становится более "размытым"


model = SVC(kernel='linear', C=10000)
model.fit(X, y)


plt.scatter(data_df_setosa['sepal_length'],data_df_setosa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'],data_df_versicolor['petal_length'])

plt.scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1], 
            s=400, 
            facecolor='none',
            edgecolors='black'
)

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

plt.scatter(X_p_setosa['sepal_length'], X_p_setosa['petal_length'], alpha = 0.2)
plt.scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha = 0.2)


# В случае перекрытия данных идеальной границы не существует. 
# У модели существует гиперпараметр, который определяет "размытие" отступа.


data_df = data[(data['species'] == 'virginica') | (data['species'] == 'versicolor')]
X = data_df[['sepal_length','petal_length']]
y = data_df['species']

data_df_virginica = data_df[data_df['species'] == 'virginica']
data_df_versicolor = data_df[data_df['species'] == 'versicolor']

c_value = [[10000, 1000, 100, 10], [1, 0.1, 0.01, 0.001]]

fig,ax = plt.subplots(2, 4, sharex='col', sharey='row')

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
    columns = ['sepal_length','petal_length']
)

for i in range(2):
    for j in range(4):
        X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
        columns = ['sepal_length','petal_length']
        )
        
        ax[i][j].scatter(data_df_virginica['sepal_length'],data_df_virginica['petal_length'])
        ax[i][j].scatter(data_df_versicolor['sepal_length'],data_df_versicolor['petal_length'])
        
        model = SVC(kernel='linear', C=c_value[i][j])
        model.fit(X, y)

        ax[i][j].scatter(model.support_vectors_[:,0],
            model.support_vectors_[:,1], 
            s=400, 
            facecolor='none',
            edgecolors='black'
        )
        
        y_p = model.predict(X_p)

        X_p['species'] = y_p

        X_p_virginica = X_p[X_p['species'] == 'virginica']
        X_p_versicolor = X_p[X_p['species'] == 'versicolor']

        ax[i][j].scatter(X_p_virginica['sepal_length'], X_p_virginica['petal_length'], alpha = 0.01)
        ax[i][j].scatter(X_p_versicolor['sepal_length'], X_p_versicolor['petal_length'], alpha = 0.01)


# **Достоинства:**
# - Зависимость от небольшого числа опорных векторов => компактность модели
# - После обучения предсказания проходят очень быстро
# - На работу метода влияют только точки, находящиеся возле отступов, поэтому методы подходят для многомерных данных
# 
# **Недостатки:**
# - При большом количестве обучающих образцов могут быть значительные вычислительные затраты
# - Большая зависимость от размытости C. Поиск может привести к большим вычислительным затратам
# - У результатов отсутсвует вероятностная интерпертация

# ## Деревья решений и случайные леса
# **Случайные леса** - непараметрический алгоритм (неизвестна формула).
# Это пример ансамблевого метода, основанного на агрегации результатов множества простых моделей.
# 
# В реализациях дерева принятия решений в машинном обучении вопросы обычно ведут к разделению данных по осям,
# то есть каждый узел разбивает данные на две группы


from sklearn.tree import DecisionTreeClassifier

species_int = []
for r in iris.values:
    match r[4]:
        case 'setosa':
            species_int.append(1)
        case 'versicolor':
            species_int.append(2)
        case 'virginica':
            species_int.append(3)
            
species_int_df = pd.DataFrame(species_int)

data = iris[['sepal_length', 'petal_length']]
data['species'] = species_int_df

data_df = data[(data['species'] == 3) | (data['species'] == 2)]

X = data_df[['sepal_length','petal_length']]
y = data_df['species']

data_df_setosa = data_df[data_df['species'] == 3]
data_df_versicolor = data_df[data_df['species'] == 2]

x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

max_depth = [[1, 2, 3, 4], [5, 6, 7, 9]]

fig,ax = plt.subplots(2, 4, sharex='col', sharey='row')

for i in range(2):
    for j in range(4):
        X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
        columns = ['sepal_length','petal_length']
        )
        
        ax[i][j].scatter(data_df_setosa['sepal_length'],data_df_setosa['petal_length'])
        ax[i][j].scatter(data_df_versicolor['sepal_length'],data_df_versicolor['petal_length'])

        model = DecisionTreeClassifier(max_depth=max_depth[i][j])
        model.fit(X, y)


        y_p = model.predict(X_p)

        X_p['species'] = y_p

        X_p_setosa = X_p[X_p['species'] == 3]
        X_p_versicolor = X_p[X_p['species'] == 2]

        ax[i][j].contourf(
            X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder=1
        )