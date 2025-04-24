# Конспект лекции 14
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

# ## Переобучение
# Присуще всем деревьям принятия решений


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

iris = sns.load_dataset('iris')

# sns.pairplot(iris, hue='species')

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

print(data.head())
print(data.shape)

data_setosa = data[data['species'] == 1]
data_versicolor = data[data['species'] == 2]
data_virginica = data[data['species'] == 3]

print(data_setosa.shape, data_versicolor.shape, data_virginica.shape)


# Разделение данных на две части


data_versicolor_A = data_versicolor.iloc[:25, :]
data_versicolor_B = data_versicolor.iloc[25:, :]
# print(data_versicolor_A.shape, data_versicolor_B.shape)

data_virginica_A = data_virginica.iloc[:25, :]
data_virginica_B = data_virginica.iloc[25:, :]

data_df_A = pd.concat([data_versicolor_A, data_virginica_A], ignore_index=True)
data_df_B = pd.concat([data_versicolor_B, data_virginica_B], ignore_index=True)
# print(data_df_A)

x1_p = np.linspace(min(data['sepal_length']), max(data['sepal_length']), 100)
x2_p = np.linspace(min(data['petal_length']), max(data['petal_length']), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(
        np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
        columns = ['sepal_length','petal_length']
)


fig,ax = plt.subplots(2, 4, sharex='col', sharey='row')

max_depth = [1, 3, 5, 7]

X = data_df_A[['sepal_length', 'petal_length']]
y = data_df_A['species']

j = 0

for md in max_depth:
    
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X,y)
    
    ax[0][j].scatter(data_virginica_A['sepal_length'], data_virginica_A['petal_length'])
    ax[0][j].scatter(data_versicolor_A['sepal_length'], data_versicolor_A['petal_length'])

    y_p = model.predict(X_p)
    ax[0][j].contourf(
            X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder=1
    )
    j += 1

X = data_df_B[['sepal_length', 'petal_length']]
y = data_df_B['species']

j = 0

for md in max_depth:
    
    model = DecisionTreeClassifier(max_depth=md)
    model.fit(X,y)
    
    ax[1][j].scatter(data_virginica_B['sepal_length'], data_virginica_B['petal_length'])
    ax[1][j].scatter(data_versicolor_B['sepal_length'], data_versicolor_B['petal_length'])

    y_p = model.predict(X_p)
    ax[1][j].contourf(
            X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder=1
    )
    j += 1


# ## Ансамблевые методы
# В их основе лежит идея объединения нескольких переобученных (!) моделей для уменьшения эффекта переобучения.
# Это называется **баггинг** (bagging).
# 
# Баггинг усредняет результаты, что ведет к оптимальной классификации.
# 
# Ансамбль случайных деревьев называется **случайным лесом**.


fig,ax = plt.subplots(1, 3, sharex='col', sharey='row')

ax[0].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
ax[0].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
ax[0].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])

md = 6

X = data[['sepal_length', 'petal_length']]
y = data['species']

model1 = DecisionTreeClassifier(max_depth=md)
model1.fit(X,y)

y_p = model1.predict(X_p)
ax[0].contourf(
    X1_p, X2_p, y_p.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder=1
)

# Bagging
ax[1].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
ax[1].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
ax[1].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])

model2 = DecisionTreeClassifier(max_depth=md)
b = BaggingClassifier(model2, n_estimators=20, max_samples=0.8, random_state=1)
b.fit(X,y)

y_p2 = b.predict(X_p)

ax[1].contourf(
    X1_p, X2_p, y_p2.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder=1
)

# Random Forest
ax[2].scatter(data_setosa['sepal_length'], data_setosa['petal_length'])
ax[2].scatter(data_versicolor['sepal_length'], data_versicolor['petal_length'])
ax[2].scatter(data_virginica['sepal_length'], data_virginica['petal_length'])

model3 = RandomForestClassifier(n_estimators=20, max_samples=0.8, random_state=1)
model3.fit(X,y)

y_p3 = model3.predict(X_p)

ax[2].contourf(
    X1_p, X2_p, y_p3.reshape(X1_p.shape), alpha=0.4, levels=2, cmap='rainbow', zorder=1
)


# ## Регрессия с помощью случайных лесов
# 


data = iris[['sepal_length', 'petal_length','species']]

data_setosa = data[data['species'] == 'setosa']

x_p = pd.DataFrame(
    np.linspace(min(data_setosa['sepal_length']), max(data_setosa['sepal_length']), 100)
)

X = pd.DataFrame(data_setosa['sepal_length'], columns=['sepal_length'])
y = data_setosa['petal_length']

model = RandomForestRegressor(n_estimators=20)
model.fit(X,y)
y_p = model.predict(x_p)

plt.scatter(data_setosa['sepal_length'], data_setosa['petal_length'])

plt.plot(x_p, y_p)


# **Достоинства:**
# - Простота и быстрота. Возможно распараллеливание процесса -> выигрыш во времени
# - Вероятностная классификация
# - Модель непараметрическая => хорошо работает с задачами, где другие модели могут оказаться недообученными
# 
# **Недостатки:**
# - Сложно интерпретировать

# ## Метод главных компонент (PCA - principal component analysis)
# Алгоритм машинного обучения БЕЗ учителя. Часто используют для понижения размерности
# 
# Задача машинного обучения БЕЗ учителя состоит в выяснении зависимости между признаками.
# 
# В PCA выполняется качественная оценка этой зависимости путем поиска главных осей координат
# и их дальнейшего использования для описания набора данных


data = iris[['petal_width', 'petal_length', 'species']]

data_v = data[data['species'] == 'versicolor']
data_v = data_v.drop(columns='species')
# print(data_v)

X = data_v['petal_width']
Y = data_v['petal_length']

p = PCA(n_components=2)
p.fit(data_v)
X_p = p.transform(data_v)
print(data_v.shape, X_p.shape)

print(p.components_)
print(p.explained_variance_)
print(p.mean_)

p1 = PCA(n_components=1)
p1.fit(data_v)
X_p1 = p1.transform(data_v)
print(data_v.shape, X_p1.shape)

print(p1.components_)
print(p1.explained_variance_)
print(p1.mean_)

X_p1_new = p1.inverse_transform(X_p1)
print(X_p1_new.shape)


plt.scatter(X, Y)

# Центр главных осей
plt.scatter(p.mean_[0], p.mean_[1])


plt.plot(
    [
        p.mean_[0], 
        p.mean_[0] + p.components_[0][0] * np.sqrt(p.explained_variance_[0])
    ],
    [
        p.mean_[1], 
        p.mean_[1] + p.components_[0][1] * np.sqrt(p.explained_variance_[0])
    ]
)

plt.plot(
    [
        p.mean_[0], 
        p.mean_[0] + p.components_[1][0] * np.sqrt(p.explained_variance_[1])
    ],
    [
        p.mean_[1], 
        p.mean_[1] + p.components_[1][1] * np.sqrt(p.explained_variance_[1])
    ]
)

plt.scatter(X_p1_new[:,0], X_p1_new[:,1], alpha=0.6)


# **Достоинства:**
# - Простота интерпретации
# - Эффективность в работе с многомерными данными
# 
# **Недостатки:**
# - Аномальные значения в данных оказывают сильное влияние на результат