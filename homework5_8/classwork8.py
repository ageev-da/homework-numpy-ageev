
# Конспект лекции 8
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


### Трехмерные точки и линии
fig = plt.figure()
ax = plt.axes(projection='3d') # 3d - xyz

z1 = np.linspace(0, 15, 1000)
y1 = np.cos(z)
x1 = np.sin(z)

ax.plot3D(x1,y1,z1, 'green') # построение 3d графика

z2 = 15 * np.random.random(100)
y2 = np.cos(z2) + 0.1 * np.random.random(100)
x2 = np.sin(z2) + 0.1 * np.random.random(100)

ax.scatter3D(x2,y2,z2,c=z2,cmap='Greens')

plt.show()



## построение контуров
def f(x,y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

fig = plt.figure()
ax = plt.axes(projection='3d') # 3d - xyz

x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)

ax.contour3D(X, Y, Z, 40, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


## НАСТРОЙКА угла просмотра графика
# первое - угол относительно плоскости xy, второе - относительно z
ax.view_init(60, 45)

plt.show()



## построение контуров - только через точки
def f(x,y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

fig = plt.figure()
ax = plt.axes(projection='3d') # 3d - xyz

x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)

ax.scatter3D(X,Y,Z, c=Z, cmap='Greens')

## НАСТРОЙКА угла просмотра графика
# первое - угол относительно плоскости xy, второе - относительно z
ax.view_init(60, 45)

plt.show()



## построение контуров - wireframe (каркасный)
def f(x,y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

fig = plt.figure()
ax = plt.axes(projection='3d') # 3d - xyz

x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)

ax.plot_wireframe(X,Y,Z)

## НАСТРОЙКА угла просмотра графика
# первое - угол относительно плоскости xy, второе - относительно z
ax.view_init(60, 45)

plt.show()


## построение контуров - поверхностный
def f(x,y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

fig = plt.figure()
ax = plt.axes(projection='3d') # 3d - xyz

x = np.linspace(-6,6,30)
y = np.linspace(-6,6,30)
X,Y = np.meshgrid(x,y)
Z = f(X,Y)

ax.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none')
ax.set_title('Example')
## НАСТРОЙКА угла просмотра графика
# первое - угол относительно плоскости xy, второе - относительно z
ax.view_init(60, 45)

plt.show()



## демонстрация срезов
def f(x,y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

fig = plt.figure()
ax = plt.axes(projection='3d') # 3d - xyz

r = np.linspace(0,6,20)
theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)

R, Theta = np.meshgrid(r, theta)

X = r * np.sin(Theta)
Y = r * np.cos(Theta)

Z = f(X,Y)

ax.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none')

plt.show()



# трингуляция - заменяет точки на сплошной набор треугольников
def f(x,y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

fig = plt.figure()
ax = plt.axes(projection='3d') # 3d - xyz

theta = 2 * np.pi + np.random.random(1000)
r = 6 * np.random.random(1000)

x = r * np.sin(theta)
y = r * np.cos(theta)

z = f(x,y)

ax.scatter(x,y,z,c=z,cmap='viridis')

ax.plot_trisurf(x,y,z,cmap='viridis')

plt.show()



### SEABORN 
# Преимущества
# - работает с DataFrame (MAtplotlib с Pandas)
# - более высокоуровневый

data = np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)
data = pd.DataFrame(data,columns=['x','y'])

print(data.head())

fig = plt.figure()
plt.hist(data['x'], alpha = 0.5)
plt.hist(data['y'], alpha = 0.5)

# аналог на seaborn
# другие мат методы - ядерная оценка - за счет этого гладко аппроксимируется, и значения немного другие
fig = plt.figure()
sns.kdeplot(data=data, fill=True) # shade/fill - закрашивание

# ВСЕ РАВНО В КОНЦЕ ДОЛЖНО БЫТЬ
plt.show()



# в seaborn есть набор загруженных датасетов
iris = sns.load_dataset('iris')
print(iris.head())

# попарное сравенение элементов
sns.pairplot(iris, hue='species', height=2.5) # hue - позволяет разделять на категории элементы

plt.show()


## гистограммы подмножеств

tips = sns.load_dataset('tips')
print(tips.head())

grid = sns.FacetGrid(tips,col='sex',row='day',hue='time')
grid.map(plt.hist, 'tip', bins=np.linspace(0,40,15))

plt.show()


## графики факторов

# sns.catplot(data=tips, x='day', y='total_bill', kind='box')

sns.jointplot(data=tips, x='tip', y='total_bill', kind='hex')

plt.show()


# графики временных рядов
# planets - датасет с инф о том, когда и какие планеты открывали
planets = sns.load_dataset('planets')
print(planets.head())

sns.catplot(data=planets, x='year', kind='count', hue='method', order=range(2005, 2015))

plt.show()



# диаграммы для анализа датасетов

tips = sns.load_dataset('tips')
print(tips.head())

## Cравнение числовых данных
# Числовые пары
fig=plt.figure()
sns.pairplot(tips)

# тепловая карта
tips_corr = tips[['total_bill', 'tip', 'size']]

fig=plt.figure()
sns.heatmap(tips_corr.corr(),cmap='RdBu_r', annot=True,vmin=-1,vmax=1)
# значения 0 - независимы
# 1 - положительная зависимость (чем больше одно - тем больше другое)
# -1 - отрицательная (обратнопропорционально)


# диаграмма рассеяния
fig = plt.figure()
sns.scatterplot(data=tips,x='total_bill',y='tip', hue='sex')

# диаграмма рассеяния с лин регрессией
fig = plt.figure()
sns.regplot(data=tips,x='total_bill',y='tip')
#sns.relplot(data=tips,x='total_bill',y='tip',hue='sex')

# простой линейный график
fig = plt.figure()
sns.lineplot(data=tips,x='total_bill',y='tip')

# сводная диаграмма
fig = plt.figure()
sns.jointplot(data=tips,x='total_bill',y='tip')
plt.show()



## Cравнение числовых и категориальных данных
# Гистограмма
fig = plt.figure()
sns.barplot(data=tips,y='total_bill',x='day',hue='sex')

# точечное
fig = plt.figure()
sns.pointplot(data=tips,y='total_bill',x='day',hue='sex')

# Ящик "с усами"
fig = plt.figure()
sns.boxplot(data=tips,y='total_bill',x='day')

# Скрипичная диаграмма
fig = plt.figure()
sns.violinplot(data=tips,y='total_bill',x='day')

# Одномерные диаграммы рассеяния
fig = plt.figure()
sns.stripplot(data=tips,y='total_bill',x='day')

plt.show()





