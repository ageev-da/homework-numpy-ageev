
# Конспект лекции 6
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
data = rng.normal(size = 1000)

### Одномерная гистограмма
plt.hist(data, 
         bins=30,
         density=True,
         alpha=0.5, # прозрачность
         histtype='step', # не заливать
         edgecolor='red' # цвет грани
)
plt.show()



x1 = rng.normal(0, 0.8, 1000)
x2 = rng.normal(-2, 1, 1000)
x3 = rng.normal(3, 2, 1000)

# параметры через переменные
args = dict(
    alpha=0.3,
    bins=40
)

plt.hist(x1, **args)
plt.hist(x2, **args)
plt.hist(x3, **args)

plt.show()

# посчет данных - метод np.histogram - array() - границы min max
print(np.histogram(x1,bins=1))
print(np.histogram(x1,bins=2))
print(np.histogram(x1,bins=40))



### Двумерная гистограмма

mean = [0,0]
cov = [[1,1], [1,2]] # ков матрица

x, y = rng.multivariate_normal(mean, cov, 10000).T

# plt.hist2d(x, y, bins=30) # рисует квадратиками

plt.hexbin(x, y, gridsize=30) # шестиугольными

cb = plt.colorbar()
cb.set_label('Point in interval')
plt.show()

# если не хотим рисовать
print(np.histogram2d(x, y, bins=1))
print(np.histogram2d(x, y, bins=10))



### Легенды

x = np.linspace(0,10,1000)
fir, ax = plt.subplots()

y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0,2,0.5))

lines = plt.plot(x,y) # вернет plt.Line2D

# задание label в legend, loc - положение
# plt.legend(lines, ['1', 'второй', 'third', '4-ый'], loc='upper left')

# можно задавать не всем
plt.legend(lines[:2], ['1', '2'])

#ax.plot(x, np.sin(x), label='Синус')
#ax.plot(x, np.cos(x), label='Косинус')
#ax.plot(x, np.cos(x) + 2)
#ax.axis('equal')

# по умолчанию все label, которые есть, добавляются на график
#ax.legend(
#    frameon=True, #legend box
#    fancybox=True,
#    shadow=True
#)

plt.show()



### Легенда 2
# загрузка данных в систему
cities = pd.read_csv('./data/california_cities.csv')

lat, lon, pop, area = cities['latd'], cities['longd'], cities['population_total'], cities['area_total_km2']

# s - size - размер точки зависит от area, с - цвет от pop
plt.scatter(lon, lat, c=np.log10(pop), s=area)
plt.xlabel('Широта')
plt.ylabel('Долгота')
plt.colorbar()
plt.clim(3,7)

# подставляем пустые графики, чтобы появилась легенда
plt.scatter([],[],c='k', alpha=0.5, s=100,label='100 $km^2$')
plt.scatter([],[],c='k', alpha=0.5, s=300,label='300 $km^2$')
plt.scatter([],[],c='k', alpha=0.5, s=500,label='500 $km^2$')

# labelspacing - расстояние между элементами в легенде
plt.legend(labelspacing=2.1, frameon=False)

plt.show()


# Легенда 3 - несколько легенд на графике

fig, ax = plt.subplots()

lines = []
styles = ['-', '--', '-.', ':']
x = np.linspace(0,10,1000)

for i in range(4):
    lines += ax.plot(
        x,
        np.sin(x - i + np.pi/2),
        styles[i]
    )

ax.axis('equal')

ax.legend(lines[:2], ['line 1', 'line 2'], loc='upper right')
# просто так не можем ещё одну добавить, надо создать новый слой

leg = mpl.legend.Legend(ax, lines[1:], ['line 2', 'line 3', 'line 4'], loc='lower left')

ax.add_artist(leg)

plt.show()



### Шкалы

x = np.linspace(0,10,1000)
y = np.sin(x) * np.cos(x[:, np.newaxis])

## Карты цветов
# 1 - последовательные
# 2 - дивергентные - есть 2 цвета на концах - один переходит в другой
# 3 - качественные - цвета смешиваются беспорядочно

# случай 1
# plt.imshow(y, cmap='binary')
# plt.imshow(y, cmap='viridis')



# случай 2
# plt.imshow(y, cmap='RdBu')
#plt.imshow(y, cmap='PuOr')


# случай 3
# plt.imshow(y, cmap='rainbow')
plt.imshow(y, cmap='jet')
plt.colorbar()

plt.show()



plt.figure()

plt.subplot(1,2,1)
plt.imshow(y, cmap='viridis')
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(y, cmap=plt.cm.get_cmap('viridis',6)) # дискретно делим на 6 частей
plt.colorbar()
plt.clim(-0.25,0.25)



plt.show()



### Доп оси координат в матлаб виде

ax1 = plt.axes()
# [нижний угол, левый уголб ширина, высота] - в процентах; ширина и высота от всего рисунка, а не от внутренних координат
ax2 = plt.axes([0.4, 0.3, 0.2, 0.1]) # ещё одна СК

ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

plt.show()



### Доп оси координат в объектно-ориент виде 
fig = plt.figure()

ax1 = fig.add_axes([0.1, 0.6, 0.8, 0.4])
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4])

ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

plt.show()



### Простые сетки

fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(1,7):
    ax = fig.add_subplot(2,3,i)
    ax.plot(np.sin(x + np.pi / 4 * i))

plt.show()

