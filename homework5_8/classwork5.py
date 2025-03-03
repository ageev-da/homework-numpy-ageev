
# Конспект лекции 5
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

# 1. сценарий
# 2. командная оболочка IPython
# 3. Jupyter


# Рассмотрим 1
# plt.show() - запускается 1 раз
# figure 

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

x = np.linspace(0,10,100)
plt.plot(x, np.sin(x))
#plt.show()

plt.plot(x, np.cos(x))
plt.show()


# IPython
# %matplotlib
# import matplotlib.pyplot as plt
# plt.plot(...)
# plt.draw() - если какие-то глюки

# Jupyter
# %matplotlib inline - в блокнот добавляется статическая картинка
# %matplotlib notebook - в блокнот добавляются интеграктивные графики

### СОХРАНЕНИЕ ГРАФИКОВ
## fig.savefig('saved_images.png')

print(fig.canvas.get_supported_filetypes()) # список доступных расширений



### Два способа вывода графиков
## - MATLAB-подобный стиль

x = np.linspace(0,10,100)

plt.figure() # создаем фигуру
plt.subplot(2,1,1) # layout - 2 строки, 1 колонка, номер
plt.plot(x,np.sin(x)) # первая картинка
plt.subplot(2,1,2)
plt.plot(x,np.cos(x)) # вторая

## - ОО (объектно-ориент) стиль

#когда одна:
# fig = plt.figure()
# ax = plt.axes()
# ax.plot(...)

fig, ax = plt.subplots(2)
ax[0].plot(x,np.sin(x))
ax[1].plot(x,np.cos(x))

# fig: plt.Figure - контейнер, который содержит объекты (СК, тексты, метки), ax:Axes - система координат - прямоугольник, деления, метки

plt.show()



### Различные настройки графиков
## Цвета линий color
# - 'blue'
# - 'rbgcmyk' -> 'rg'
# - '0.14' - градация серого от 0 - 1
# - 'RRGGBB' - 'FF00EE'
# - RGB - (1.0, 0.2, 0.3)
# - HTML - 'salmon'

## Стили линий linestyle
# - сплошная '-', 'solid'
# - штрихованная '--', 'dashed'
# - штрихпунктирная '-.', 'dashdot'
# - пунктирная ':', 'dotted'

fig = plt.figure()
ax = plt.axes()

ax.plot(x, np.sin(x), color = 'blue')
ax.plot(x, np.sin(x - 1), color = 'g', linestyle = 'solid')
ax.plot(x, np.sin(x - 2), color = '0.75', linestyle = 'dashed')
ax.plot(x, np.sin(x - 3), color = '#FF00EE', linestyle = 'dashdot')
ax.plot(x, np.sin(x - 4), color = (1.0, 0.2, 0.3), linestyle = 'dotted')
ax.plot(x, np.sin(x - 5), color = 'salmon')
# сокрашенное использование linestyle - комбинирование с color:
ax.plot(x, np.sin(x - 6), '--k')

plt.show()



fig, ax = plt.subplots(4)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.sin(x))
ax[2].plot(x, np.sin(x))
ax[3].plot(x, np.sin(x))

## настройка границ графика
ax[1].set_xlim(-2, 12)
ax[1].set_ylim(-1.5, 1.5)

## отзеркаливание графиков
ax[2].set_xlim(12, -2)
ax[2].set_ylim(1.5, -1.5)

## автоматическое масштабирование
ax[3].autoscale(tight=True)

plt.show()



## названия, легенды и тд
plt.subplot(3, 1, 1)
plt.plot(x, np.sin(x))

plt.title('Синус') # название графика
plt.xlabel('x') # подпись оси x
plt.ylabel('sin(x)')

plt.subplot(3, 1, 2)
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')

plt.title('Синус и косинус') # название графика
plt.xlabel('x') # подпись оси x
plt.ylabel('sin(x), cos(x)')

plt.legend() # легенда

plt.subplot(3, 1, 3)
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')

plt.title('Синус и косинус') # название графика
plt.xlabel('x') # подпись оси x
plt.ylabel('sin(x), cos(x)')
plt.axis('equal') # выравнивание

plt.legend() # легенда

### ПОЗВОЛЯЕТ УПРАВЛЯТЬ РАССТОЯНИЕМ МЕЖДУ ГРАФИКАМИ И ТД
plt.subplots_adjust(hspace = 1)
plt.show()



### отображение точек, маркеры бывают разными
x = np.linspace(0,10,30)
plt.plot(x,np.sin(x), 'o', color = 'green')
plt.plot(x,np.sin(x) + 1, '.', color = 'red')
plt.plot(x,np.sin(x) + 2, '>', color = 'blue')
plt.plot(x,np.sin(x) + 3, '^', color = 'black')
plt.plot(x,np.sin(x) + 4, 's', color = 'orange')
plt.show()



## задание линий с маркерами
x = np.linspace(0,10,30)
plt.plot(x, np.sin(x), '--p', markersize = 15, linewidth = 4, markerfacecolor='white', markeredgecolor='gray',markeredgewidth=2)
plt.show()



## специальный метод построение диаграмм
rng = np.random.default_rng(0)
colors = rng.random(30)
sizes = 100 * rng.random(30)
plt.scatter(x, np.sin(x), marker = 'o', c=colors, s = sizes)
plt.colorbar() # легенда

# если точек больше 1000, то plot предпочтительнее из-за производительности

plt.show()
# отличие от plot - позволяет задавать свойства каждой точки



## визуализация погрешности
x = np.linspace(0,10,50)
dy = 0.4
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x,y,yerr=dy, fmt='.k') # fmt='.k' - формат точки с errorbar
plt.fill_between(x, y - dy, y + dy, color = 'red', alpha = 0.4) # закрашивание ошибок


plt.show()



## 3D графики

def f(x,y): 
    return np.sin(x) ** 5 + np.cos(20 + x * y) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x,y)

Z = f(X,Y)

c = plt.contour(X, Y, Z, color='red')
#plt.contourf(X, Y, Z, cmap='RdGy') # заливка
plt.clabel(c)
plt.imshow(Z, extent=[0,5,0,5], cmap='RdGy', interpolation='gaussian', origin='lower', aspect='equal') # можем совмещать с contour
# aspect - соотношение сторон
# origin - переворачивает оси для Z
plt.colorbar()
plt.show()





