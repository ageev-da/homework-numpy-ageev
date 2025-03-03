# Домашнее задание 8
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# График 1
x = np.array([1, 5, 10, 15, 20])
y1 = np.array([1, 7, 3, 5, 11])
y2 = np.array([4, 3, 1, 8, 12])

fig, ax = plt.subplots()
ax.plot(x, y1, '-o', color = 'red', label='line 1')
ax.plot(x, y2, '--o', color = 'green', label='line 1')
plt.legend()

# График 2
plt.figure()
x = np.linspace(1,5,5)
y1 = np.array([1, 7, 6, 3, 5])
y2 = np.array([9, 4, 2, 4, 9])
y3 = np.array([-7, -4, 2, -4, -7])
grid = plt.GridSpec(2, 4)
plt.subplot(grid[0, 0:])
plt.plot(x, y1)
plt.subplot(grid[1, 0:2])
plt.plot(x, y2)
plt.subplot(grid[1, 2:])
plt.plot(x, y3)

# График 3
plt.figure()
fig, ax = plt.subplots()
x = np.arange(-5,6,1)
y = x**2
plt.plot(x, y)
ax.annotate('min', xy=(0, 0), xytext=(-0.3,10),
            arrowprops=dict(facecolor='green'))

# График 4
plt.figure()
z = np.random.uniform(low=0.0, high=10.0, size=(7, 7))
plt.imshow(z, cmap='viridis')
plt.colorbar()

# График 5
plt.figure()
x = np.linspace(0,5,1000)
fig, ax = plt.subplots()
y = np.cos(np.pi*x)
ax.plot(x, y, color = 'red')
plt.fill_between(x, y, where=(x >= 0), color="blue", alpha=0.3)

# График 6
plt.figure()
fig, ax = plt.subplots()
x1 = 0
for i in range(1,7,2):
    x2 = i
    x = np.linspace(x1,x2,1000)
    y = np.cos(np.pi*x)
    val = y >= -0.5
    plt.plot(x[val], y[val], color="blue")
    x1 = i
ax.set_ylim(-1,1)

# График 7
plt.figure(figsize=(14,4))
x = np.arange(0,7,1)
x1=[0,1,0.5]
for i in range(3):
    xl = [0]
    x2 = x1[i]
    y1 = 0
    yl = []
    if i == 0:
        xl = []
    while x2 < 6.1:
        xl += [x2,x2]
        yl += [y1,y1]
        x2 += 1
        y1 += 1
    if i == 0:
        xl.pop()
        yl.pop(0)
    if i == 1:
        yl.append(6)
    if i == 2:
        xl.append(6)
        yl += [6,6]
    print(xl)
    print(yl)
    plt.subplot(1,3,i+1)
    plt.plot(x,x,'o', color = 'green')
    plt.plot(xl,yl, color = 'green')
    plt.grid(True)

# График 8
plt.figure()
fig, ax = plt.subplots()
x = np.arange(0,11,1)
y1 = x*(10-x)*4/25
y2 = x*(10-x)*14/25
y3 = x*(14-x)*26/49
plt.plot(x,y1,label="y1")
plt.plot(x,y2,label="y2")
plt.plot(x,y3,label="y3")
plt.fill_between(x, y1, color="blue")
plt.fill_between(x, y1, y2, color="orange")
plt.fill_between(x, y2, y3, color="green")
plt.legend()

# График 9
plt.figure()
labels = ['Toyota','Ford','Jaguar','AUDI','BMV']
sizes = [12, 16, 23, 13, 36]
colors = ['orange', 'blue', 'purple', 'red','green']
explode = (0, 0, 0, 0,0.1)
sizes_reversed = sizes[::-1] # решил так сделать, чтобы попрактиковаться
labels_reversed = labels[::-1]
colors_reversed = colors[::-1]
explode_reversed = explode[::-1]
plt.pie(sizes_reversed, labels=labels_reversed, colors=colors_reversed, explode=explode_reversed, startangle=90)

# График 10
plt.figure()
plt.pie(sizes_reversed, labels=labels_reversed, colors=colors_reversed, startangle=90)
centre_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)




