
# Конспект лекции 2
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ


import numpy as np
import matplotlib.pyplot as plt



### суммирование значений (и другие агрегатные функции)

rng = np.random.default_rng(1)
s = rng.random(50)

print(s)
print(sum(s))
print(np.sum(s))

a = np.array([
    [1,2,3,4,5],
    [6,7,8,9,10]
])

print(np.sum(a)) # сумма элементов
print(np.sum(a,axis=0)) # axis - измер, получ некот свернутое (сумма по столбцам)
print(np.sum(a,axis=1)) # по строкам

# функция минимума
print(np.min(a))
print(np.min(a,axis=0)) 
print(np.min(a,axis=1))

# можем применять функцию непостредственно к массиву
print(a.min())
print(a.min(0))
print(a.min(1))

# безопасная версия запуска функций (при возможном наличии NaN - not a number)
# функция минимума
print(np.nanmin(a))
print(np.nanmin(a,axis=0)) 
print(np.nanmin(a,axis=1))


### транслирование (broadcasting)
# набор правил, которые позволяют осущевстлять бинарные операции с массивами разных форм и размеров

a = np.array([0,1,2])
b = np.array([5,5,5])

print(a + b)
print(a + 5) # сложение двух массивов разных форм, в процессе бродкастинга 5 расширяется (транслируется) в массив [5,5,5]
# т.e. подстраивается под размер массива а

a = np.array([[0,1,2], [3,4,5]])
print(a + 5)

# трансляция обоих массивов
a = np.array([0,1,2])
b = np.array([[0],[1],[2]])
print(a + b)

# ПРАВИЛА
## 1. Если размерности массивов отличаются, то форма массива с меньшей размерностью дополняется 1 с левой стороны

a = np.array([[0,1,2], [3,4,5]])
b = np.array([5])

print(a.ndim, a.shape) # ndim - кол-во измерений, shape - размерности
print(b.ndim, b.shape)

# a (2,3) -> (2,3)
# b (1,0) -> (1,1)

print(a + b)

## 2. Если формы массивов не совпадают (shape) в каком-то измерении, то если у массива форма равна 1, то он растягивается до соотв-я формы
## второго массива: (1,1) -> (2,3)

## 3. Если после применения этих правил в каком-либо измерении размеры отличаются и ни один из них не равен 1, то генерируется ошибка

# ещё примеры:

a = np.ones((2,3))
b = np.arange(3)

print(a)
print(a.ndim, a.shape) # ndim - кол-во измерений, shape - размерности

print(b)
print(b.ndim, b.shape)

# (2,3) -> (2,3) -> (2,3)
# (3,) -> (1,3) (ПР1) -> (2,3) (ПР2)

c = a + b
print(c)
print(c.ndim, c.shape)


a = np.arange(3).reshape((3,1))
b = np.arange(3)

print(a)
print(b)

print(a.ndim, a.shape)
print(b.ndim, b.shape)

# (3,1) -> (3,1) -> (3,3)
# (3,) -> (1,3) -> (3,3)

# [ 0 0 0 ]   [0 1 2]
# [ 1 1 1 ] + [0 1 2]
# [ 2 2 2 ]   [0 1 2]

c = a + b

print(c, c.shape)


# пример, когда не работает 

a = np.ones((3,2))
b = np.arange(3)

# 2 (3,2) -> (3,2) -> (3,2) ПР3 - генерация ошибки
# 1 (3,) -> (1,3) -> (3,3)

#c = a + b

# Q1: Что надо изменить в последнем примере, чтобы он заработал без ошибок



# центрирование по столбцам 

X = np.array([
    [1,2,3,4,5,6,7,8,9],
    [9,8,7,6,5,4,3,2,1]
])

Xmean0 = X.mean(0) # ср знач по строке
print(Xmean0)

Xcenter0 = X - Xmean0
print(Xcenter0)


# центрирование по строке

Xmean1 = X.mean(1)
print(Xmean1)

Xmean1 = Xmean1[:, np.newaxis] #тк просто не получится

Xcenter1 = X - Xmean1
print(Xcenter1)



x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]

z = np.sin(x)**3 + np.cos(20+y*x) * np.sin(y)
print(z.shape)

plt.imshow(z)
plt.colorbar()
plt.show()



### сравнение
# какие из элементов массива удовлетворяют некоторым условиям

x = np.array([1,2,3,4,5])
y = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(x < 3) # < - универсальная функция -> на выходе массив bool удовл или не удовл
print(np.less(x,3))

# False = 0
# True = 1

print(np.sum(x < 4)) # количество элементов
print(np.sum(y < 4, axis=0))
print(np.sum(y < 4, axis=1))

# & (И), | (или), ^, ~

# Q2: Пример для y. Вычислить количество элементов (по обоим размерностям), значения которых больше 3 и меньше 9



# Маски - булевые массивы

x = np.array([1,2,3,4,5])
y = print(x < 3)

print(x[x < 3])



print(bin(42))
print(bin(59))
print(bin(42 & 59))



### векторизация индекса
x = np.array([0,1,2,3,4,5,6,7,8,9])

index = [1,5,7]

print(x[index])

index = [[1,5,7], [2,4,8]]

print(x[index]) # результат отражает размерность массива индексов, а не как выглядел изначальный массив

## форма результата отражает форму массива индексов, а не форму исходного массива



x = np.arange(12).reshape((3,4))

print(x)
print(x[2])
print(x[2, [2,0,1]]) # берем вторую строку и выбираем конкретные элементы
print(x[1:, [2,0,1]]) 



# с помощью векторизации можем модифицировать части массива
x = np.arange(10)
i = np.array([2,1,8,4])

print(x)
x[i] = 999
print(x)



### сортировка массивов

x = [3,2,4,5,6,1,4,1,7,8]

print(sorted(x)) # чистый python
print(np.sort(x)) # numpy сортировка - работает быстрее на больших данных


### структурированные массивы

data = np.zeros(4, dtype = {
    'names':(
        'name', 'age'
    ),
    'formats':(
        'U10', 'i4'
    )
})

print(data.dtype)

name = ['name1', 'name2', 'name3', 'name4']
age = [10, 20, 30, 40]

data['name'] = name
data['age'] = age

print(data)


print(data['age'] > 20)
print(data[data['age'] > 20]['name'])



### массивы записей
# к их элементам можно обращаться не как к атрибутам, а по индексам

data_rec = data.view(np.recarray)
print(data_rec)
print(data_rec[0])
print(data_rec[-1].name)




