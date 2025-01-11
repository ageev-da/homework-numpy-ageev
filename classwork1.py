import numpy as np 
import sys
import array

# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

# Типы данных Python
# тип переменной меняется в runtime
x = 1
print(type(x))
print(sys.getsizeof(x)) #размер

x = 'hello'
print(type(x))

x = True
print(type(x))

# в x хранится некот структура данных, кот занимает место

# список

l1 = list([])
print(sys.getsizeof(l1))

l2 = list([1,2,3])
print(sys.getsizeof(l2))

l3 = list([1,'2',True])
print(l3)
print(sys.getsizeof(l3))

# такое свойство приводит к замедлению процесса


# массив
# не позволяет разные типы в одном месте
a1 = array.array('i', [1,2,3]) # i - int
print(type(a1))
print(sys.getsizeof(a1))


# ЗАДАНИЕ 1 
# 1. Какие ещё существуют коды типов?
# 2. Напишите код, подобный приведенному выше, но с другим типом


# в numpy эффективно хранятся данные и работа с ними 

a = np.array([1,2,3,4,5])
print(type(a), a)

# np не позволяет использовать одновременно разные типы
# "повышающее" приведение типов
a = np.array([1.23,2,3,4,5])
print(type(a), a)

a = np.array([1.23,2,3,4,5], dtype=int)
print(type(a), a)


# многомерные массивы

a = np.array([range(i, i+3) for i in [2,4,6]])
print(a, type(a))


# полезные функции
# нулевой массив
a = np.zeros(10, dtype=int)
print(a, type(a))

# массив из 1
print(np.ones((3,5), dtype=float))

# массив определенных значений
print(np.full((4,5), 3.1415))

# последовательность чисел
print(np.arange(0, 20, 2))

# единичная матрица определенного размера
print(np.eye(4))


# ЗАДАНИЕ 2
# 3. Напишите код для создания массива с 5 значениями, располагающимися через равные интервалы в диапазоне от 0 до 1
# 4. Напишите код для создания массива с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1
# 5. Напишите код для создания массива с 5 нормально распределенными случайными значениями с мат ожиданием = 0 и дисперсией 1
# 6. Напишите код для создания массива с 5 случайными целыми числами в от [0,10)


# Массивы
# каждый сид - уникальная последовательность
np.random.seed(1)

x1 = np.random.randint(10, size=3)
x2 = np.random.randint(10, size=(3,2))
x3 = np.random.randint(10, size=(3,2,2))
#print(x1)
#print(x2)
#print(x3)

# Свойства массивов
# 1 Число размерностей - ndim
# 2 Размер каждой размерности - shape
# 3 общая - size

print(x1.ndim, x1.shape, x1.size)
print(x2.ndim, x2.shape, x2.size)
print(x3.ndim, x3.shape, x3.size)


# доступы к элементу массива
# Индекс (с 0 для int)
a = np.array([1,2,3,4,5])
print(a[0])

# в обратом с -1
print(a[-2])

a[1] = 20

print(a)


# доступ по индексу для многомерного массива
a = np.array([[1,2], [3,4]])
print(a)

print(a[0,0])
print(a[-1,-1])

a[1,0] = 100
print(a)


# в np тип данных фиксированный
a = np.array([1,2,3,4])
b = np.array([1.0,2,3,4])

print(a)
print(b)

a[0] = 10
print(a)

# уже не повышает!!!
a[0] = 10.123 # приводит к int
print(a) 


# Сред - некоторый подмассив [start:finish:step]
# по умолчанию [0:shape(размер измерения):1]

a = np.array([1,2,3,4,5,6])
print(a[0:3:1])

# то же самое
print(a[:3])

# с индексом больше 2
print(a[3:])

# все элементы из середины
print(a[1:5])
print(a[1:-1])

#каждый второй
print(a[1:6:2])
# проще
print(a[1::2])

# весь массив
print(a[::1])

# в обратном порядке
print(a[::-1])

# СРЕЗ - НЕ КОПИЯ, А КУСОЧЕК
a = np.array([1,2,3,4,5,6])

b = a[:3]
print(b)

b[0] = 100
print(a)


# ЗАДАНИЕ 3
# 7. Написать код для создания срезов массива 3 строки на 4 стобца
# - две строки и три столбца
# - первые три строки и второй столбец
# - все строки и столбцы в обратном порядке
# - второй столбец
# - третья строка
# 8. Продемонстрируйте, как сделать срез-копию


a = np.arange(1,13)
print(a)

# смена размерности

print(a.reshape(2,6))
print(a.reshape(3,4))
# размер исходного = созданному


# ЗАДАНИЕ 4
# 9. Продемонстрируйте использование newaxis для получения вектора-столбца и вектора-строки


# склеивание массивов
# гориз склейка 1
x = np.array([1,2,3])
y = np.array([4,5])
z = np.array([6])

print(np.concatenate([x,y,z]))


# другой случай - вертикальная склейка
x = np.array([1,2,3])
y = np.array([4,5,6])

r1 = np.vstack([x,y])
print(r1)

# гориз склейка 1 - размерности должны соотв
print(np.hstack([r1,r1]))


# ЗАДАНИЕ 5
# 10. Разберитесь, как работает dstack
# 11. Разберитесь, как работают методы split, vsplit, dsplit


### Вычисления с массивами

# Векторизированная операция - независимо к каждому элементу массива

x = np.arange(10)
print(x)

# операторы *, + и ид развертываются в более сложные конструкции
print(x*2 + 1)

# Универсальные функции
print(np.add(np.multiply(x,2), 1)) # то же самое, что и выше

# - - вычитание 
# - - перед числом
# / // ** % 


# ЗАДАНИЕ 6
# 12. Привести пример использования всех универсальных функций, которые привели


# ещё есть: np.abs, sin/cos/tan, exp, log и тд

x = np.arange(5)

# доп параметры
y = np.empty(5)
print(np.multiply(x,10, out=y))
print(y)

# запись в сред
y = np.zeros(10)
print(np.multiply(x,10, out=y[::2]))
print(y)


# свертка массива

x = np.arange(1,5)
print(x)

#свертка к 1 элементу (сумма)
print(np.add.reduce(x))

# нараст. сумма
print(np.add.accumulate(x))


# Векторные произведения

x = np.arange(1,10)
print(np.add.outer(x,x))
# по сути таблица сложения

print(np.multiply.outer(x,x))






