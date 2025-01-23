# Домашняя работа 3
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ


import numpy as np
import pandas as pd



# Задание 1. Привести различные способы создания объектов типа Series

# 1) Список python
d1 = [1, 2, 3, 4]
ser1 = pd.Series(d1)
print('Series список python')
print(ser1)

# 2) массив numpy
d2 = np.array([1, 2, 3, 4])
ser2 = pd.Series(d2)
print('Series массив numpy')
print(ser2)

# 3) скалярные значения
d3 = 123
ser3 = pd.Series(d3, index = [0,1,2,3])
print('Series скалярные значения')
print(ser3)

# 4) словари
d4 = {
    "d1" : 1, 
    "d2" : 2, 
    "d3" : 3, 
    "d4" : 4 
}
ser4 = pd.Series(d4)
print('Series словари')
print(ser4)



# Задание 2. Привести различные способы создания объектов типа DataFrame

# 1) через объекты Series
ser1 = pd.Series([1,2,3,4])
ser2 = pd.Series([5,6,7,8])
data1 = pd.DataFrame([ser1,ser2])
print('DataFrame через объекты Series')
print(data1)

# 2) списки словарей
l1 = [{'a1': 1, 'b1': 2}, {'a1': -1, 'b1': -2}]
data2 = pd.DataFrame(l1)
print('DataFrame через списки словарей')
print(data2)

# 3) словари объектов Series
d1 = {'col1': ser1, 'col2': ser2}
data3 = pd.DataFrame(d1)
print('DataFrame через словари объектов Series')
print(data3)

# 4) двумерный массив numpy
a1 = np.array([[1,2,3], [4,5,6]])
data4 = pd.DataFrame(a1)
print('DataFrame через двумерный массив numpy')
print(data4)

# 5) структурированный массив numpy - массив, позволяющий хранить данные различных типов в одной и той же структури (по типу таблицы)
a2 = np.array([('city_1', 1009), ('city_2', 2009)], dtype=[('city', 'U10'), ('pop', 'i4')])
print(a2)
data5 = pd.DataFrame(a2)
print('DataFrame через структурированный массив numpy')
print(data5)



# Задание 3. Объединить два объекта Series с неодинаковыми множествами ключей (индексов) так, 
# чтобы вместо NaN было установлено значение 1
pop = pd.Series({
    'city_1' : 1001,
    'city_2' : 1002,
    'city_3' : 1003,
    'city_41' : 1004,
    'city_51' : 1005,
})

area = pd.Series({
    'city_1' : 901,
    'city_2' : 11,
    'city_3' : 103,
    'city_42' : 105,
    'city_52' : 10011,
})

# fillna() - позволяет устанавливать вместо Nan значение в ()

data = pd.DataFrame({'area1':area, 'pop1':pop}).fillna(1)
print(data)



# Задание 4. Переписать пример с транслирование для DataFrame так, чтобы вычитание происходило по столбцам
rng = np.random.default_rng(1)
A = rng.integers(0, 10, (3,4))
df = pd.DataFrame(A, columns = ['a', 'b', 'c', 'd'])
print('Исходный DF')
print(df)

print('DF вычитание по столбцам (способ 1)')
# обращаемся к 1 столбцу по индексу и добавляем ось для правильного транслирования
print(df - df.iloc[:,0].values[:,np.newaxis])

print('DF вычитание по столбцам (способ 2)')
# явно указываем ось в универс функции sub
print(df.sub(df.iloc[:, 0], axis=0))



# Задание 5. На примере объектов DataFrame продемонстрировать использование методов ffill() и bfill()
df = pd.DataFrame({
    'd1': [1, np.nan, 3, np.nan, 10],
    'd2': [np.nan, 4, np.nan, 5, 11],
    'd3': [6, np.nan, 7, np.nan, 12]
})
print('Исходный DF')
print(df)

# ffill() - заполнение NaN известными значениями в направлении сверху вниз
print('DF при использовании ffill()')
print(df.ffill())

# Можно указать, чтобы заполняло по строкам, а не по столбцам
print('DF при использовании ffill() по строкам')
print(df.ffill(axis = 1))

# bfill() - заполнение NaN известными значениями в направлении снизу вверх
print('DF при использовании bfill()')
print(df.bfill())

print('DF при использовании bfill() по строкам')
print(df.bfill(axis = 1))





