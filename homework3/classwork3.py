# конспект лекции 3
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ


import numpy as np
import pandas as pd


# pandas - расширение numpy (структурированные массивы). Строки и столбцы индексируются метками, а не только числовыми значениями

# основные структуры: Series, DataFrame, Index

## Series

data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print(type(data))

# внутренние данные data
print(data.values) # значения
print(type(data.values)) # numpy массив
print(data.index) # индексы от 0 до последнего 
print(type(data.index)) # pandas range index



data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data[0])
print(data[1:3]) # срез



# ОПРЕДЕЛЕНИЕ ИНДЕКСА В PANDAS
# как отдельный объект
data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
print(data)
print(data['a'])
print(data['b':'d']) # ТУТ ВКЛЮЧИТЕЛЬНО - отличие

print(type(data.index)) # теперь просто Index, а не RangeIndex



# можем задавать индексы произвольных типов
data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1, 10, 7, 'd'])
print(data)

print(data[1])
print(data[10:'d'])



# другой вид задания Series

population_dict = {
    'city_1' : 1001,
    'city_2' : 1002,
    'city_3' : 1003,
    'city_4' : 1004,
    'city_5' : 1005,
}

population = pd.Series(population_dict)
print(population)
print(population['city_4'])
print(population['city_4':'city_5'])



# для создания Series можно использовать 
# - списки питона или массивы numpy
# - скалярные значения
# - словари

# Q1 - привести различные способы создания объектов Series



## DataFrame - двумерный массив с явно определенными индексами. Или последовательность "согласованных объектов" Series

population_dict = {
    'city_1' : 1001,
    'city_2' : 1002,
    'city_3' : 1003,
    'city_4' : 1004,
    'city_5' : 1005,
}

# согласованность - ключи
area_dict = {
    'city_1' : 901,
    'city_2' : 11,
    'city_3' : 103,
    'city_4' : 105,
    'city_5' : 10011,
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)

print(population)
print(area)

states = pd.DataFrame({
    'population1': population,
    'area1': area
})

print(states)



# внутренние элементы - практически такие же

print(states.values)
print(type(states.values))
print(states.index)
print(type(states.index))
print(states.columns)
print(type(states.columns))



# пример работы с DataFrame
print(states['area1'])



# DataFrame. Способы создания
# - через объекты Series
# - списки словарей
# - словари объектов Series
# - двумерный массив numpy
# - структурированный массив numpy

# Q2 - привести различные способы задания объектов типа DataFrame



## Index - способ организации ссылки на данные объектов Series и DataFrame
# Index - неизменяем (по сути кортеж), упорядочен, является мультимножеством (могут быть повторяющиеся значения)!!! 

ind = pd.Index([2, 3, 5, 7, 11])
print(ind[1])
print(ind[::2])

# ind[1] = 5 - не получится 



# Index - следует соглашениям объекта set (python)

indA = pd.Index([1,2,3,4,5])
indB = pd.Index([2,3,4,5,6])

print(indA.intersection(indB)) # пересечение и тд


### Выборка данных из Series
## как словарь

data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])

print('a' in data)
print('z' in data)

print(data.keys())

print(list(data.items())) # превращение в список

print(data)

data['a'] = 100
data['z'] = 1000
print(data)

## как одномерный массив

data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])

print(data['a':'c'])
print(data[0:2])
print(data[data > 0.5])
print(data[(data > 0.5) & (data < 1)])
print(data[['a','d']]) # векторизованная



## особенность
## атрибуты-индексаторы!!!
data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1, 3, 10, 15])

print(data[1]) # как быть? 

print(data.loc[1])
print(data.iloc[1]) # индексы внутри индекса


### Выборка данных из DataFrame
## как словарь

pop = pd.Series({
    'city_1' : 1001,
    'city_2' : 1002,
    'city_3' : 1003,
    'city_4' : 1004,
    'city_5' : 1005,
})

# согласованность - ключи
area = pd.Series({
    'city_1' : 901,
    'city_2' : 11,
    'city_3' : 103,
    'city_4' : 105,
    'city_5' : 10011,
})


data = pd.DataFrame({
    'area1': area,
    'pop1': pop,
    'pop' : pop
})

print(data)

print(data['area1'])
print(data.area1) # то же самое

print(data.pop1 is data['pop1'])
print(data.pop is data['pop'])


data['new'] = data['area1'] # увелич размерности

print(data)

data['new'] = data['area1'] / data['area1'] # универсальные функции

print(data)



## как двумерный массив

pop = pd.Series({
    'city_1' : 1001,
    'city_2' : 1002,
    'city_3' : 1003,
    'city_4' : 1004,
    'city_5' : 1005,
})

# согласованность - ключи
area = pd.Series({
    'city_1' : 901,
    'city_2' : 11,
    'city_3' : 103,
    'city_4' : 105,
    'city_5' : 10011,
})


data = pd.DataFrame({
    'area1': area,
    'pop1': pop
})

print(data)

print(data.values)

print(data.T) # .T -транспонирование

print(data['area1'])

print(data.values[0]) # получаем строчку (как в numpy)

print(data.values[0:3]) 



# атрибуты-индексаторы

data = pd.DataFrame({
    'area1': area,
    'pop1': pop, 
    'pop': pop
})

print(data)

print(data.iloc[:3, 1:2]) # уже двумерный массив
print(data.loc[:'city_4', 'pop1':'pop'])

print(data.loc[data['pop'] > 1002, ['area1', 'pop']])

data.iloc[0,2] = 999999

print(data)



# про универсальные функции

rng = np.random.default_rng()
s = pd.Series(rng.integers(0,10,4))

print(s)

print(np.exp(s))



# если не согласованы

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

# хотим объединить - получаются добавления с NaN

data = pd.DataFrame({'area1':area, 'pop1':pop})
print(data)

# Q3 - объедините два объекта Series с неодинаковыми множествами ключей (индесами) так, чтобы вместо NaN было установлено значение 1



# объединение DataFrame 

dfA = pd.DataFrame(rng.integers(0,10, (2,2)), columns = ['a', 'b'])
dfB = pd.DataFrame(rng.integers(0,10, (3,3)), columns = ['a', 'b', 'c'])

print(dfA)
print(dfB)

print(dfA + dfB) # Nan - тк Nan + 8 = Nan и тд



# транслирование

rng = np.random.default_rng(1)

A = rng.integers(0, 10, (3,4))
print(A)

print(A[0])

print(A - A[0]) # тут транслирование

# в pandas похоже

df = pd.DataFrame(A, columns = ['a', 'b', 'c', 'd'])
print(df)

print(df.iloc[0])

print(df - df.iloc[0]) # тоже действует транслирование

# если взять часть строчки

print(df.iloc[0, ::2])

print(df - df.iloc[0, ::2]) # ЗДЕСЬ СОГЛАСОВАНИЕ ИНДЕКСОВ ПРОИСХОДИТ

# Q4 - Переписать пример с транслированием для DataFrame так, чтобы вычитание происходило по столбцам



# Nan = not a number

#NA-значения: Nan, null, какой-нибудь индикатор (н-р, -9999999)

## Pandas. Два способа хранения отсутсвующих значений
# индикаторы Nan, None
# обозначение через null

# None - объект (использование - накладные расходы). Не работает с sum, min и другими агрег операторами
# пример:

val1 = np.array([1,2,3])
print(val1.sum())

#val1 = np.array([1,None,2,3])
#print(val1.sum()) # не работает

val1 = np.array([1,np.nan,2,3])
print(val1)
print(val1.sum()) # выдает nan

val1 = np.array([1,np.nan,2,3])
print(val1)
print(np.nansum(val1)) # тут уже sum проходит

# РАБОТАЕТ ТАК ТОЛЬКО С ЧИСЛАМИ, СТРОКИ - НЕТ



# NA в Pandas

x = pd.Series(range(10), dtype=int)
print(x)

x[0] = None # ведут себя одинаково
x[1] = np.nan

print(x)

# НО:

x = pd.Series(['a', 'b', 'c'])
print(x)

x[0] = None # тут уже нет - объект как объект
x[1] = np.nan

print(x)



## в Pandas есть обобщающий NA-элемент

x2 = pd.Series([1,2,3,np.nan, None, pd.NA])

print(x2)

x3 = pd.Series([1,2,3,np.nan, None, pd.NA], dtype='Int32')

print(x3) # привели к одному типу - все элементы становятся одинаковыми



# Как рабоать с <NA>

print(x3.isnull()) # получаем bool массив
print(x3[x3.isnull()])
print(x3[x3.notnull()])

print(x3.dropna()) # сама выбросит



# По аналогии <NA> в DataFrame

df = pd.DataFrame(
    [
        [1,2,3,np.nan, None, pd.NA],
        [1,2,3, None,5,6],
        [1,np.nan,3,None,np.nan,6],
    ]

)

print(df)
print(df.dropna())
print('dddddd')
print(df.dropna(axis=0)) # по какой размерности
print(df.dropna(axis=1))

# ещё другие аргументы
# how 
# - all - все значения NA
# - any - хотя бы одно значение
# - thresh = x, останется, если присутсвует МИНИМУМ x НЕПУСТЫХ значений
print(df.dropna(axis=1, how='all'))
print(df.dropna(axis=1, how='any'))
print(df.dropna(axis=1, thresh=2))

# Q5 - на примере объектов DataFrame продемонстрируйте использование методов ffill() и bfill()





