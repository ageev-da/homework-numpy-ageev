# Домашняя работа 4
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ



import numpy as np 
import pandas as pd



# Задание 1. Разобраться как использовать мультииндексные ключи в данном примере
index = [
    ('city_1',2010),
    ('city_1',2020),
    ('city_2',2010),
    ('city_2',2020),
    ('city_3',2010),
    ('city_3',2020)
]

population = [
    101,
    1010,
    201,
    2010,
    102,
    1020,
]

pop = pd.Series(population, index=index)
index1 = pd.MultiIndex.from_tuples(index)

pop_df = pd.DataFrame(
    {
        'total':pop,
        'something':[
                11,
                12,
                13,
                14,
                15,
                16
        ]
    },
    index=index1
)

print('Исходный DF')
print(pop_df)

print('DF1')
pop_df_1 = pop_df.loc[['city_1'],'something'] # возвращает Series, тк выбирается только один столбец
print(pop_df_1)
pop_df_1 = pop_df.loc[['city_1'],['something']] # возвращает DF
print(pop_df_1)

print('DF2')
pop_df_2 = pop_df.loc[['city_1', 'city_3'],['total', 'something']]
print(pop_df_2)

print('DF3')
pop_df_3 = pop_df.loc[['city_1', 'city_3'],'something'] # Возвращает Series
print(pop_df_3)
pop_df_3 = pop_df.loc[['city_1', 'city_3'],['something']] # Возвращает DF
print(pop_df_3)



# Задание 2. Из получившихся данных выбрать данные по 
# - 2020 году (для всех столбцов)
# - job_1 (для всех строк)
# - для city_1 и job_2 
index = pd.MultiIndex.from_product(
    [
        ['city_1','city_2'],
        [2010,2020]
    ],
    names=['city','year']
)

columns = pd.MultiIndex.from_product(
    [
        ['person_1','person_2','person_3'],
        ['job_1','job_2']
    ],
    names=['worker','job']
)

rng = np.random.default_rng(1)
data = rng.random((4, 6))

data_df = pd.DataFrame(data, index=index, columns=columns)
print('Исходный DF')
print(data_df)

print('Данные по 2020 году (для всех столбцов)')
print(data_df.loc[(slice(None), 2020), :]) # slice(None) - выбрать все элементы по этому измерению индекса (аналог : в iloc)

print('Данные для job_1 (для всех строк)')
print(data_df.loc[:, (slice(None),'job_1')])

print('Данные для city_1 и job_2')
print(data_df.loc[('city_1', slice(None)), (slice(None),'job_2')])



# Задание 3. Взять за основу DataFrame из задания 2
# Выполнить запрос на получение следующих данных
# - все данные по person_1 и person_3
# - все данные по первому городу и первым двум person-ам (с использование срезов)
# Приведите пример (самостоятельно) с использованием pd.IndexSlice

print('Все данные по person_1 и person_3')
print(data_df.loc[:, (['person_1', 'person_3'], slice(None))])

# IndexSlice позволяет избавиться от slice(None), что делает код читабельнее
idx = pd.IndexSlice
print(data_df.loc[:, idx[['person_1','person_3'],:]])

print('Все данные по первому городу и первым двум person-ам (с использование срезов)')
print(data_df.loc[('city_1', slice(None)), (slice('person_1', 'person_2'), slice(None))])
print(data_df.loc[idx['city_1',:], idx['person_1':'person_2',:]]) # с использованием IndexSlice


# Задание 4. Привести пример использования inner и outer джойнов для DF (для Series нормально продемонстрировать не получилось)

df1 = pd.DataFrame([['x', 10, 100], ['y', 20, 200], ['z', 30, 300]], columns=['col_1', 'col_2', 'col_4'])
print('DF1')
print(df1)

df2 = pd.DataFrame([['a', 40, 400], ['b', 50, 500], ['c', 60, 600]], columns=['col_1', 'col_2', 'col_3'])
print('DF2')
print(df2)

# outer - объединяет все столбцы из обоих DF, заполняя отсутствующие значения NaN
print('outer join')
print(pd.concat([df1, df2], join='outer'))

# inner - объединяет только те столбцы, которые присутствуют в обоих DF
print('inner join')
print(pd.concat([df1, df2], join='inner'))





