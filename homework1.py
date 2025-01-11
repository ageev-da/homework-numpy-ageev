import numpy as np
import sys
import array


# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ



# Задания 1 и 2. Коды типов для array и примеры кодов.

# 'u' - символ Unicode
unicode_array = array.array('u', ['a', 'b', 'c'])
print(unicode_array, type(unicode_array), sys.getsizeof(unicode_array))

# 'f' - число с плавающей точкой (float)
float_array = array.array('f', [1.0, 1.5, -1.0])
print(float_array, type(float_array), sys.getsizeof(float_array))

# 'd' - float с двойной точностью (double)
double_array = array.array('d', [1.0, 1.5, 3.141592653589793])
print(double_array, type(double_array), sys.getsizeof(double_array))

# 'i' - целые числа (int)
int_array = array.array('i', [1, -2, 3])
print(int_array, type(int_array), sys.getsizeof(int_array))

# 'h' - короткие целые числа (short - 2 байта)
short_array = array.array('h', [32000, -32000, 12345])
print(short_array, type(short_array), sys.getsizeof(short_array))

# 'l' - длинные целые числа (long)
long_array = array.array('l', [123456789, -123456789, 0])
print(long_array, type(long_array), sys.getsizeof(long_array))

# 'q' - длинных целыt числа (long long - 8 байт)
long_long_array = array.array('q', [9223372036854775807, -9223372036854775808, 0])
print(long_long_array, type(long_long_array), sys.getsizeof(long_long_array))

# также есть 'I', 'H', 'L', 'Q' - беззнаковые альтернативы 'i', 'h', 'l', 'q'



# Задание 3. Массив с 5 значениями, располагающимися через равные интервалы в диапазоне от 0 до 1.

# способ 1: np.linspace(start, stop, num), где num - кол-во точек, которое нужно создать
interval_array_1 = np.linspace(0, 1, 5)
print(interval_array_1)

# способ 2
interval_array_2 = np.arange(0, 1.1, 0.25)
print(interval_array_2)



# Задание 4. Массив с 5 равномерно распределенными случайными значениями в диапазоне от 0 до 1.

# np.random.uniform(low, high, size), где size - количество случайных чисел в массиве
unif_array = np.random.uniform(0, 1, 5)
print(unif_array)



# Задание 5. Массив с 5 нормально распределенными случайными значениями с мат. ожиданием = 0 и дисперсией 1

# np.random.normal(loc, scale, size), где loc - ср. знач (мат. ожидание 0)б scale - станд. откл (дисперсия 1)
normal_array = np.random.normal(0, 1, 5)
print(normal_array)



# Задание 6. Массив с 5 случайными числами в диапазоне [0,10)
rint_array = np.random.randint(0, 10, 5)
print(rint_array)



# Задание 7. Код для создания срезов массива 3 на 4

# исходный массив 3 на 4
x1 = np.random.randint(0, 10, size=(3, 4))
print(x1)

print("Первые 2 строки и 3 столбца")
print(x1[:2, :3])

print("Первые 3 строки и 2-й столбец")
print(x1[:3, 1:2])

print("Все строки и стоблцы в обратном порядке")
print(x1[::-1, ::-1])

print("Второй столбец")
print(x1[:,1])

print("Третья строка")
print(x1[2,:])



# Задание 8. Срез-копия

a = np.random.randint(0, 10, size=(3, 4))
print("Исходный а:")
print(a)

print("Срез-копия")
# в случае, если нужно получить срез-копию, не влияющую на исходный массив, то можно поступить так:
slice_copy = a[:2, :3].copy()

print("Срез-копия массива:")
print(slice_copy)
slice_copy[0, 0] = 11

print("Измененный slice_copy:")
print(slice_copy)

print("Оригинальный а:")
print(a)

print("Обычный срез")

# если нужен просто срез, то можно сделать так:
slice_view = a[:2, :3]

print("Срез массива slice_view:")
print(slice_view)
slice_view[0, 0] = 33

print("Измененный slice_view:")
print(slice_copy)

print("Оригинальный а после изменения в slice_view:")
print(a)


# Задание 9. Использование newaxis для получения вектора-столбца и вектора-строки

# исходный одномерный массив
v = np.random.randint(0, 10, 4)
print(v)

# вектор-столбец
col_v = v[:, np.newaxis]
print(col_v)

# вектор-строка
row_v = v[np.newaxis, :]
print(row_v)



# Задание 10. Метод dstack.

# данный метод необходим для объединения массивов вдоль третьей оси (глубины оси)
# при этом исходные массивы, которые необходимо объединить, должны иметь одинаковую размерность по первым двум осям

#исходные массивы (3,3)
a1 = np.random.randint(0, 10, (3, 3))
print(a1)
a2 = np.random.randint(0, 10, (3, 3))
print(a2)


# итог объединения - массив (3,3,2)
result = np.dstack((a1,a2))
print(result)
print(result.shape)



# Задание 11. Методы split, vsplit, hsplit, dsplit.

print("split")
# split - разделение массивов вдоль указанной оси
# np.split(array, indices_or_sections, axis=0), где array - исходный массив, 
# indices_or_sections - может быть целым числом (количество частей, на которое нужно разделить массив) 
# или списком индексов, которые определяют места разбиения, 
# axis - ось, вдоль которой разделение
a = np.arange(9).reshape(3, 3)
result = np.split(a, 3, axis=0)
print(result)

print("vsplit")
# vsplit - вдоль вертикальной оси
# np.vsplit(array, indices_or_sections)
a = np.arange(9).reshape(3, 3)
result = np.vsplit(a, 3)
print(result)

print("hsplit")
# hsplit - вдоль горизонтальной оси
# np.hsplit(array, indices_or_sections)
a = np.arange(9).reshape(3, 3)
result = np.hsplit(a, 3)
print(result)

print("dsplit")
# dsplit - вдоль глубины
# np.dsplit(array, indices_or_sections)
a = np.arange(27).reshape(3, 3, 3)
result = np.dsplit(a, 3)
print(result)



# Задание 12. Пример использовани универсальных функций

x = np.arange(10)

print("x: ", x)

print("x + 9: ", x + 9)
print(np.add(x, 9))

print("-x: ", -x) # перед числом
print(np.negative(x))

print("x - 9: ", x - 9) # вычитание
print(np.subtract(x, 9)) 

print("x * 3: ", x * 3)
print(np.multiply(x, 3))

print("x / 2: ", x / 2)
print(np.divide(x, 2)) 

print("x // 2: ", x // 2) # целочисленное деление (округление вниз)
print(np.floor_divide(x, 2))

print("x**2: ", x**2)
print(np.power(x, 2))

print("x%4:", x%4) # остаток от деления
print(np.mod(x, 4))





