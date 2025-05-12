# Конспект лекции 15
# ВЫПОЛНИЛ: АГЕЕВ ДАНИИЛ ЭДУАРДОВИЧ

# # Нейронные сети
# - **Сверточные нейронные (конволюционные) нейронные сети (CNN)** - компьютерне зрение, классификация изображений
# - **Рекуррентные нейронные сети (RNN)** - распознавание текста, обработка естественного языка
# - **Генеративные состязательные сети** - создание художественных, музыкальных произведений
# - **Многослойный перцептрон (MLP)** - простейший тип нейронной сети

# ### Основные понятия
# 
# Многослойный перцептрон состоит из слоев нейронов
# 
# **Глубина** - количество слоев, **ширина** - количество нейронов в слое
# 
# Имеются **входной** слой, **средние (скрытые)** слои, **выходной** слой - один нейрон
# 
# Нейросети работают только с **вещественными** числами
# 
# **Полносвязный** слой - связь всех нейронов слоя со всеми нейронами соседних слоев
# 
# На каждом нейроне присутствует число - **смещение**, у каждой связи есть число - **вес**
# 
# После входного слоя, на связях каждого слоя с последующим имеется **функция активации**. Обычно в качестве нее выступает **функция выпрямленных линейный единиц (ReLU)**: f(x) = x, если x > 0; f(x) = 0, если x <= 0.

# ### Обучение нейросети
# Инициализация:
# - начальные значения весов - случайные небольшие числа
# - смещения принимаются равными нулю
# 
# На вход подаются тренировочные данные. Для выходных данных вычисляется функция потерь, находится ошибка. Запускается алгоритм обратного распространения ошибки. Происходит новая настройка весов и смещения. Затем запускаются новые тренировочные данные. Это происходит до тех пор, пока величина ошибка не станет удовлетворительной.
# 
# Основной составляющей является оптимизатор, который находит минимум. Это включает в себя: вычисление частных производных и градиентный спуск с учетом скорости обучения.

# ### Основные фреймворки
# Рассчитанные на обучение:
# - TensorFlow (keras)
# - PyTorch
# 
# Выходом фреймворка является модель. 
# 
# Рассчитанные на прогнозирование:
# - TensorFlow Lite - мобильное устройство
# - TensorFlow vs - браузер
# 
# ONNX - Open Neural Network Exchange - перенос модели между библиотеками

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import array_to_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions
import numpy as np
import matplotlib.pyplot as plt

img_path = './data/test/cat.jpg'
img = image.load_img(img_path, target_size=(224,224))

img_array = image.img_to_array(img)
print(img_array.shape)

img_batch = np.expand_dims(img_array, axis=0)
print(img_batch.shape)

# Нормализация
img_processed = preprocess_input(img_batch)

model = ResNet50()

prediction = model.predict(img_processed)

print(decode_predictions(prediction, top=5)[0])
plt.imshow(img) 

# ### Перенос обучения
# Сверточные нейронные сети состоят из набора слоев, данные проходят через слои.
# При этом происходит выбор признаков.
# 
# Первая часть сети состоит из вннутренних, **сверточных**, слоев, которые преобразуют пиксели в признаки - происходит выбор признаков, что уменьшает размер данных.
# 
# Второй частью являются **полносвязные** слои - каждый нейрон уровня связан со всеми нейронами следующего.
# Данные нейроны принимают решение (например, классификация).
# 
# Выход сети называется **вершиной**. При движении к ней знание становится все более специализированным.
# Сначала идут слои с обобщенным знанием, далее идут слои с узкоспециализированным знанием.
# 
# Последовательность слоев как последовательность фильтров. Каждый слой реагирует только на свои признаки.
# 
# Первые слои сетей пропускают дальше, если распознают простые формы (например, края). 
# Дальше распознаюатся более узкоспециальные детали. 
# Таким образом, первые слои могут быть использованы повторны. 
# Чем дальше, тем меньше возможность переиспользования.
# 
# Идея **переноса обучения** - использование уже настроенной сверточной части для своей задачи. 
# Границу для неизменяемых слоев можно подвинуть (тонкая настройка), но есть вероятность возникновения переобучение.

# ### Последовательность действий
# - Организация данных (обучающие и проверочные данные, часто добавляют контрольные)
# - Построение пайплайна подготовки
# - Аугментация данных, обогащение набора
# - Определение модели, заморозка коэффициентов (алгоритм оптимизатора, метрика оценки)
# - Обучение модели -> итерации -> пока метрика не станет приемлемой
# - Сохранение модели

from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import (Input, Flatten, Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import math

TRAIN_DATA_DIR = './data/train_data/'
VALIDATION_DATA_DIR = './data/val_data/'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
# Многоклассовая классификация
NUM_CLASSES = 2
IMG_WIDTH, IMG_HIGHT = 224, 224
# Размер пакет определяет, сколько изображений модель будет получать за один раз
BATCH_SIZE = 64

# Нормализация и аугментация
train_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2
)

val_datagen = image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=12345,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VALIDATION_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical'
)


def model_maker():
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False

    input = Input(shape=(IMG_WIDTH, IMG_HIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    prediction = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=prediction)

model = model_maker()
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(),
    metrics=['acc']
)

num_steps = math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE)

model.fit(
    train_generator,
    steps_per_epoch=num_steps,
    epochs=10, # Шаг обучения, в котором просматривается весь набор данных
    validation_data=val_generator,
    validation_steps=num_steps
)

print(val_generator.class_indices)

model.save('./data/model.keras')

from keras.models import load_model
model = load_model('./data/model.keras')

# img_path = './data/val_data/cats/100.jpg'
img_path = './data/val_data/dogs/61.jpg'
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# Нормализация
img_processed = preprocess_input(img_batch)

prediction = model.predict(img_processed)
print(prediction)



