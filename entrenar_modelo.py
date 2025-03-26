
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Ruta del dataset
DATASET_PATH = r'C:/Users/billy/OneDrive/Escritorio/fer2013'

# Carga de datos con prefetching
train_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(48, 48),
    batch_size=32
)

val_ds = keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(48, 48),
    batch_size=32
)

# Revisión de balance de clases
class_names = train_ds.class_names
class_counts = {class_name: 0 for class_name in class_names}
for _, labels in train_ds:
    for label in labels.numpy():
        class_counts[class_names[label]] += 1
print("Distribución de clases:", class_counts)

# Preprocesamiento
def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalización implícita [0, 1]
    return image, label

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# Aumentación de datos
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])

# Construcción del modelo con regularización L2 y más dropout
model = keras.Sequential([
    layers.Input(shape=(48, 48, 3)),
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(2, activation='softmax')  # Dos clases
])

# Compilación
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Entrenamiento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

# Guardar el modelo
model.save('modelo_emociones_mejorado.keras')
print("Modelo entrenado y guardado como 'modelo_emociones_mejorado.keras'")

# Visualización de resultados
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Rutas
train_dir = r'C:\Users\billy\detector_emociones\fer2013\train'
modelo_guardado = r'C:\Users\billy\detector_emociones\modelo_entrenado\modelo_emociones.h5'

# Preprocesamiento de imágenes
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Modelo CNN
modelo = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 emociones
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento
historial = modelo.fit(train_generator, validation_data=val_generator, epochs=20)

# Guardar el modelo
modelo.save(modelo_guardado)
print(f"Modelo guardado en: {modelo_guardado}")

----------------------------------------------------------------------------------------------------------------------------
CODIGO DE ENTRENAMIENTO USADA :

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

# Rutas
train_dir = r'C:\Users\billy\detector_emociones\fer2013\train'
modelo_guardado = r'C:\Users\billy\detector_emociones\modelo_entrenado\modelo_emociones.keras'

# Preprocesamiento y Aumentación de datos
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)

# Modelo CNN optimizado
modelo = keras.Sequential([
    layers.Input(shape=(48, 48, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')  # 7 emociones
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Entrenamiento
historial = modelo.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping, reduce_lr]
)

# Guardar el modelo
modelo.save(modelo_guardado)
print(f"Modelo guardado en: {modelo_guardado}")

# Gráficas de entrenamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(historial.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión Validación')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.title('Precisión durante el Entrenamiento')

plt.subplot(1, 2, 2)
plt.plot(historial.history['loss'], label='Pérdida Entrenamiento')
plt.plot(historial.history['val_loss'], label='Pérdida Validación')
plt.legend()
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.title('Pérdida durante el Entrenamiento')

plt.show()

----------------------------------------------------------------------------------------------------------------------------
CODIGO DE DETECTAR_EMOCIONES :

import cv2
import tensorflow as tf
import numpy as np

# Cargar el modelo
modelo = tf.keras.models.load_model(r'C:\Users\billy\detector_emociones\modelo_entrenado\modelo_emociones.keras')

# Lista de emociones
emociones = ['Enojado', 'Disgustado', 'Temeroso', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']

# Inicializar la captura de video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bucle para detección en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar la imagen.")
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        rostro = gray[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (48, 48))
        rostro = np.expand_dims(rostro, axis=0)
        rostro = np.expand_dims(rostro, axis=-1)
        rostro = rostro / 255.0  # Normalización

        # Predicción de la emoción
        prediccion = modelo.predict(rostro, verbose=0)
        emocion_predicha = emociones[np.argmax(prediccion)]

        # Dibujar el rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Mostrar la emoción predicha en el frame
        cv2.putText(frame, emocion_predicha, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Mostrar el frame con la emoción detectada
    cv2.imshow("Detección de Emociones", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()


