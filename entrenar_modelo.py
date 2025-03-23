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
