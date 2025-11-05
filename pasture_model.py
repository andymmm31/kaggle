import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
import os
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURACIÓN Y CONSTANTES ---

# Rutas de Kaggle (basado en tu captura de pantalla)
BASE_PATH = '/kaggle/input/csiro-biomass'
TRAIN_IMG_PATH = os.path.join(BASE_PATH, 'train')
TEST_IMG_PATH = os.path.join(BASE_PATH, 'test')
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV_PATH = os.path.join(BASE_PATH, 'test.csv')

# Constantes del modelo
IMG_SIZE = 300 # Tamaño para EfficientNetB3
BATCH_SIZE = 16 # Reducido para B3 y imágenes más grandes
LEARNING_RATE = 0.001
EPOCHS = 10 # Empezar con pocas para probar
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
N_TARGETS = len(TARGET_NAMES)

# Pesos para la métrica y la función de pérdida
# Dry_Green_g: 0.1, Dry_Dead_g: 0.1, Dry_Clover_g: 0.1, GDM_g: 0.2, Dry_Total_g: 0.5
LOSS_WEIGHTS = tf.constant([0.1, 0.1, 0.1, 0.2, 0.5])


# --- 2. PRE-PROCESAMIENTO DE DATOS ---

def load_and_pivot_data():
    """Carga train.csv y lo pivota a formato ancho, solo con targets."""
    df_train_long = pd.read_csv(TRAIN_CSV_PATH)

    # Pivoteamos los targets para tener una fila por imagen
    df_train_wide = df_train_long.pivot(
        index='image_path',
        columns='target_name',
        values='target'
    ).reset_index()

    # Quitar filas con targets faltantes
    df_train_wide = df_train_wide.dropna()

    print(f"Datos 'anchos' para entrenamiento: {df_train_wide.shape}")
    print(df_train_wide.head())

    return df_train_wide

# Cargar los datos
df_train = load_and_pivot_data()

# Dividir df_train en conjuntos de entrenamiento y validación
train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=42)

# Crear rutas de imagen completas
train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))
val_df['image_path'] = val_df['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))


def preprocess_image(image_path):
    """Carga y pre-procesa una imagen desde una ruta completa."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img) # Normalización de EfficientNet
    return img

def create_dataset(df):
    """Crea un tf.data.Dataset de (imagen, target) desde el dataframe."""

    image_paths = df['image_path'].values
    targets = df[TARGET_NAMES].values

    # Dataset de rutas de imágenes
    ds_img = tf.data.Dataset.from_tensor_slices(image_paths)
    ds_img = ds_img.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Dataset de targets
    ds_tgt = tf.data.Dataset.from_tensor_slices(targets)

    # Combinar en un dataset de (img, tgt)
    ds = tf.data.Dataset.zip((ds_img, ds_tgt))

    # --- Aumento de Datos ---
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.1), # Reducido de 0.2
        layers.RandomContrast(0.1),  # Reducido de 0.2
    ])

    # Aplicar aumento solo a la imagen
    ds = ds.map(lambda img, tgt: (data_augmentation(img), tgt), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Crear los datasets de entrenamiento y validación
train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)


# --- 3. MÉTRICA Y FUNCIÓN DE PÉRDIDA PERSONALIZADAS ---

class WeightedR2Score(tf.keras.metrics.Metric):
    """
    Calcula el R² score global ponderado.
    La fórmula es: 1 - (sum(w*(y_true-y_pred)^2) / sum(w*(y_true-y_mean)^2))
    donde y_mean es la media global de y_true para cada target.
    """
    def __init__(self, name='weighted_r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights = tf.constant([0.1, 0.1, 0.1, 0.2, 0.5], dtype=tf.float32)

        # Variables de estado para calcular SS_res
        self.sum_ss_res = self.add_weight(name='sum_ss_res', initializer='zeros')

        # Variables de estado para calcular SS_tot (usando la fórmula expandida)
        self.sum_y_true = self.add_weight(name='sum_y_true', shape=(N_TARGETS,), initializer='zeros')
        self.sum_y_true_sq = self.add_weight(name='sum_y_true_sq', shape=(N_TARGETS,), initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Actualizar sumas para SS_res
        ss_res = tf.reduce_sum(self.weights * tf.square(y_true - y_pred))
        self.sum_ss_res.assign_add(ss_res)

        # Actualizar sumas para SS_tot
        self.sum_y_true.assign_add(tf.reduce_sum(y_true, axis=0))
        self.sum_y_true_sq.assign_add(tf.reduce_sum(tf.square(y_true), axis=0))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        # Calcular SS_tot a partir de las sumas acumuladas
        # SS_tot = sum(w * (sum(y^2) - (sum(y))^2 / N))
        ss_tot_per_target = self.sum_y_true_sq - (tf.square(self.sum_y_true) / self.count)
        total_ss_tot = tf.reduce_sum(self.weights * ss_tot_per_target)

        # Evitar división por cero
        return 1.0 - (self.sum_ss_res / (total_ss_tot + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.sum_ss_res.assign(0.0)
        self.sum_y_true.assign(tf.zeros(shape=(N_TARGETS,)))
        self.sum_y_true_sq.assign(tf.zeros(shape=(N_TARGETS,)))
        self.count.assign(0.0)


def weighted_mse_loss(y_true, y_pred):
    """Calcula el Mean Squared Error ponderado."""
    # y_true y y_pred tendrán forma (batch_size, 5)

    # Calcular el error cuadrado por cada target
    error_sq = tf.square(y_true - y_pred)

    # Multiplicar por los pesos
    # (batch_size, 5) * (5,) -> (batch_size, 5)
    weighted_error_sq = error_sq * LOSS_WEIGHTS

    # Devolver la media de todos los errores ponderados en el batch
    return tf.reduce_mean(weighted_error_sq)


# --- 4. ARQUITECTURA DEL MODELO MULTIMODAL ---

def build_model():
    """Construye el modelo de solo visión (CNN) usando EfficientNetB3."""

    # --- Entrada de Imagen (CNN) ---
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')

    base_model = EfficientNetB3(
        include_top=False,
        weights=None,
        input_tensor=image_input
    )
    # Cargando los pesos de EfficientNetB3 Noisy Student que ya están en el notebook.
    base_model.load_weights('/kaggle/input/tf-efficientnet-noisy-student-weights/efficientnet-b3_noisy-student_notop.h5', by_name=True)
    base_model.trainable = False # Empezar congelando el 'backbone'

    # Cabezal del Modelo
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(N_TARGETS, activation='linear', name='output')(x)

    # --- Crear el Modelo ---
    model = keras.Model(
        inputs=image_input,
        outputs=output
    )

    return model, base_model

model, base_model = build_model()

# Compilar el modelo con la pérdida personalizada y la nueva métrica
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=weighted_mse_loss,
    metrics=[WeightedR2Score()]
)

model.summary()


# --- 5. ENTRENAMIENTO CON FINE-TUNING Y LR SCHEDULER ---

# Descongelamos las capas superiores del modelo para permitir el fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-40]: # Congelamos más capas para B3, que es más grande
    layer.trainable = False

# Re-compilamos el modelo para que los cambios de 'trainable' tengan efecto
# Usamos la tasa de aprendizaje inicial aquí. El scheduler la reducirá.
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=weighted_mse_loss,
    metrics=[WeightedR2Score()]
)

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# Aumentamos las épocas para dar tiempo al scheduler a trabajar
EPOCHS = 100

print("--- Iniciando Entrenamiento con Fine-Tuning ---")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[early_stopping, reduce_lr]
)

print("Entrenamiento completado.")


# --- 6. PREDICCIÓN Y SUMISIÓN ---

print("Generando predicciones de sumisión...")
df_test_long = pd.read_csv(TEST_CSV_PATH)

# El test.csv tiene múltiples filas por imagen, pero solo necesitamos predecir UNA VEZ por imagen
df_test_unique_images = df_test_long[['image_path']].drop_duplicates()

# Crear rutas de imagen completas para el conjunto de prueba
df_test_unique_images['image_path'] = df_test_unique_images['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))


def create_test_dataset(df):
    """Crea un dataset de solo imágenes para inferencia."""

    image_paths = df['image_path'].values

    ds_img = tf.data.Dataset.from_tensor_slices(image_paths)
    ds_img = ds_img.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds_img.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

test_ds = create_test_dataset(df_test_unique_images)

# Predecir con el modelo
# El resultado (preds) será un array (N_imagenes, 5)
preds = model.predict(test_ds)

# --- Lógica de Sumisión a Prueba de Errores ---

# 1. Crear un dataframe con las predicciones en formato ancho
df_preds_wide = pd.DataFrame(preds, columns=TARGET_NAMES)
# `df_test_unique_images.image_path` tiene la ruta completa, necesitamos la ruta relativa para el merge
df_preds_wide['image_path'] = df_test_unique_images['image_path'].apply(lambda x: os.path.relpath(x, BASE_PATH)).values

# 2. "Derretir" (melt) el dataframe para pasarlo a formato largo
df_preds_long = df_preds_wide.melt(
    id_vars=['image_path'],
    value_vars=TARGET_NAMES,
    var_name='target_name',
    value_name='predicted_target' # Renombrar para evitar colisión en el merge
)

# 3. Usar el df_test_long original como base para la sumisión
# Esto garantiza que todos los sample_id y el orden son correctos.
# Necesitamos el df_test_long ANTES del pre-procesamiento, así que lo volvemos a cargar.
df_submission_scaffold = pd.read_csv(TEST_CSV_PATH)

# 4. Unir (merge) el scaffold con nuestras predicciones
df_submission = pd.merge(
    df_submission_scaffold,
    df_preds_long,
    on=['image_path', 'target_name'],
    how='left' # 'left' para mantener todas las filas del archivo de test original
)

# 5. Seleccionar las columnas finales y renombrar
df_submission = df_submission[['sample_id', 'predicted_target']]
df_submission = df_submission.rename(columns={'predicted_target': 'target'})

# [Opcional] Forzar que no haya biomasa negativa
df_submission['target'] = df_submission['target'].clip(lower=0)

# 6. ¡Guardar!
df_submission.to_csv('submission.csv', index=False)

print("Archivo submission.csv creado exitosamente.")
print(df_submission.head())
