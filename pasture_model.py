import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
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
IMG_SIZE = 224 # Tamaño para EfficientNetB0
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10 # Empezar con pocas para probar
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
N_TARGETS = len(TARGET_NAMES)

# Pesos para la métrica y la función de pérdida
# Dry_Green_g: 0.1, Dry_Dead_g: 0.1, Dry_Clover_g: 0.1, GDM_g: 0.2, Dry_Total_g: 0.5
LOSS_WEIGHTS = tf.constant([0.1, 0.1, 0.1, 0.2, 0.5])


# --- 2. PRE-PROCESAMIENTO DE DATOS ---

def load_and_pivot_data():
    """Carga train.csv y lo pivota a formato ancho."""
    df_train_long = pd.read_csv(TRAIN_CSV_PATH)
    
    # --- Ingeniería de Características ---
    # 1. Procesar 'Sampling_Date'
    df_train_long['Sampling_Date'] = pd.to_datetime(df_train_long['Sampling_Date'])
    df_train_long['day_of_year'] = df_train_long['Sampling_Date'].dt.dayofyear
    
    # 2. One-Hot Encoding para 'State' y 'Species'
    df_train_long = pd.get_dummies(df_train_long, columns=['State', 'Species'], prefix=['State', 'Species'])

    # 3. Consolidar características
    # Primero, definimos las columnas de características que no cambian por imagen
    # (NDVI, Altura, fecha, y las nuevas columnas one-hot)
    feature_cols = ['Pre_GSHH_NDVI', 'Height_Ave_cm', 'day_of_year'] + \
                   [col for col in df_train_long.columns if col.startswith('State_') or col.startswith('Species_')]

    # Agrupamos por imagen y tomamos el primer valor (ya que son constantes por imagen)
    df_tabular_features = df_train_long.groupby('image_path')[feature_cols].first().reset_index()
    
    # Pivoteamos los targets para tener una fila por imagen
    df_train_wide = df_train_long.pivot(
        index='image_path',
        columns='target_name',
        values='target'
    ).reset_index()
    
    # Asegurarnos que el orden de columnas sea el correcto
    df_train_wide = df_train_wide.merge(df_tabular_features, on='image_path')
    df_train_wide = df_train_wide.dropna() # Quitar filas con targets faltantes
    
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

# Obtener la lista de características tabulares después de la ingeniería de características
TABULAR_FEATURES = [col for col in train_df.columns if col not in ['image_path'] + TARGET_NAMES]
N_FEATURES = len(TABULAR_FEATURES)

# Asegurarse de que todas las características tabulares sean float32
for col in TABULAR_FEATURES:
    train_df[col] = train_df[col].astype('float32')
    val_df[col] = val_df[col].astype('float32')


def preprocess_image(image_path):
    """Carga y pre-procesa una imagen desde una ruta completa."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img) # Normalización de EfficientNet
    return img

def create_dataset(df):
    """Crea un tf.data.Dataset desde el dataframe ancho."""
    
    tabular_features = df[TABULAR_FEATURES].values
    
    image_paths = df['image_path'].values
    targets = df[TARGET_NAMES].values
    
    # Dataset de rutas de imágenes
    ds_img = tf.data.Dataset.from_tensor_slices(image_paths)
    ds_img = ds_img.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Dataset de features tabulares
    ds_tab = tf.data.Dataset.from_tensor_slices(tabular_features)
    
    # Dataset de targets
    ds_tgt = tf.data.Dataset.from_tensor_slices(targets)
    
    # Combinar en un dataset de ( (img, tab), tgt )
    ds = tf.data.Dataset.zip(( (ds_img, ds_tab), ds_tgt ))
    
    # --- Aumento de Datos ---
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    ds = ds.map(lambda x, y: ( (data_augmentation(x[0]), x[1]), y), num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Crear los datasets de entrenamiento y validación
train_ds = create_dataset(train_df)
val_ds = create_dataset(val_df)


# --- 3. FUNCIÓN DE PÉRDIDA PERSONALIZADA ---

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

def build_model(n_tabular_features):
    """Construye el modelo multimodal (CNN + MLP)."""
    
    # --- Rama 1: Entrada de Imagen (CNN) ---
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    
    # ¡¡¡IMPORTANTE!!!
    # 'weights="imagenet"' REQUIERE INTERNET. Fallará en la sumisión.
    # [POR HACER]: 
    # 1. En Kaggle, haz clic en "Add Input" -> "Kaggle Datasets".
    # 2. Busca y añade un dataset de pesos de EfficientNetB0 (ej. "efficientnet-b0-keras-tl-weights").
    # 3. Cambia 'weights="imagenet"' por 'weights=None'.
    # 4. Después de 'base_model = ...', carga los pesos manualmente:
    #    base_model.load_weights('../input/dataset-de-pesos/efficientnetb0_notop.h5')
    
    base_model = EfficientNetB0(
        include_top=False, 
        weights=None,
        input_tensor=image_input
    )
    base_model.load_weights('/kaggle/input/tf-efficientnet-noisy-student-weights/efficientnet-b0_noisy-student_notop.h5', by_name=True)
    base_model.trainable = False # Empezar congelando el 'backbone'
    
    # Cabezal de la CNN
    x_img = layers.GlobalAveragePooling2D()(base_model.output)
    x_img = layers.Dense(128, activation='relu')(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    
    # --- Rama 2: Entrada Tabular (MLP) ---
    tabular_input = layers.Input(shape=(n_tabular_features,), name='tabular_input')
    
    # Capa de Normalización
    normalizer = layers.Normalization(name='normalizer')
    normalizer.adapt(train_df[TABULAR_FEATURES].values)
    
    x_tab = normalizer(tabular_input)
    x_tab = layers.Dense(32, activation='relu')(x_tab)
    x_tab = layers.Dense(16, activation='relu')(x_tab)

    
    # --- Fusión ---
    concatenated = layers.Concatenate()([x_img, x_tab])
    
    # --- Cabezal de Regresión (Head) ---
    x = layers.Dense(64, activation='relu')(concatenated)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(N_TARGETS, activation='linear', name='output')(x) # 'linear' para regresión
    
    # [Opcional] Usar 'relu' si la biomasa nunca puede ser negativa
    # output = layers.Dense(N_TARGETS, activation='relu', name='output')(x)

    
    # --- Crear el Modelo ---
    model = keras.Model(
        inputs=[image_input, tabular_input],
        outputs=output
    )
    
    return model

model = build_model(n_tabular_features=N_FEATURES)

# Compilar el modelo con la pérdida personalizada
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=weighted_mse_loss
)

model.summary()


# --- 5. ENTRENAMIENTO ---

print("Iniciando entrenamiento...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    epochs=50, # Aumentar épocas
    validation_data=val_ds,
    callbacks=[early_stopping]
)
print("Entrenamiento completado.")


# --- 6. PREDICCIÓN Y SUMISIÓN ---

print("Generando predicciones de sumisión...")
df_test_long = pd.read_csv(TEST_CSV_PATH)

# Aplicar la misma ingeniería de características al conjunto de prueba
df_test_long['Sampling_Date'] = pd.to_datetime(df_test_long['Sampling_Date'])
df_test_long['day_of_year'] = df_test_long['Sampling_Date'].dt.dayofyear
df_test_long = pd.get_dummies(df_test_long, columns=['State', 'Species'], prefix=['State', 'Species'])

# Alinear columnas con el conjunto de entrenamiento (importante para one-hot encoding)
train_cols = set(train_df.columns)
test_cols = set(df_test_long.columns)

missing_in_test = list(train_cols - test_cols)
for col in missing_in_test:
    if col.startswith('State_') or col.startswith('Species_'):
        df_test_long[col] = 0

# Asegurarse de que el orden de las columnas sea el mismo
df_test_features = df_test_long.groupby('image_path')[TABULAR_FEATURES].first().reset_index()

# El test.csv tiene múltiples filas por imagen, pero solo necesitamos predecir UNA VEZ por imagen
df_test_unique_images = df_test_features.drop_duplicates(subset=['image_path'])

# Crear rutas de imagen completas para el conjunto de prueba
df_test_unique_images['image_path'] = df_test_unique_images['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))

# Asegurarse de que todas las características tabulares sean float32 en el conjunto de prueba
for col in TABULAR_FEATURES:
    df_test_unique_images[col] = df_test_unique_images[col].astype('float32')

def create_test_dataset(df):
    """Crea un dataset para inferencia (solo inputs)."""
    
    tabular_features = df[TABULAR_FEATURES].values
    
    image_paths = df['image_path'].values
    
    # La función de preprocesamiento de entrenamiento funciona aquí también, ya que ahora toma rutas completas
    ds_img = tf.data.Dataset.from_tensor_slices(image_paths)
    ds_img = ds_img.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds_tab = tf.data.Dataset.from_tensor_slices(tabular_features)
    
    ds = tf.data.Dataset.zip((ds_img, ds_tab))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

test_ds = create_test_dataset(df_test_unique_images)

# Predecir con el modelo
# El resultado (preds) será un array (N_imagenes, 5)
preds = model.predict(test_ds)

# Mapear predicciones de (N, 5) a formato largo
# Crear un dataframe con las predicciones en formato ancho
df_preds_wide = pd.DataFrame(preds, columns=TARGET_NAMES)
df_preds_wide['image_path'] = df_test_unique_images['image_path'].values

# "Derretir" (melt) el dataframe para pasarlo a formato largo
df_preds_long = df_preds_wide.melt(
    id_vars=['image_path'],
    value_vars=TARGET_NAMES,
    var_name='target_name',
    value_name='target'
)

# Crear el 'sample_id'
df_preds_long['image_path_basename'] = df_preds_long['image_path'].apply(os.path.basename)
df_preds_long['sample_id'] = df_preds_long['image_path_basename'] + '__' + df_preds_long['target_name']

# Generar el archivo de sumisión
# Nos aseguramos de tener el mismo orden y columnas que sample_submission.csv
df_submission = df_preds_long[['sample_id', 'target']]

# [Opcional] Si el modelo predijo 'relu' (solo positivos), está bien.
# Si usó 'linear', podríamos forzar que no haya biomasa negativa.
df_submission['target'] = df_submission['target'].clip(lower=0) 

# ¡Guardar!
df_submission.to_csv('submission.csv', index=False)

print("Archivo submission.csv creado exitosamente.")
print(df_submission.head())
