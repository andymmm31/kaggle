import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import os

# --- 1. CONFIGURACIÓN Y CONSTANTES ---

# Rutas de Kaggle (basado en tu captura de pantalla)
BASE_PATH = '../input/csiro-image2biomass-prediction'
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
    
    # [POR HACER] Procesar características tabulares
    # Aquí debes manejar 'Sampling_Date', 'State', 'Species' si los vas a usar.
    # Por ejemplo, extraer mes/año, hacer one-hot encoding a 'State', etc.
    # Guardar las características únicas por 'image_path'.
    
    # Ejemplo de datos tabulares (solo los numéricos por ahora)
    # Agrupamos por imagen y tomamos la media (o primer valor) de los features
    df_tabular_features = df_train_long.groupby('image_path')[['Pre_GSHH_NDVI', 'Height_Ave_cm']].mean().reset_index()
    
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

# [POR HACER] Aquí deberías dividir df_train en train y validation
# train_df, val_df = train_test_split(df_train, test_size=0.1)


def preprocess_image(image_path):
    """Carga y pre-procesa una imagen."""
    img = tf.io.read_file(os.path.join(TRAIN_IMG_PATH, image_path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img) # Normalización de EfficientNet
    return img

def create_dataset(df):
    """Crea un tf.data.Dataset desde el dataframe ancho."""
    
    # [POR HACER] Actualiza esto para que coincida con tus features tabulares
    # (ej. NDVI, Altura, Estado_one_hot, Especie_one_hot, etc.)
    tabular_features = df[['Pre_GSHH_NDVI', 'Height_Ave_cm']].values
    
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
    
    # [POR HACER] Aplicar aumento de datos (data augmentation) aquí en `ds_img`
    # (ej. flips, rotaciones)
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Crear los datasets (usamos el df completo como ejemplo)
# Deberías usar train_df y val_df separados
train_ds = create_dataset(df_train)
# val_ds = create_dataset(val_df)


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
        weights='imagenet', # ¡CAMBIAR ESTO!
        input_tensor=image_input
    )
    base_model.trainable = False # Empezar congelando el 'backbone'
    
    # Cabezal de la CNN
    x_img = layers.GlobalAveragePooling2D()(base_model.output)
    x_img = layers.Dense(128, activation='relu')(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    
    # --- Rama 2: Entrada Tabular (MLP) ---
    tabular_input = layers.Input(shape=(n_tabular_features,), name='tabular_input')
    
    # [POR HACER] Normalizar los datos tabulares antes de pasarlos al modelo
    # (usando una capa de Normalization o pre-procesándolos)
    
    x_tab = layers.Dense(32, activation='relu')(tabular_input)
    x_tab = layers.Dense(16, activation='relu')(x_tab)

    
    # --- Fusión ---
    concatenated = layers.Concatenate()([x_img, x_tab])
    
    # --- Cabezal de Regresión (Head) ---
    x = layers.Dense(64, activation='relu')(concatenated)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(N_TARGETS, activation='linear', name='output')(output) # 'linear' para regresión
    
    # [Opcional] Usar 'relu' si la biomasa nunca puede ser negativa
    # output = layers.Dense(N_TARGETS, activation='relu', name='output')(x)

    
    # --- Crear el Modelo ---
    model = keras.Model(
        inputs=[image_input, tabular_input],
        outputs=output
    )
    
    return model

# [POR HACER] Asegúrate que este número coincida con tus features tabulares
N_FEATURES = 2 # (NDVI, Height)
model = build_model(n_tabular_features=N_FEATURES)

# Compilar el modelo con la pérdida personalizada
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=weighted_mse_loss
)

model.summary()


# --- 5. ENTRENAMIENTO ---

print("Iniciando entrenamiento...")
# [POR HACER] Descomentar para entrenar y añadir el 'val_ds'
# history = model.fit(
#     train_ds,
#     epochs=EPOCHS,
#     # validation_data=val_ds 
# )
print("Entrenamiento (simulado) completado.")


# --- 6. PREDICCIÓN Y SUMISIÓN ---

print("Generando predicciones de sumisión...")
df_test_long = pd.read_csv(TEST_CSV_PATH)

# [POR HACER] Necesitas procesar los features tabulares del test set
# de la misma forma que lo hiciste con el train set.
# Por ahora, usamos los mismos features numéricos.
df_test_features = df_test_long.groupby('image_path')[['Pre_GSHH_NDVI', 'Height_Ave_cm']].mean().reset_index()

# El test.csv tiene múltiples filas por imagen, pero solo necesitamos predecir UNA VEZ por imagen
df_test_unique_images = df_test_features.drop_duplicates(subset=['image_path'])

def create_test_dataset(df):
    """Crea un dataset para inferencia (solo inputs)."""
    
    # [POR HACER] Actualiza esto para que coincida con tus features tabulares
    tabular_features = df[['Pre_GSHH_NDVI', 'Height_Ave_cm']].values
    
    image_paths = df['image_path'].values
    
    # Pre-procesar imágenes (desde la carpeta de test)
    def preprocess_test_image(image_path):
        img = tf.io.read_file(os.path.join(TEST_IMG_PATH, image_path)) # ¡Ruta de TEST!
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    ds_img = tf.data.Dataset.from_tensor_slices(image_paths)
    ds_img = ds_img.map(preprocess_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    
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
df_preds_long['sample_id'] = df_preds_long['image_path'].str.replace('test/', '', regex=False) + '__' + df_preds_long['target_name']

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
