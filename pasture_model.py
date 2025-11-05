import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
import os
from sklearn.model_selection import KFold

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

# Crear rutas de imagen completas
df_train['image_path'] = df_train['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))


def preprocess_image(image_path):
    """Carga y pre-procesa una imagen desde una ruta completa."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img) # Normalización de EfficientNet
    return img

def create_dataset(df, is_train=True):
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

    # Aplicar aumento de datos solo al set de entrenamiento
    if is_train:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda img, tgt: (data_augmentation(img), tgt), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


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


# --- 5. ENTRENAMIENTO CON CROSS-VALIDATION ---
N_FOLDS = 5
EPOCHS = 60 # Épocas por fold
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
    print(f"\n--- Iniciando Entrenamiento para Fold {fold+1}/{N_FOLDS} ---")

    # Crear dataframes para este fold
    train_df_fold = df_train.iloc[train_idx]
    val_df_fold = df_train.iloc[val_idx]

    # Crear datasets para este fold
    train_ds = create_dataset(train_df_fold, is_train=True)
    val_ds = create_dataset(val_df_fold, is_train=False) # No aumentar datos en validación

    # Construir y compilar un modelo nuevo para cada fold
    model, base_model = build_model()

    # Descongelar para fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=weighted_mse_loss,
        metrics=[WeightedR2Score()]
    )

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

    # Entrenar el modelo
    model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=[early_stopping, reduce_lr]
    )

    # Guardar los pesos del modelo entrenado
    model.save_weights(f'model_fold_{fold}.h5')

print("\n--- Entrenamiento con Cross-Validation completado. ---")


# --- 6. PREDICCIÓN CON ENSEMBLE Y TTA ---

print("Generando predicciones de sumisión con Ensemble y TTA...")
df_test_long = pd.read_csv(TEST_CSV_PATH)
df_test_unique_images = df_test_long[['image_path']].drop_duplicates()
df_test_unique_images['image_path'] = df_test_unique_images['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))

def create_test_dataset(df, augment=False):
    """Crea un dataset de test, con opción de aumento para TTA."""
    ds = tf.data.Dataset.from_tensor_slices(df['image_path'].values)
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Para TTA, solo usaremos un aumento simple como el volteo horizontal
        ds = ds.map(lambda img: tf.image.random_flip_left_right(img), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Crear datasets de test (original y aumentado para TTA)
test_ds_original = create_test_dataset(df_test_unique_images, augment=False)
test_ds_augmented = create_test_dataset(df_test_unique_images, augment=True)

# Construir un modelo solo para cargar los pesos
inference_model, _ = build_model()
all_preds = []

for fold in range(N_FOLDS):
    print(f"Prediciendo con modelo del Fold {fold+1}/{N_FOLDS}...")
    inference_model.load_weights(f'model_fold_{fold}.h5')

    # Predecir en imágenes originales
    preds_original = inference_model.predict(test_ds_original)
    # Predecir en imágenes aumentadas (TTA)
    preds_augmented = inference_model.predict(test_ds_augmented)

    # Promediar las predicciones de TTA y añadir a la lista
    avg_preds = (preds_original + preds_augmented) / 2.0
    all_preds.append(avg_preds)

# Promediar las predicciones de todos los folds (Ensemble)
final_preds = np.mean(all_preds, axis=0)

# --- Generar archivo de sumisión (misma lógica que antes) ---
df_preds_wide = pd.DataFrame(final_preds, columns=TARGET_NAMES)
df_preds_wide['image_path'] = df_test_unique_images['image_path'].apply(lambda x: os.path.relpath(x, BASE_PATH)).values

df_preds_long = df_preds_wide.melt(
    id_vars=['image_path'],
    value_vars=TARGET_NAMES,
    var_name='target_name',
    value_name='predicted_target'
)

df_submission_scaffold = pd.read_csv(TEST_CSV_PATH)
df_submission = pd.merge(
    df_submission_scaffold,
    df_preds_long,
    on=['image_path', 'target_name'],
    how='left'
)

df_submission = df_submission[['sample_id', 'predicted_target']].rename(columns={'predicted_target': 'target'})
df_submission['target'] = df_submission['target'].clip(lower=0)
df_submission.to_csv('submission.csv', index=False)

print("\nArchivo submission.csv creado exitosamente.")
print(df_submission.head())
