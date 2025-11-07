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

def load_and_process_data():
    """Carga train.csv, pivota los targets y conserva los metadatos."""
    df_train_long = pd.read_csv(TRAIN_CSV_PATH)

    # Pivota los targets
    df_targets = df_train_long.pivot(index='image_path', columns='target_name', values='target').reset_index()

    # Obtiene los metadatos únicos por imagen
    df_meta = df_train_long[['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']].drop_duplicates()

    # Une los targets y los metadatos
    df_train_wide = pd.merge(df_meta, df_targets, on='image_path')

    # Elimina filas con valores faltantes
    df_train_wide = df_train_wide.dropna()

    print(f"Datos de entrenamiento procesados: {df_train_wide.shape}")
    print(df_train_wide.head())

    return df_train_wide

# Cargar los datos
df_train = load_and_process_data()

# --- INGENIERÍA DE CARACTERÍSTICAS ---
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def feature_engineer(df):
    # Procesamiento de Fechas
    df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
    df['day_of_year'] = df['Sampling_Date'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365.25)
    return df

print("Aplicando ingeniería de características al conjunto de entrenamiento...")
df_train = feature_engineer(df_train)

# One-Hot Encoding para categóricas
categorical_features = ['State', 'Species']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(df_train[categorical_features])
encoded_cols = encoder.get_feature_names_out(categorical_features)
df_train[encoded_cols] = encoder.transform(df_train[categorical_features])

# Normalización para numéricas
numerical_features = ['Pre_GSHH_NDVI', 'Height_Ave_cm']
scaler = StandardScaler()
scaler.fit(df_train[numerical_features])
df_train[numerical_features] = scaler.transform(df_train[numerical_features])

# Lista final de meta-features
META_FEATURES = ['day_of_year_sin', 'day_of_year_cos'] + numerical_features + list(encoded_cols)
N_META_FEATURES = len(META_FEATURES)

print(f"Número total de meta-features: {N_META_FEATURES}")

# Crear rutas de imagen completas
df_train['image_path'] = df_train['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))


def preprocess_image(image_path):
    """Carga y pre-procesa una imagen desde una ruta completa."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img) # Normalización de EfficientNet
    return img

def create_dataset(df, meta_features, is_train=True):
    """Crea un tf.data.Dataset para el modelo multi-modal."""

    # Datasets de entradas
    ds_img = tf.data.Dataset.from_tensor_slices(df['image_path'].values).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_meta = tf.data.Dataset.from_tensor_slices(meta_features.astype(np.float32))

    # Combinar las entradas en un diccionario
    ds_inputs = tf.data.Dataset.zip(({'image_input': ds_img, 'meta_input': ds_meta}))

    # Dataset de targets
    ds_tgt = tf.data.Dataset.from_tensor_slices(df[TARGET_NAMES].values.astype(np.float32))

    # Combinar (entradas, targets)
    ds = tf.data.Dataset.zip((ds_inputs, ds_tgt))

    if is_train:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])
        # Aplicar aumento solo a la imagen
        ds = ds.map(lambda inputs, tgt: ({'image_input': data_augmentation(inputs['image_input']), 'meta_input': inputs['meta_input']}, tgt),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# --- 3. MÉTRICA Y FUNCIÓN DE PÉRDIDA PERSONALIZADAS ---

class WeightedR2Score(tf.keras.metrics.Metric):
    """Métrica R² ponderada alineada con la competición."""
    def __init__(self, name='weighted_r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights = tf.constant([0.1, 0.1, 0.1, 0.2, 0.5], dtype=tf.float32)
        self.y_true_sum = self.add_weight(name='y_true_sum', shape=(N_TARGETS,), initializer='zeros')
        self.y_true_sq_sum = self.add_weight(name='y_true_sq_sum', shape=(N_TARGETS,), initializer='zeros')
        self.y_pred_sq_sum = self.add_weight(name='y_pred_sq_sum', shape=(N_TARGETS,), initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        self.y_true_sum.assign_add(tf.reduce_sum(y_true, axis=0))
        self.y_true_sq_sum.assign_add(tf.reduce_sum(tf.square(y_true), axis=0))
        self.y_pred_sq_sum.assign_add(tf.reduce_sum(tf.square(y_true - y_pred), axis=0))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], self.dtype))

    def result(self):
        y_true_mean = self.y_true_sum / self.count

        # Media ponderada global de y_true
        weighted_mean = tf.reduce_sum(y_true_mean * self.weights)

        # Suma de cuadrados residual (SS_res)
        ss_res = tf.reduce_sum(self.y_pred_sq_sum * self.weights)

        # Suma de cuadrados total (SS_tot)
        ss_tot = tf.reduce_sum(tf.square(self.y_true_sum - weighted_mean) * self.weights)
        ss_tot_alternative = tf.reduce_sum((self.y_true_sq_sum - 2 * self.y_true_sum * weighted_mean + self.count * tf.square(weighted_mean)) * self.weights)

        return 1 - (ss_res / (ss_tot_alternative + tf.keras.backend.epsilon()))

    def reset_state(self):
        self.y_true_sum.assign(tf.zeros_like(self.y_true_sum))
        self.y_true_sq_sum.assign(tf.zeros_like(self.y_true_sq_sum))
        self.y_pred_sq_sum.assign(tf.zeros_like(self.y_pred_sq_sum))
        self.count.assign(0.)


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


# --- 4. ARQUITECTURA DEL MODELO MULTI-MODAL ---

def build_model():
    """Construye el modelo multi-modal (CNN + MLP)."""

    # --- Rama 1: Entrada de Imagen (CNN) ---
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    base_model = EfficientNetB3(include_top=False, weights=None, input_tensor=image_input)
    base_model.load_weights('/kaggle/input/tf-efficientnet-noisy-student-weights/efficientnet-b3_noisy-student_notop.h5', by_name=True)
    base_model.trainable = False
    cnn_output = layers.GlobalAveragePooling2D()(base_model.output)
    cnn_output = layers.Dense(128, activation='relu')(cnn_output)

    # --- Rama 2: Entrada de Metadatos (MLP) ---
    meta_input = layers.Input(shape=(N_META_FEATURES,), name='meta_input')
    mlp_output = layers.Dense(64, activation='relu')(meta_input)
    mlp_output = layers.Dense(32, activation='relu')(mlp_output)

    # --- Fusión y Cabezal Final ---
    concatenated = layers.Concatenate()([cnn_output, mlp_output])
    x = layers.Dense(128, activation='relu')(concatenated)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(N_TARGETS, activation='linear', name='output')(x)

    # --- Crear el Modelo ---
    model = keras.Model(inputs={'image_input': image_input, 'meta_input': meta_input}, outputs=output)

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

    # Extraer los meta-features para este fold
    train_meta_fold = train_df_fold[META_FEATURES].values
    val_meta_fold = val_df_fold[META_FEATURES].values

    # Crear datasets para este fold
    train_ds = create_dataset(train_df_fold, train_meta_fold, is_train=True)
    val_ds = create_dataset(val_df_fold, val_meta_fold, is_train=False)

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
    model.save_weights(f'model_fold_{fold}.weights.h5')

print("\n--- Entrenamiento con Cross-Validation completado. ---")


# --- 6. PREDICCIÓN CON ENSEMBLE Y TTA ---

print("Generando predicciones de sumisión con Ensemble y TTA...")
# Cargar datos de prueba y obtener filas únicas por imagen
df_test_long = pd.read_csv(TEST_CSV_PATH)
df_test = df_test_long.drop(columns=['sample_id', 'target_name']).drop_duplicates().reset_index(drop=True)

# Aplicar ingeniería de características al conjunto de prueba
print("Aplicando ingeniería de características al conjunto de prueba...")
df_test = feature_engineer(df_test)
df_test[encoded_cols] = encoder.transform(df_test[categorical_features])
df_test[numerical_features] = scaler.transform(df_test[numerical_features])

# Crear rutas de imagen completas
df_test['image_path'] = df_test['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))

# Extraer los meta-features para la predicción
df_test_meta = df_test[META_FEATURES].values

def create_test_dataset(df, meta_features, augment=False):
    """Crea un dataset de test para el modelo multi-modal."""

    ds_img = tf.data.Dataset.from_tensor_slices(df['image_path'].values).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds_img = ds_img.map(lambda img: tf.image.flip_left_right(img), num_parallel_calls=tf.data.AUTOTUNE)

    ds_meta = tf.data.Dataset.from_tensor_slices(meta_features.astype(np.float32))

    ds_inputs = tf.data.Dataset.zip(({'image_input': ds_img, 'meta_input': ds_meta}))

    ds = ds_inputs.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# Construir un modelo solo para cargar los pesos
inference_model, _ = build_model()
all_preds = []

for fold in range(N_FOLDS):
    print(f"Prediciendo con modelo del Fold {fold+1}/{N_FOLDS}...")
    inference_model.load_weights(f'model_fold_{fold}.weights.h5')

    # Crear datasets de test para este fold
    test_ds_original = create_test_dataset(df_test, df_test_meta, augment=False)
    test_ds_augmented = create_test_dataset(df_test, df_test_meta, augment=True)

    # Predecir en datos originales y aumentados
    preds_original = inference_model.predict(test_ds_original)
    preds_augmented = inference_model.predict(test_ds_augmented)

    # Promediar las predicciones de TTA y añadir a la lista
    avg_preds = (preds_original + preds_augmented) / 2.0
    all_preds.append(avg_preds)

# Promediar las predicciones de todos los folds (Ensemble)
final_preds = np.mean(all_preds, axis=0)

# --- Generar archivo de sumisión (misma lógica que antes) ---
df_preds_wide = pd.DataFrame(final_preds, columns=TARGET_NAMES)
df_preds_wide['image_path'] = df_test['image_path'].apply(lambda x: os.path.relpath(x, BASE_PATH)).values

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
