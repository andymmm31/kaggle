import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
import os
from sklearn.model_selection import KFold

# --- 1. CONFIGURACIÓN Y CONSTANTES ---
BASE_PATH = '/kaggle/input/csiro-biomass-challenge'
TRAIN_IMG_PATH = os.path.join(BASE_PATH, 'train')
TEST_IMG_PATH = os.path.join(BASE_PATH, 'test')
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
TEST_CSV_PATH = os.path.join(BASE_PATH, 'test.csv')

IMG_SIZE = 300
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 30
TARGET_NAMES = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
N_TARGETS = len(TARGET_NAMES)
LOSS_WEIGHTS = tf.constant([0.1, 0.1, 0.1, 0.2, 0.5])

# --- 2. PRE-PROCESAMIENTO DE DATOS ---
def load_and_pivot_data():
    df_train_long = pd.read_csv(TRAIN_CSV_PATH)
    df_train_wide = df_train_long.pivot(index='image_path', columns='target_name', values='target').reset_index()
    df_train_wide = df_train_wide.dropna()
    print(f"Datos 'anchos' para entrenamiento: {df_train_wide.shape}")
    return df_train_wide

df_train = load_and_pivot_data()
df_train['image_path'] = df_train['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))

def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def create_dataset(df, is_train=True):
    image_paths = df['image_path'].values
    targets = df[TARGET_NAMES].values
    ds_img = tf.data.Dataset.from_tensor_slices(image_paths).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_tgt = tf.data.Dataset.from_tensor_slices(targets)
    ds = tf.data.Dataset.zip((ds_img, ds_tgt))
    if is_train:
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
        ])
        ds = ds.map(lambda img, tgt: (data_augmentation(img), tgt), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# --- 3. MÉTRICA Y FUNCIÓN DE PÉRDIDA PERSONALIZADAS ---
class WeightedR2Score(tf.keras.metrics.Metric):
    def __init__(self, name='weighted_r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.weights = tf.constant([0.1, 0.1, 0.1, 0.2, 0.5], dtype=tf.float32)
        self.sum_ss_res = self.add_weight(name='sum_ss_res', initializer='zeros')
        self.sum_y_true = self.add_weight(name='sum_y_true', shape=(N_TARGETS,), initializer='zeros')
        self.sum_y_true_sq = self.add_weight(name='sum_y_true_sq', shape=(N_TARGETS,), initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        ss_res = tf.reduce_sum(self.weights * tf.square(y_true - y_pred))
        self.sum_ss_res.assign_add(ss_res)
        self.sum_y_true.assign_add(tf.reduce_sum(y_true, axis=0))
        self.sum_y_true_sq.assign_add(tf.reduce_sum(tf.square(y_true), axis=0))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
    def result(self):
        ss_tot_per_target = self.sum_y_true_sq - (tf.square(self.sum_y_true) / self.count)
        total_ss_tot = tf.reduce_sum(self.weights * ss_tot_per_target)
        return 1.0 - (self.sum_ss_res / (total_ss_tot + tf.keras.backend.epsilon()))
    def reset_state(self):
        self.sum_ss_res.assign(0.0)
        self.sum_y_true.assign(tf.zeros(shape=(N_TARGETS,)))
        self.sum_y_true_sq.assign(tf.zeros(shape=(N_TARGETS,)))
        self.count.assign(0.0)

def weighted_mse_loss(y_true, y_pred):
    error_sq = tf.square(y_true - y_pred)
    weighted_error_sq = error_sq * LOSS_WEIGHTS
    return tf.reduce_mean(weighted_error_sq)

# --- 4. ARQUITECTURA DEL MODELO ---
def build_model():
    image_input = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    base_model = EfficientNetB3(include_top=False, weights=None, input_tensor=image_input)
    # Asumiendo que los pesos están en un dataset adjunto con este nombre
    base_model.load_weights('/kaggle/input/efficientnetb3-notop-h5/efficientnetb3_notop.h5', by_name=True)
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(N_TARGETS, activation='linear', name='output')(x)
    model = keras.Model(inputs=image_input, outputs=output)
    return model, base_model

# --- 5. ENTRENAMIENTO CON CROSS-VALIDATION ---
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
    print(f"--- Fold {fold+1}/{N_FOLDS} ---")
    train_df_fold, val_df_fold = df_train.iloc[train_idx], df_train.iloc[val_idx]
    train_ds = create_dataset(train_df_fold, is_train=True)
    val_ds = create_dataset(val_df_fold, is_train=False)
    model, base_model = build_model()
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    # Calendario de Tasa de Aprendizaje
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=len(train_df_fold) // BATCH_SIZE * 5, # Decae cada 5 épocas
        decay_rate=0.5)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=weighted_mse_loss, metrics=[WeightedR2Score()])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)
    model.save_weights(f'model_fold_{fold}.weights.h5')

# --- 6. PREDICCIÓN CON ENSEMBLE Y TTA ---
df_test_long = pd.read_csv(TEST_CSV_PATH)
df_test_unique_images = df_test_long[['image_path']].drop_duplicates()
df_test_unique_images['image_path'] = df_test_unique_images['image_path'].apply(lambda x: os.path.join(BASE_PATH, x))

def create_test_dataset(df, augment=False):
    ds = tf.data.Dataset.from_tensor_slices(df['image_path'].values).map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(lambda img: tf.image.flip_left_right(img), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

test_ds_original = create_test_dataset(df_test_unique_images, augment=False)
test_ds_augmented = create_test_dataset(df_test_unique_images, augment=True)
inference_model, _ = build_model()
all_preds = []
for fold in range(N_FOLDS):
    inference_model.load_weights(f'model_fold_{fold}.weights.h5')
    preds_original = inference_model.predict(test_ds_original)
    preds_augmented = inference_model.predict(test_ds_augmented)
    avg_preds = (preds_original + preds_augmented) / 2.0
    all_preds.append(avg_preds)

final_preds = np.mean(all_preds, axis=0)
df_preds_wide = pd.DataFrame(final_preds, columns=TARGET_NAMES)
df_preds_wide['image_path'] = df_test_unique_images['image_path'].apply(lambda x: os.path.relpath(x, BASE_PATH)).values
df_preds_long = df_preds_wide.melt(id_vars=['image_path'], value_vars=TARGET_NAMES, var_name='target_name', value_name='predicted_target')
df_submission_scaffold = pd.read_csv(TEST_CSV_PATH)
df_submission = pd.merge(df_submission_scaffold, df_preds_long, on=['image_path', 'target_name'], how='left')
df_submission = df_submission[['sample_id', 'predicted_target']].rename(columns={'predicted_target': 'target'})
df_submission['target'] = df_submission['target'].clip(lower=0)
df_submission.to_csv('submission.csv', index=False)
print("submission.csv created successfully.")
