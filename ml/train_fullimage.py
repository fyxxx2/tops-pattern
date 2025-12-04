import tensorflow as tf
import numpy as np
import os

# ===============================
# 1. 경로 설정
# ===============================
DATA_DIR = "C:/Users/rlarh/Desktop/tops-pattern/data/raw/tops"
SAVE_PATH = "C:/Users/rlarh/Desktop/tops-pattern/ml/models/efficientnet_fullimage.keras"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ===============================
# 2. 데이터셋 만들기
# ===============================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# Prefetch (GPU/CPU 효율적 사용)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# ===============================
# 3. EfficientNetB0 모델 로드
# ===============================
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", pooling="avg"
)
base_model.trainable = False  # 1단계: 전체 fine-tuning 전 freeze

# ===============================
# 4. Classification Head 추가
# ===============================
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# 5. 1차 학습 (Head만 학습)
# ===============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ===============================
# 6. Fine-tuning 단계 (EfficientNet 상위 레이어 풀기)
# ===============================
base_model.trainable = True

for layer in base_model.layers[:150]:
    layer.trainable = False  # 나머지는 학습

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("🔥 Fine-tuning 시작합니다…")

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# ===============================
# 7. 저장
# ===============================
model.save(SAVE_PATH)

print(f"Model saved at: {SAVE_PATH}")