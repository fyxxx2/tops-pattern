import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ============================
# 설정
# ============================

CSV_PATH = "C:/Users/PC/Desktop/tops-pattern/ml/data/pattern_dataset_filtered.csv"
MODEL_SAVE_PATH = "C:/Users/PC/Desktop/tops-pattern/ml/models/efficientnet_pattern_crop.h5"

IMG_SIZE = (224, 224)

# ============================
# 1) 데이터 로드
# ============================

df = pd.read_csv(CSV_PATH)
print("Loaded:", df.shape)

# ============================
# 2) bbox crop 적용 함수
# ============================

def load_and_crop(row):
    path = row["filepath"]

    img = cv2.imread(path)
    if img is None:
        return None  # 오류 이미지 처리

    x, y, w, h = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_width"]), int(row["bbox_height"])

    # bbox 잘못된 경우 예외 처리
    h_img, w_img = img.shape[:2]
    x2, y2 = min(x + w, w_img), min(y + h, h_img)

    if x >= w_img or y >= h_img:
        return None

    crop = img[y:y2, x:x2]
    if crop.size == 0:
        return None

    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    crop = cv2.resize(crop, IMG_SIZE)
    crop = preprocess_input(crop)

    return crop


# ============================
# 3) 전체 이미지 로드 + bbox crop
# ============================

images = []
labels = []

print("Cropping images...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    crop = load_and_crop(row)
    if crop is not None:
        images.append(crop)
        labels.append(row["label_name"])

images = np.array(images)
print("Final image shape:", images.shape)

# ============================
# 4) 라벨 인코딩
# ============================

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

print("Classes:", le.classes_)

# ============================
# 5) 데이터 분리
# ============================

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    images, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# ============================
# 6) EfficientNet 기반 모델 구성
# ============================

base = EfficientNetB0(include_top=False, pooling="avg", input_shape=(224, 224, 3))
base.trainable = False  # 먼저 고정 (fine-tuning 전 단계)

inputs = Input(shape=(224, 224, 3))
x = base(inputs)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(len(le.classes_), activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================
# 7) 1차 학습 (head layer만 학습)
# ============================

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# ============================
# 8) EfficientNet 일부 레이어 해제 → 진짜 fine-tuning
# ============================

base.trainable = True

for layer in base.layers[:200]:  # 앞부분 70%는 다시 freeze
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Fine-tuning 시작!")

history2 = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)

# ============================
# 9) 모델 저장
# ============================

model.save(MODEL_SAVE_PATH)
print("Model saved at:", MODEL_SAVE_PATH)