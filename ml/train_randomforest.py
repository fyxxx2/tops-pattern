import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# 1) Load CSV
# -----------------------------
CSV_PATH = "C:/Users/rlarh/Desktop/tops-pattern/data/annotations/pattern_dataset_filtered.csv"
df = pd.read_csv(CSV_PATH)
df['filepath'] = df['filepath'].str.replace("\\", "/", regex=False)

# -----------------------------
# 2) Label Encoding
# -----------------------------
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label_name'])
labels = df['label_id'].values
num_classes = len(le.classes_)

print("Classes:", le.classes_)

# -----------------------------
# 3) Image Loader (Correct EfficientNet Preprocessing!)
# -----------------------------
IMG_SIZE = (224, 224)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)   # ★ EfficientNet preprocessing 적용
    return img

print("Loading Images...")
X = np.array([load_image(p) for p in tqdm(df['filepath'])])

# -----------------------------
# 4) Train/Test Split (정상 stratify)
# -----------------------------
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# One-hot encoding AFTER splitting
y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test_raw, num_classes)

# -----------------------------
# 5) Build EfficientNet Model
# -----------------------------
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling='avg'
)

# Fine-Tuning: 상위 레이어 50개 학습 허용
for layer in base_model.layers[:-50]:
    layer.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=True)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# -----------------------------
# 6) Compile (Lower LR for stability)
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 7) Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=12
)

# -----------------------------
# 8) Save Model (Keras recommended format)
# -----------------------------
SAVE_DIR = "C:/Users/rlarh/Desktop/tops-pattern/ml/models"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_PATH = os.path.join(SAVE_DIR, "efficientnet_pattern.keras")
model.save(SAVE_PATH)

print(f"\nModel saved at: {SAVE_PATH}")
