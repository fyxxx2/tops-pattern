import argparse
import cv2
import numpy as np
import tensorflow as tf
import os

# 모델 경로
MODEL_PATH = "ml/models/efficientnet_pattern_crop.h5"

# EfficientNet 입력 크기
IMG_SIZE = (224, 224)

# 클래스 이름 (훈련 시 사용한 순서와 동일해야 함)
CLASS_NAMES = ["floral", "plaid", "polka_dot", "solid", "stripe"]


def load_and_preprocess_image(path):
    """이미지 경로에서 이미지를 읽어서 EfficientNet 입력 형식으로 변환."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(path):
    """이미지를 분류하고 예측 결과를 반환."""
    print(f"[INFO] 모델 로드 중... {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"[INFO] 이미지 불러오는 중... {path}")
    img = load_and_preprocess_image(path)

    preds = model.predict(img)
    pred_index = np.argmax(preds)
    confidence = float(preds[0][pred_index])

    label = CLASS_NAMES[pred_index]

    print("\n==============================")
    print(f"📌 예측 결과: {label}")
    print(f"📌 Confidence: {confidence:.4f}")
    print("==============================\n")

    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    args = parser.parse_args()

    predict_image(args.image)