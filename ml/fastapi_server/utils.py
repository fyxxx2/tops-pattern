import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)

def apply_clahe(img):
    """조명 보정(CLAHE)"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def center_crop(img):
    """세로 또는 가로로 긴 이미지를 중앙 crop"""
    h, w, _ = img.shape
    min_dim = min(h, w)

    start_x = w // 2 - min_dim // 2
    start_y = h // 2 - min_dim // 2

    return img[start_y:start_y + min_dim, start_x:start_x + min_dim]


def sharpen(img):
    """샤프닝 필터 적용"""
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(img, -1, kernel)


def load_and_preprocess_image(image_bytes):
    # 바이트 → numpy
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("이미지 디코딩 실패")

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1️.중앙 crop (패턴 왜곡 방지)
    img = center_crop(img)

    # 2️.조명 보정
    img = apply_clahe(img)

    # 3️.샤프닝(패턴 강조)
    img = sharpen(img)

    # 4️.모델 입력 사이즈로 resize
    img = cv2.resize(img, IMG_SIZE)

    # 5️.전처리 (EfficientNet 방식)
    img = img.astype("float32")
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return np.expand_dims(img, axis=0)