import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)

def load_and_preprocess_image(image_bytes):
    # 이미지 바이트 → NumPy 배열 변환
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("이미지 디코딩 실패")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return np.expand_dims(img, axis=0)