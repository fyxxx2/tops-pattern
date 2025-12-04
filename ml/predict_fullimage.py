import tensorflow as tf
import numpy as np
import cv2
import argparse

IMG_SIZE = (224, 224)

# 클래스 이름 (학습 시 자동으로 생성된 순서와 동일해야 함)
CLASS_NAMES = ['floral', 'plaid', 'polka_dot', 'solid', 'stripe']

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return np.expand_dims(img, axis=0)

def predict(path, model_path):
    model = tf.keras.models.load_model(model_path)
    img = load_image(path)

    pred = model.predict(img)[0]
    idx = np.argmax(pred)
    label = CLASS_NAMES[idx]
    confidence = pred[idx]

    print(f"\n📌 예측 결과: {label}")
    print(f"📌 Confidence: {confidence:.4f}")
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str,
                        default="C:/Users/rlarh/Desktop/tops-pattern/ml/models/efficientnet_fullimage.keras")
    args = parser.parse_args()

    predict(args.image, args.model)