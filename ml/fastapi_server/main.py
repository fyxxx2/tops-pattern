from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
from utils import load_and_preprocess_image
import os

MODEL_PATH = "C:/Users/PC/Desktop/tops-pattern/ml/models/efficientnet_fullimage.keras" #데스크탑 노트북 구별 잘하기

print(" 모델 로딩 중…")
model = tf.keras.models.load_model(MODEL_PATH)
print(" 모델 로드 완료!")

# 클래스 이름 (train_fullimage에서 사용된 순서 그대로)
CLASS_NAMES = ["floral", "plaid", "polka_dot", "solid", "stripe"]

app = FastAPI(title="Pattern Classification API")

@app.post("/predict")
async def predict_pattern(file: UploadFile = File(...)):
    # 이미지 파일 읽기
    image_bytes = await file.read()

    try:
        img = load_and_preprocess_image(image_bytes)
    except:
        return {"error": "이미지 처리 실패"}

    # 예측
    preds = model.predict(img)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))

    return {
        "prediction": CLASS_NAMES[class_id],
        "confidence": round(confidence, 4)
    }


# FastAPI 서버 실행 (터미널에서 직접 실행해야 함)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)