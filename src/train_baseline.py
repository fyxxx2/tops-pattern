# ====== 필요한 라이브러리 불러오기 ======
# os, pathlib: 파일/폴더 경로 다루기
import os
from pathlib import Path

# numpy: 수치 계산 라이브러리 (배열, 벡터, 행렬 연산)
import numpy as np

# cv2 (OpenCV): 이미지 불러오기, 전처리, 필터링에 사용
import cv2

# skimage.feature: 이미지에서 특징(패턴)을 뽑아내는 함수들이 있음
# local_binary_pattern → 작은 텍스처 패턴(점, 얼룩)
# greycomatrix / greycoprops → 명암 대비, 질감 같은 특성 추출
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops

# scikit-learn: 머신러닝 라이브러리
# - train_test_split → 학습/테스트 데이터 분리
# - SVC → SVM(Support Vector Machine) 분류기
# - classification_report → 정밀도, 재현율, F1-score 출력
# - confusion_matrix → 어떤 클래스에서 잘못 예측했는지 행렬로 보여줌
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# joblib: 학습한 모델을 파일로 저장하거나 불러올 때 사용
from joblib import dump


# ====== 설정 ======
# 구분하려는 5가지 패턴 클래스
CLASSES = ["solid", "stripe", "plaid", "polka_dot", "floral"]

# 데이터가 들어있는 폴더 경로
DATA_DIR = Path("data/raw/tops")

# 학습된 모델을 저장할 위치
MODEL_PATH = Path("models/baseline_svm.joblib")

# 이미지 리사이즈할 크기
IMG_SIZE = 256


# ====== 유틸 함수 ======
def load_image(path, size=IMG_SIZE):
    """
    이미지를 불러와서:
    1. 짧은 변을 256픽셀로 리사이즈
    2. 중앙 부분을 크롭해서 정사각형 만들기
    3. 흑백(그레이스케일)으로 변환
    """
    img = cv2.imread(str(path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    # 짧은 변을 기준으로 비율 유지하면서 리사이즈
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 중앙 크롭
    x0 = (new_w - size) // 2
    y0 = (new_h - size) // 2
    img = img[y0:y0+size, x0:x0+size]
    
    # 흑백으로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def feat_gabor(gray):
    """
    Gabor 필터:
    - 줄무늬 방향과 같은 "방향성 있는 패턴"을 잘 잡아냄
    - 여러 방향(0,45,90,135도)과 여러 주파수(줄 간격)로 필터링
    - 결과값의 평균, 표준편차를 특징으로 사용
    """
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    freqs = [0.1, 0.2]
    feats = []
    for th in thetas:
        for f in freqs:
            kernel = cv2.getGaborKernel((21, 21), 4.0, th, 1.0/f, 0.5, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
            feats.extend([resp.mean(), resp.std()])
    return np.array(feats, dtype=np.float32)


def feat_lbp(gray):
    """
    Local Binary Pattern (LBP):
    - 픽셀 주변 8개 점을 비교해서 텍스처(점, 얼룩, 작은 패턴)를 코드로 바꿈
    - 도트, 작은 무늬 같은 걸 잘 구분
    - 히스토그램(빈도수 분포)을 특징으로 사용
    """
    P, R = 8, 1
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, P+3), range=(0, P+2), density=True)
    return hist.astype(np.float32)


def feat_glcm(gray):
    """
    Gray Level Co-occurrence Matrix (GLCM):
    - 픽셀 쌍의 관계(예: 옆 픽셀과 밝기 차이)를 이용해서 '질감(texture)'을 나타냄
    - contrast(대비), homogeneity(균일성), energy(에너지), correlation(상관성) 등 계산
    """
    reduced = (gray / 32).astype(np.uint8)  # 밝기를 0~7 수준으로 축소
    glcm = greycomatrix(reduced, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=8, symmetric=True, normed=True)
    props = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = greycoprops(glcm, prop).ravel()
        props.extend(vals.tolist())
    return np.array(props, dtype=np.float32)


def extract_feature(gray):
    """
    Gabor + LBP + GLCM 특징을 모두 합쳐서 하나의 벡터로 만듦
    """
    g = feat_gabor(gray)
    l = feat_lbp(gray)
    m = feat_glcm(gray)
    return np.concatenate([g, l, m], axis=0)


def load_dataset():
    """
    데이터셋을 불러와서 X(특징), y(레이블)로 반환
    """
    X, y = [], []
    for idx, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / cls
        if not cls_dir.exists():
            print(f"[경고] 폴더 없음: {cls_dir}")
            continue
        for p in cls_dir.glob("*.*"):
            gray = load_image(p)
            if gray is None:
                continue
            X.append(extract_feature(gray))
            y.append(idx)
    return np.array(X), np.array(y)


# ====== 메인 실행부 ======
if __name__ == "__main__":
    # 데이터 불러오기
    X, y = load_dataset()
    if len(X) == 0:
        raise SystemExit("데이터가 없습니다. data/raw/tops/* 폴더에 이미지를 넣어주세요.")

    # 학습/테스트 데이터 분리 (8:2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # SVM 분류기 학습
    clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    clf.fit(X_tr, y_tr)

    # 테스트 데이터로 평가
    y_pred = clf.predict(X_te)
    print("\n=== 분류 리포트 ===")
    print(classification_report(y_te, y_pred, target_names=CLASSES, digits=4))
    print("=== 혼동 행렬 ===")
    print(confusion_matrix(y_te, y_pred))

    # 모델 저장
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump({"model": clf, "classes": CLASSES}, MODEL_PATH)
    print(f"\n모델 저장 완료: {MODEL_PATH}")