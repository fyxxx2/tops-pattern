import os
import pandas as pd

# 원본 makesense CSV 파일
csv_path = "C:/Users/rlarh/Desktop/tops-pattern/data/annotations/labels.csv"
df = pd.read_csv(csv_path)

# 이미지가 실제로 존재하는 폴더
base_path = "C:/Users/rlarh/Desktop/tops-pattern/data/raw/tops"

# 새로운 경로 저장 변수
correct_paths = []
exists = []

for _, row in df.iterrows():
    image = row["image_name"]      # ex) asos_203232535.jpg
    label = row["label_name"]      # ex) floral, plaid, solid ...

    # 올바른 파일 경로 구성
    full_path = os.path.join(base_path, label, image)

    correct_paths.append(full_path)
    exists.append(os.path.exists(full_path))

df["filepath"] = correct_paths
df["exists"] = exists

# 저장
df.to_csv("C:/Users/rlarh/Desktop/tops-pattern/data/annotations/pattern_dataset.csv", index=False)

print("완료!")
print(df["exists"].value_counts())