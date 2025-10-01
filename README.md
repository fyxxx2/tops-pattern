패턴 종류(무지, 스트라이프, 체크, 도트, 플로럴)를 분석하는 프로젝트입니다.


📊 상의 패턴 이미지 수집 통합 전략

목표: 국내 주요 쇼핑몰(허용 범위)에서 상의 패턴(무지/스트라이프/체크/도트/플로럴) 이미지를 수집·정리하고, 이후 분류 모델 학습에 활용

1️⃣ 사이트별 크롤링 가능 여부

⚠️ 주의: robots.txt 허용은 “기술적 접근 허용”을 의미합니다. 상업적 이용/재배포 등은 각 사이트 이용약관을 반드시 확인하세요.

사이트	로봇 정책 요약	비고
W컨셉	User-agent: * 허용, /error/, /publ/ 등 일부만 차단	여성 의류 비중 높음
29CM	User-agent: * Allow: /	남·여 공통 카테고리 풍부
지그재그	User-agent: * Allow: /	여성 의류, 입점 셀러 다양
에이블리	User-agent: * Allow: /, 단 /api/*, /markets/*/info 차단	사이트맵 제공
2️⃣ 데이터 수집 목표

클래스(패턴): solid, stripe, plaid(check), polka_dot, floral

목표 수량: 클래스별 최소 300장 이상

균형 확보 전략

남성 데이터 → 주로 29CM

여성 데이터 → W컨셉 + 지그재그 + 에이블리

floral/polka dot → 여성 위주 사이트에서 우선 확보

3️⃣ 단계별 진행 전략
📍 Step 1: 사이트맵/카테고리 기반 URL 수집

W컨셉 / 에이블리: sitemap.xml 제공 → 상품 상세 URL 직접 확보

29CM / 지그재그: 카테고리 페이지를 크롤링 → 상품 상세 URL 수집

📍 Step 2: 상품명 + 카테고리 필터링

상품명/카테고리에 상의 키워드 포함 시만 채택

예: shirt, t-shirt, tee, blouse, top, 맨투맨, 셔츠

패턴 키워드 매핑

stripe → 스트라이프, stripe

plaid → 체크, plaid

polka_dot → 도트, polka dot

floral → 플로럴, 꽃무늬

solid → 솔리드, 무지, plain

📍 Step 3: 대표 이미지 추출

상품 상세 페이지의 <meta property="og:image"> 먼저 사용

없으면 상세 이미지 리스트 중 최대 해상도 선택

해상도 기준: 짧은 변 ≥ 600px 이상만 저장

📍 Step 4: 데이터 저장 구조
data/
└── raw/
    └── tops/
        ├── solid/
        ├── stripe/
        ├── plaid/
        ├── polka_dot/
        └── floral/


각 클래스별 폴더에 저장

파일명 규칙: 사이트명_상품ID.jpg (예: 29cm_abc123.jpg)

📍 Step 5: 중복 방지

manifest.csv 유지 (아래 스키마 참고)

site, product_id, label, title, url, img_url, file_path, width, height, phash_hex, created_at

중복 검사: 저장 전

상품ID 중복 스킵

pHash(퍼셉추얼 해시) 해밍 거리 임계값(예: ≤ 5)으로 유사 이미지 스킵

📍 Step 6: 품질 관리

수작업 샘플링으로 오탐 제거(특히 floral의 ‘꽃사진’ 오탐)

너무 작은 썸네일 컷 제외

패턴 인식 불가한 흰 배경/색상만 보이는 컷 제외

4️⃣ 실행/운영 가이드
환경 준비

Python 3.10+ 권장

(예시) requirements.txt

requests
beautifulsoup4
pillow
imagehash
python-dotenv

크롤링 모범 규칙

요청 간격: 1–3초 랜덤 대기, 야간 대량 요청 지양

금지 경로(Disallow) 및 민감 페이지(로그인/결제/마이페이지 등) 접근 금지

User-Agent 명시, 과도한 병렬화 금지

연구/내부 용도 우선, 상업적 사용은 반드시 사전 허가/약관 확인