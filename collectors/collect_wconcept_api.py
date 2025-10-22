# collectors/collect_wconcept_api.py
# W컨셉 카테고리 API 크롤러 (+ 라벨별 검색 v2 보강, 여성전용/원피스 허용, 스킴보정)
# - 카테고리 API: 남/여 상의 순회 (기존 로직 유지)
# - 검색 v2: 라벨별 키워드/룰(여성 전용, 원피스 허용)로 복합 수집
# - 제목/본문/파일명 힌트 + (옵션) polka-dot 비전 감지
# - 이미지 URL 스킴 //... → https:// 보정
#
# 저장: data/raw/tops/<label>/wconcept_<itemCd>.jpg
# manifest: data/raw/tops/_manifest_wconcept.csv

import io, time, json, re, random, argparse
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import requests
from PIL import Image
import imagehash
import cv2
import numpy as np

# 공용 유틸
try:
    from crawler_utils import (
        SAVE_ROOT, fetch_image_bytes, ensure_min_side,
        is_apparel, load_manifest, append_manifest,
        polite_sleep, MIN_SIDE, PHASH_THRESHOLD, USE_PHASH,
        robots_allowed, HEADERS
    )
except Exception:
    # 최소 폴백(필요 최소 기능만)
    SAVE_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "tops"
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    MIN_SIDE = 400
    USE_PHASH = True
    PHASH_THRESHOLD = 6
    HEADERS = {"User-Agent": "Mozilla/5.0"}

    def fetch_image_bytes(url: str, timeout: int = 20):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.content
        except Exception:
            pass
        return None

    def ensure_min_side(img: Image.Image, min_side: int = 400) -> bool:
        return min(img.size) >= min_side

    def is_apparel(name: str) -> bool:
        if not name: return False
        t = name.lower()
        return any(k in t for k in ["shirt","tee","t-shirt","knit","sweater","cardigan","blouse","hood",
                                    "pullover","top","vest","dress","one piece","one-piece","onepiece",
                                    "셔츠","티셔츠","블라우스","니트","스웨터","가디건","후드","탑","원피스","드레스"])

    def load_manifest(path: Path):
        ids, hashes = set(), []
        if path.exists():
            import csv
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    ids.add(str(row.get("product_id","")).strip())
                    hx = row.get("phash_hex","").strip()
                    if hx:
                        hashes.append((hx, row.get("label",""), row.get("file_path","")))
        return ids, hashes

    def append_manifest(path: Path, row: Dict):
        import csv
        write_header = not path.exists()
        with open(path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "site","product_id","label","title","url","img_url","file_path",
                "width","height","phash_hex","created_at"
            ])
            if write_header:
                w.writeheader()
            w.writerow(row)

    def polite_sleep(sec: float = 0.6):
        time.sleep(sec)

    def robots_allowed(_: str) -> bool:
        return True  # 폴백에선 항상 허용

SITE = "wconcept"
MANIFEST = SAVE_ROOT / "_manifest_wconcept.csv"

SESSION = requests.Session()

# ─────────────────────────────────────────────────────────
# 카테고리 API 엔드포인트 모음
CATEGORIES = [
    # 남성 상의
    ("men-shirts", "M83985861", "001", "004"),
    ("men-tees",   "M83985861", "001", "002"),
    ("men-knit",   "M83985861", "001", "003"),
    # 여성 상의
    ("women-blouse","M33439436", "001", "004"),
    ("women-shirts","M33439436", "001", "013"),
    ("women-tees",  "M33439436", "001", "003"),
    ("women-knit",  "M33439436", "001", "002"),
]

def build_url(cat_id: str, sub1: str, sub2: Optional[str]=None) -> str:
    if sub2:
        return f"https://api-display.wconcept.co.kr/display/api/v1/category/products/{cat_id}/{sub1}/{sub2}"
    return f"https://api-display.wconcept.co.kr/display/api/v1/category/products/{cat_id}/{sub1}"

# 헤더/페이로드
DISPLAY_API_KEY = "VWmkUPgs6g2fviPZ5JQFQ3pERP4tIXv/J2jppLqSRBk="  # 필요 시 최신값 교체
SORT_MODES = ["NEW", "POPULAR", "WCK"]

def make_headers() -> Dict[str, str]:
    return {
        **HEADERS,
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json; charset=UTF-8",
        "Origin": "https://display.wconcept.co.kr",
        "Referer": "https://display.wconcept.co.kr/",
        "devicetype": "PC",
        "display-api-key": DISPLAY_API_KEY,
    }

def make_payload(page_no: int, page_size: int = 60, sort_mode: str = "WCK") -> Dict:
    return {
        "custNo": "",
        "gender": "All",
        "sort": sort_mode,
        "pageNo": page_no,
        "pageSize": page_size,
        "bcds": [],
        "colors": [],
        "benefits": [],
        "discounts": [],
        "domainType": "pc",
        "shopCds": [],
        "status": ["01"]
    }

# ─────────────────────────────────────────────────────────
# 라벨/키워드
LABELS = ["solid","stripe","plaid","polka_dot","floral"]
TOPS_TERMS = {"티셔츠","셔츠","블라우스","니트","스웨터","가디건","맨투맨","후드","탑","베스트","조끼","카디건","티","니트탑"}
TOPS_CATEGORIES_2 = {"니트","셔츠","블라우스","티셔츠","가디건","맨투맨","후드","베스트","스웨터","탑"}
TOPS_CATEGORIES_3 = {"풀오버","가디건","셔츠","블라우스","티셔츠","후드","맨투맨","베스트","니트","탑"}

PATTERN_SYNONYM = {
    "solid":     ["무지","솔리드","plain","basic","solid"],
    "stripe":    ["스트라이프","줄무늬","단가라","보더","stripe","striped","pinstripe","pin-striped","세로줄","가로줄"],
    "plaid":     ["체크","타탄","글렌체크","하운드투스","깅엄","버버리체크","플래드","check","checked","plaid","tartan","gingham","houndstooth"],
    "polka_dot": ["도트","땡땡이","물방울","polka dot","polka-dot","dot","dotted","polkadot"],
    "floral":    [
        "플로럴","플라워","꽃무늬","꽃","보태니컬","botanical","botanic","floral","flower","ditsy","bouquet",
        "리버티","liberty","로즈","rose","장미","데이지","daisy","튤립","tulip","하와이안","hawaiian","트로피컬","tropical","가든","garden"
    ],
}

def normalize_product_url(webViewUrl: Optional[str]) -> Optional[str]:
    if not webViewUrl:
        return None
    u = webViewUrl.strip()
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        return "https://www.wconcept.co.kr" + u
    if u.startswith("http"):
        return u
    return "https://www.wconcept.co.kr/" + u.lstrip("/")

def normalize_img_url(u: str) -> str:
    if not u:
        return u
    u = u.strip()
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        return "https://www.wconcept.co.kr" + u
    return u

def is_tops_by_category(cat2: str, cat3: str) -> bool:
    c2 = (cat2 or "").strip()
    c3 = (cat3 or "").strip()
    return (c2 in TOPS_CATEGORIES_2) or (c3 in TOPS_CATEGORIES_3)

def gender_hint(name: str, c2: str, c3: str) -> Optional[str]:
    blob = " ".join([(name or ""), (c2 or ""), (c3 or "")]).lower()
    if any(k in blob for k in ["여성", "women", "woman", "ladies", "girl", "femme"]):
        return "women"
    if any(k in blob for k in ["남성", "남자", "men", "man", "mens", "men's", "man’s", "맨즈"]):
        return "men"
    return None

def looks_mens(name: str, c2: str, c3: str) -> bool:
    return gender_hint(name, c2, c3) == "men"

# ─────────────────────────────────────────────────────────
# 제목/본문 기반 라벨 감지(화이트리스트 + 완화 규칙)
LABEL_SCORE_THRESHOLD = {"polka_dot": 0, "floral": 0}

FLORAL_WHITELIST = [
    "플라워 프린트","플로럴 프린트","꽃무늬 프린트","flower print","floral print",
    "플라워 패턴","플로럴 패턴","꽃 패턴","flower pattern","floral pattern",
    "플라워 자카드","floral jacquard","꽃 자카드","플라워 올오버","floral allover","all-over floral",
    "보태니컬 프린트","botanical print","리버티 프린트","liberty print","로즈 프린트","장미 프린트",
    "데이지 프린트","daisy print","튜울립 프린트","tulip print","hawaiian print","tropical print",
    "flower allover","allover flower","all over flower","all-over flower",
    "floral embroidery","flower embroidery","플라워 자수","꽃 자수"
]
DOT_WHITELIST = [
    "도트 프린트","dot print","polka dot print",
    "도트 패턴","dot pattern","polka dot pattern",
    "도트 자카드","dot jacquard","polka dot jacquard",
    "도트 올오버","allover dot","all-over dot","올오버 도트",
    "mini dot","micro dot","tiny dot","스몰 도트","마이크로 도트"
]
PATTERN_CARRIERS = ["패턴","프린트","프린팅","올오버","올 오버","전체","나염","자카드","자수","텍스타일","패브릭","패치워크"]
NEG_CONTEXT_FLORAL = ["한송이","한 송이","꽃 단추","꽃단추","꽃 버튼","브로치","코사지","플라워 포인트","꽃 포인트",
                      "자수 포인트","프린트 포인트","로고","레터링","스티치","포켓","트리밍","파이핑"]
NEG_CONTEXT_DOT = ["도트 버튼","도트 단추","버튼","단추","브로치","코사지","도트 포인트","자수 포인트",
                   "프린트 포인트","로고","레터링","스티치","포켓","파이핑","리벳"]
DOT_NEAR_NEG_RE = re.compile(r"(도트|polka ?dot|polka-dot|땡땡이).{0,6}(버튼|단추|브로치|코사지)", re.I)
FLORAL_NEAR_NEG_RE = re.compile(r"(플라워|꽃|floral|flower).{0,6}(버튼|단추|브로치|코사지|포인트)", re.I)

def _any_in(text: str, words: List[str]) -> bool:
    t = (text or "").lower()
    return any(w.lower() in t for w in words)

def detect_label_from_title(name: str) -> Optional[str]:
    if not name:
        return None
    lname = name.lower()

    if any(w.lower() in lname for w in DOT_WHITELIST):
        if not (DOT_NEAR_NEG_RE.search(lname) or _any_in(lname, NEG_CONTEXT_DOT)):
            return "polka_dot"
    if any(w.lower() in lname for w in FLORAL_WHITELIST):
        if not (FLORAL_NEAR_NEG_RE.search(lname) or _any_in(lname, NEG_CONTEXT_FLORAL)):
            return "floral"

    if _any_in(lname, PATTERN_SYNONYM["stripe"]):
        return "stripe"
    if _any_in(lname, PATTERN_SYNONYM["plaid"]):
        return "plaid"

    carriers = _any_in(lname, PATTERN_CARRIERS) or _any_in(lname, ["allover", "all-over"])

    if _any_in(lname, PATTERN_SYNONYM["polka_dot"]):
        if not (DOT_NEAR_NEG_RE.search(lname) or _any_in(lname, NEG_CONTEXT_DOT)):
            score = 1 + (1 if carriers else 0) + (1 if _any_in(lname, ["올오버","올 오버","전체","allover","all-over"]) else 0)
            if score >= LABEL_SCORE_THRESHOLD.get("polka_dot", 2):
                return "polka_dot"

    if _any_in(lname, PATTERN_SYNONYM["floral"]):
        if not (FLORAL_NEAR_NEG_RE.search(lname) or _any_in(lname, NEG_CONTEXT_FLORAL)):
            score = 1 + (1 if carriers else 0) + (1 if _any_in(lname, ["올오버","올 오버","전체","allover","all-over"]) else 0)
            if score >= LABEL_SCORE_THRESHOLD.get("floral", 2):
                return "floral"

    if _any_in(lname, PATTERN_SYNONYM["solid"]) and not carriers:
        return "solid"
    return None

# 상세페이지 본문 라이트 파싱 + 갤러리 이미지 추출
def fetch_detail_text(detail_url: str) -> str:
    try:
        if not detail_url:
            return ""
        res = SESSION.get(detail_url, headers=HEADERS, timeout=20)
        if res.status_code != 200 or not res.text:
            return ""
        html = res.text
        parts = []
        m = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        if m: parts.append(m.group(1))
        m2 = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        if m2: parts.append(m2.group(1))
        body_text = re.sub(r"<[^>]+>", " ", html)
        body_text = re.sub(r"\s+", " ", body_text).strip()
        if body_text: parts.append(body_text[:4000])
        return " ".join(parts)
    except Exception:
        return ""

def extract_gallery_images(detail_html: str) -> List[str]:
    urls = set()
    for m in re.finditer(r'"imageUrl"\s*:\s*"([^"]+)"', detail_html):
        urls.add(normalize_img_url(m.group(1)))
    for m in re.finditer(r'<img[^>]+src=["\']([^"\']+productimg[^"\']+)["\']', detail_html, re.I):
        urls.add(normalize_img_url(m.group(1)))
    for m in re.finditer(r'"imageUrlMobile"\s*:\s*"([^"]+)"', detail_html):
        urls.add(normalize_img_url(m.group(1)))
    return list(urls)[:10]

def detect_label_from_texts(title: str, detail: str) -> Optional[str]:
    lab = detect_label_from_title(title or "")
    if lab:
        return lab

    text = f"{title or ''} {detail or ''}".lower()
    if any(w.lower() in text for w in DOT_WHITELIST):
        if not (DOT_NEAR_NEG_RE.search(text) or _any_in(text, NEG_CONTEXT_DOT)):
            return "polka_dot"
    if any(w.lower() in text for w in FLORAL_WHITELIST):
        if not (FLORAL_NEAR_NEG_RE.search(text) or _any_in(text, NEG_CONTEXT_FLORAL)):
            return "floral"

    if _any_in(text, PATTERN_SYNONYM["stripe"]):
        return "stripe"
    if _any_in(text, PATTERN_SYNONYM["plaid"]):
        return "plaid"

    carriers = _any_in(text, PATTERN_CARRIERS) or _any_in(text, ["allover", "all-over"])

    if _any_in(text, PATTERN_SYNONYM["polka_dot"]):
        if not (DOT_NEAR_NEG_RE.search(text) or _any_in(text, NEG_CONTEXT_DOT)):
            score = 1 + (1 if carriers else 0) + (1 if _any_in(text, ["올오버","올 오버","전체","allover","all-over"]) else 0)
            if score >= LABEL_SCORE_THRESHOLD.get("polka_dot", 2):
                return "polka_dot"

    if _any_in(text, PATTERN_SYNONYM["floral"]):
        if not (FLORAL_NEAR_NEG_RE.search(text) or _any_in(text, NEG_CONTEXT_FLORAL)):
            score = 1 + (1 if carriers else 0) + (1 if _any_in(text, ["올오버","올 오버","전체","allover","all-over"]) else 0)
            if score >= LABEL_SCORE_THRESHOLD.get("floral", 2):
                return "floral"

    # 약한 플로럴 백업
    if any(k in (title or "") for k in ["블라우스","셔츠","티셔츠","니트","가디건"]) \
       and any(w in text for w in ["꽃","플라워","floral","flower"]) \
       and not (FLORAL_NEAR_NEG_RE.search(text) or _any_in(text, NEG_CONTEXT_FLORAL)):
        return "floral"

    if _any_in(text, PATTERN_SYNONYM["solid"]) and not carriers:
        return "solid"
    return None

def hint_label_from_media(item: dict) -> Optional[str]:
    cand = []
    for k in ("imageName", "imageUrlMobile", "webViewUrl"):
        v = (item.get(k) or "").lower()
        if v: cand.append(v)
    if not cand:
        return None
    blob = " ".join(cand)
    if any(w in blob for w in ["polka-dot","polka_dot","polkadot","polka dot","/dot","_dot","-dot"," dot "]):
        return "polka_dot"
    if any(w in blob for w in ["floral","flower","bouquet","ditsy","botanic","botanical","rose","daisy","tulip","liberty"]):
        return "floral"
    if "stripe" in blob or "striped" in blob:
        return "stripe"
    if any(w in blob for w in ["plaid","check","checked","gingham","tartan","houndstooth"]):
        return "plaid"
    if "solid" in blob or "plain" in blob:
        return "solid"
    return None

# ─────────────────────────────────────────────────────────
# polka-dot 간단 비전 감지(선택적)
def _density(count: int, area: int) -> float:
    if area <= 0: return 0.0
    return float(count) / float(area)

def looks_polkadot(img_bgr: np.ndarray) -> bool:
    try:
        h, w = img_bgr.shape[:2]
        area = h * w
        if min(h, w) < 400:
            return False

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 31, 3)
        thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(thr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        dots = 0
        center_hits = 0
        cx0, cy0 = int(w*0.2), int(h*0.2)
        cx1, cy1 = int(w*0.8), int(h*0.8)

        for c in cnts:
            a = cv2.contourArea(c)
            if a < 8 or a > area * 0.004:
                continue
            perim = cv2.arcLength(c, True)
            if perim == 0:
                continue
            circularity = 4 * np.pi * (a / (perim * perim))
            if circularity > 0.45:
                dots += 1
                M = cv2.moments(c)
                if M["m00"] != 0:
                    x = int(M["m10"]/M["m00"]); y = int(M["m01"]/M["m00"])
                    if (cx0 <= x <= cx1) and (cy0 <= y <= cy1):
                        center_hits += 1

        if dots < 8: return False
        if _density(dots, area) < 6e-5: return False
        if center_hits < 5: return False
        return True
    except Exception:
        return False

def looks_polkadot_blob(img_bgr: np.ndarray) -> bool:
    try:
        h, w = img_bgr.shape[:2]
        if min(h, w) < 400:
            return False
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 6
        params.maxArea = max(80, (h*w) * 0.004)
        params.filterByCircularity = True
        params.minCircularity = 0.55
        params.filterByInertia = False
        params.filterByConvexity = False
        params.filterByColor = False

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(gray)
        if len(keypoints) < 8:
            return False

        cx0, cy0 = int(w*0.2), int(h*0.2)
        cx1, cy1 = int(w*0.8), int(h*0.8)
        center_hits = 0
        for k in keypoints:
            x, y = int(k.pt[0]), int(k.pt[1])
            if cx0 <= x <= cx1 and cy0 <= y <= cy1:
                center_hits += 1
        return center_hits >= 5
    except Exception:
        return False

# ─────────────────────────────────────────────────────────
def fetch_page(base_url: str, page_no: int, page_size: int=60, sort_mode: str="WCK") -> Optional[dict]:
    try:
        payload = make_payload(page_no, page_size, sort_mode=sort_mode)
        r = SESSION.post(base_url, headers=make_headers(), data=json.dumps(payload), timeout=20)
        if r.status_code != 200:
            print(f"[http {r.status_code}] {base_url}")
            return None
        return r.json()
    except Exception as e:
        print("[fetch error]", e)
        return None

def iterate_category_random(base_url: str, max_pages:int=10, page_size:int=60, sort_mode:str="WCK") -> List[dict]:
    out: List[dict] = []
    first = fetch_page(base_url, page_no=1, page_size=page_size, sort_mode=sort_mode)
    if not first:
        print(f"[probe] 첫 페이지 로드 실패: {base_url} ({sort_mode})")
        return out
    try:
        pl = first.get("data", {}).get("productList", {})
        total_pages = int(pl.get("totalPages", 1))
        content = pl.get("content", []) or []
    except Exception:
        print(f"[probe] 응답 파싱 실패: {base_url} ({sort_mode}), keys={list(first.keys())}")
        return out

    out.extend(content)
    print(f"[api][{sort_mode}] p1/{total_pages} → {len(content)} items")

    if total_pages > 1 and max_pages > 1:
        candidates = list(range(2, total_pages+1))
        random.shuffle(candidates)
        take = min(len(candidates), max_pages-1)
        pick_pages = sorted(candidates[:take])

        prev_ids = {str(i.get("itemCd")) for i in content if i.get("itemCd")}
        repeat = 0

        for p in pick_pages:
            polite_sleep()
            d = fetch_page(base_url, page_no=p, page_size=page_size, sort_mode=sort_mode)
            if not d:
                print(f"[api][{sort_mode}] p{p} 실패 → 건너뜀")
                continue
            pl = d.get("data", {}).get("productList", {})
            content = pl.get("content", []) or []
            print(f"[api][{sort_mode}] p{p}/{total_pages} → {len(content)} items")
            out.extend(content)

            curr_ids = {str(i.get("itemCd")) for i in content if i.get("itemCd")}
            if curr_ids and prev_ids and len(curr_ids & prev_ids) / max(len(curr_ids), 1) >= 0.8:
                repeat += 1
            else:
                repeat = 0
            prev_ids = curr_ids

            if repeat >= 2:
                print(f"[api][{sort_mode}] 중복률 높음 → p{p}에서 이 정렬 모드 조기 종료")
                break
    return out

# 스킵 카운터
class SkipStats:
    def __init__(self):
        self.c: Dict[str, int] = {}
    def add(self, reason: str):
        self.c[reason] = self.c.get(reason, 0) + 1
    def dump(self):
        print("[skip-stats]", self.c)

# ─────────────────────────────────────────────────────────
# 검색 v2(여성/원피스 허용 설정)
SEARCH_ENDPOINT_V2 = "https://api-display.wconcept.co.kr/display/api/v2/search/result/product"
SEARCH_CONFIG: Dict[str, Dict] = {
    "floral": {
        "keywords": ["floral", "floral print", "flower print", "플로럴", "플라워", "꽃무늬", "보태니컬", "liberty"],
        "gender": "women",
        "allow_dress": True,
    },
    "polka_dot": {
        "keywords": ["polka dot", "polkadot", "dot print", "도트"],
        "gender": "women",
        "allow_dress": True,
    },
}
SEARCH_LIST_PATHS = [
    ["data", "list"], ["data", "products"], ["content"], ["products"], ["result"],
    ["data", "productList", "content"],
]
SEARCH_ID_KEYS   = ["itemCd", "productId", "goodsId", "goodsNo", "id"]
SEARCH_NAME_KEYS = ["itemName", "productName", "goodsName", "name", "title"]
SEARCH_IMG_KEYS  = ["imageUrlMobile", "imageUrl", "representImageUrl", "imgUrl", "thumbnailUrl"]
SEARCH_CAT2_KEYS = ["categoryDepthName2", "category2"]
SEARCH_CAT3_KEYS = ["categoryDepthName3", "category3"]
DRESS_TERMS = {"원피스", "ONE PIECE", "ONE-PIECE", "ONEPIECE", "DRESS"}

def pick_first(d, keys):
    for k in keys:
        v = d.get(k)
        if v: return v
    return None

def get_by_path(d, path):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

def extract_list_from_search_resp(resp: dict):
    if not isinstance(resp, dict):
        return []
    for path in SEARCH_LIST_PATHS:
        lst = get_by_path(resp, path)
        if isinstance(lst, list):
            return lst
    return []

def parse_search_item(raw: dict) -> dict:
    item = {}
    item["itemCd"] = str(pick_first(raw, SEARCH_ID_KEYS) or "").strip()
    item["itemName"] = pick_first(raw, SEARCH_NAME_KEYS) or ""
    item["imageUrlMobile"] = normalize_img_url(pick_first(raw, SEARCH_IMG_KEYS) or "")
    item["categoryDepthName2"] = pick_first(raw, SEARCH_CAT2_KEYS) or ""
    item["categoryDepthName3"] = pick_first(raw, SEARCH_CAT3_KEYS) or ""
    item["webViewUrl"] = normalize_product_url(raw.get("webViewUrl") or raw.get("detailUrl") or "")
    return item

def is_dress(name: str, c2: str, c3: str) -> bool:
    hay = " ".join([name or "", c2 or "", c3 or ""]).upper()
    return any(term in hay for term in DRESS_TERMS)

def make_search_payload(keyword: str, page_no: int, page_size: int) -> dict:
    return {
        "custNo": "",
        "gender": "all",        # 결과는 all로 받고 후처리
        "keyword": keyword,
        "sort": "WCK",
        "pageNo": page_no,
        "pageSize": page_size,
        "bcds": [],
        "colors": [],
        "benefits": [],
        "discounts": [],
        "device": "PC",
        "lcds": [],
        "searchType": "recent",
        "source": "Rn/Women",
        "status": ["01"],
    }

def fetch_search_page_v2(keyword: str, page_no: int, page_size: int = 60) -> Optional[dict]:
    payload = make_search_payload(keyword, page_no, page_size)
    headers = make_headers()
    try:
        r = SESSION.post(SEARCH_ENDPOINT_V2, headers=headers, data=json.dumps(payload), timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[search v2 http {r.status_code}] {SEARCH_ENDPOINT_V2}")
            return None
    except Exception as e:
        print("[search v2 fetch error]", e)
        return None

def iterate_search_v2(keyword: str, max_pages: int = 8, page_size: int = 60) -> List[dict]:
    out = []
    for p in range(1, max_pages + 1):
        polite_sleep()
        resp = fetch_search_page_v2(keyword, p, page_size)
        if not resp:
            if p == 1:
                print(f"[search v2] '{keyword}' p{p} 응답 없음 → 중단")
            break
        items = extract_list_from_search_resp(resp)
        if not items:
            top = list(resp.keys()) if isinstance(resp, dict) else type(resp)
            data_keys = list((resp.get("data") or {}).keys()) if isinstance(resp, dict) and isinstance(resp.get("data"), dict) else None
            print(f"[search v2] '{keyword}' p{p} 리스트 비어있음 → 중단 (top={top}, data_keys={data_keys})")
            break
        print(f"[search v2] '{keyword}' p{p} → {len(items)} items")
        out.extend(items)
    return out

# 공통 저장 파이프라인
def try_save_item(item: dict,
                  detected: Optional[str],
                  known_ids: set,
                  known_hashes: List[Tuple[str,str,str]],
                  saved: Dict[str,int],
                  stats: SkipStats,
                  labels_filter=None) -> bool:

    itemCd = str(item.get("itemCd") or "").strip()
    if not itemCd:
        stats.add("no_id"); return False
    name = (item.get("itemName") or "").strip()

    detail_url = normalize_product_url(item.get("webViewUrl") or "")
    if not detail_url or not robots_allowed(detail_url):
        stats.add("robots_block"); return False

    gallery_urls = []
    try:
        res = SESSION.get(detail_url, headers=HEADERS, timeout=20)
        if res.status_code == 200 and res.text:
            gallery_urls = extract_gallery_images(res.text)
    except Exception:
        pass

    first_img = normalize_img_url(item.get("imageUrlMobile") or "")
    candidate_img_urls = ([first_img] if first_img else []) + gallery_urls
    candidate_img_urls = [u for u in candidate_img_urls if u]

    raw = None
    picked_url = None
    for u in candidate_img_urls:
        raw_try = fetch_image_bytes(u)
        if not raw_try:
            continue

        if not detected:
            try:
                npimg = np.frombuffer(raw_try, dtype=np.uint8)
                img_bgr = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                if img_bgr is not None and (looks_polkadot(img_bgr) or looks_polkadot_blob(img_bgr)):
                    detected = "polka_dot"
            except Exception:
                pass

        if not detected:
            fn_hint = u.split("/")[-1].lower()
            if any(w in fn_hint for w in ["polka-dot","polka_dot","polkadot","polka dot","_dot","-dot"," dot "]):
                detected = "polka_dot"
            elif any(w in fn_hint for w in ["floral","flower","bouquet","ditsy","botanic","botanical","rose","daisy","tulip","liberty"]):
                detected = "floral"

        if detected:
            raw = raw_try
            picked_url = u
            break

    if not detected:
        stats.add("no_pattern"); return False
    if labels_filter and detected not in labels_filter:
        stats.add("label_filter"); return False
    if not raw:
        stats.add("img_fetch_fail"); return False

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        stats.add("img_open_fail"); return False
    if not ensure_min_side(img, MIN_SIDE):
        stats.add("too_small"); return False

    phx = imagehash.phash(img).__str__() if USE_PHASH else None
    if phx and any(bin(int(phx,16)^int(hx,16)).count("1") <= PHASH_THRESHOLD for hx,_,_ in known_hashes):
        stats.add("phash_dup"); return False

    out_dir = SAVE_ROOT / detected
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"wconcept_{itemCd}.jpg"
    out_path = out_dir / fname
    with open(out_path, "wb") as f:
        f.write(raw)

    append_manifest(MANIFEST, {
        "site": SITE,
        "product_id": itemCd,
        "label": detected,
        "title": name,
        "url": detail_url,
        "img_url": picked_url or first_img,
        "file_path": str(out_path.as_posix()),
        "width": img.size[0], "height": img.size[1],
        "phash_hex": phx or "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    if phx:
        known_hashes.append((phx, detected, str(out_path)))
    saved[detected] = saved.get(detected, 0) + 1
    print(f"  ✔ {SITE} | {detected} | {name[:60]}... → {fname} ({saved[detected]})")
    return True

# 메인 수집
def collect_api(per_label_target=20,
                labels_filter=None,
                max_pages=10,
                search_pages=8,
                use_category_for=("solid","stripe","plaid"),
                use_search_for=("floral","polka_dot"),
                search_config: Dict[str,Dict]=None):

    if search_config is None:
        search_config = SEARCH_CONFIG

    known_ids, known_hashes = load_manifest(MANIFEST)
    print(f"[{SITE}] manifest ids={len(known_ids)}, phash={len(known_hashes)}")

    saved = {lb: 0 for lb in LABELS}
    seen = set(known_ids)
    stats = SkipStats()

    # 1) 카테고리 API (원하면 비활성화 가능)
    if use_category_for:
        for cat_name, cat_id, sub1, sub2 in CATEGORIES:
            base_url = build_url(cat_id, sub1, sub2)
            print(f"[use] endpoint={base_url} ({cat_name})")

            contents_all: List[dict] = []
            for sort_mode in SORT_MODES:
                contents = iterate_category_random(base_url, max_pages=max_pages, page_size=60, sort_mode=sort_mode)
                contents_all.extend(contents)
                if sum(saved.values()) >= 3 * per_label_target:
                    break

            for item in contents_all:
                if all(saved.get(lb,0) >= per_label_target for lb in use_category_for):
                    break

                itemCd = str(item.get("itemCd") or "").strip()
                if not itemCd or itemCd in seen:
                    continue
                seen.add(itemCd)

                name = (item.get("itemName") or "").strip()
                cat2 = item.get("categoryDepthName2") or ""
                cat3 = item.get("categoryDepthName3") or ""
                if not (is_tops_by_category(cat2, cat3) or any(t in name for t in TOPS_TERMS)):
                    stats.add("not_tops"); continue
                if not is_apparel(name):
                    stats.add("not_apparel"); continue

                detail_url = normalize_product_url(item.get("webViewUrl"))
                if not detail_url or not robots_allowed(detail_url):
                    stats.add("robots_block"); continue

                detail_text = fetch_detail_text(detail_url)
                detected = detect_label_from_texts(name, detail_text) or hint_label_from_media(item)

                if not detected or detected not in use_category_for:
                    stats.add("cat_label_skip"); continue
                if saved.get(detected,0) >= per_label_target:
                    stats.add("label_full"); continue

                try_save_item(item, detected, known_ids, known_hashes, saved, stats, labels_filter)

    # 2) 검색 v2 (여성 전용/원피스 허용 설정 포함)
    for label in use_search_for:
        cfg = search_config.get(label, {})
        kws = cfg.get("keywords", [])
        target_gender = cfg.get("gender", "all").lower()
        allow_dress = bool(cfg.get("allow_dress", False))
        if not kws:
            continue

        print(f"\n[search v2 mode] label={label} gender={target_gender} allow_dress={allow_dress} → keywords={kws}")
        for kw in kws:
            if saved.get(label,0) >= per_label_target:
                break

            raw_items = iterate_search_v2(kw, max_pages=search_pages, page_size=60)
            if not raw_items:
                continue

            for raw_it in raw_items:
                if saved.get(label,0) >= per_label_target:
                    break

                item = parse_search_item(raw_it)
                itemCd = item["itemCd"]
                if not itemCd or itemCd in seen:
                    continue
                seen.add(itemCd)

                name = (item["itemName"] or "").strip()
                c2 = item.get("categoryDepthName2") or ""
                c3 = item.get("categoryDepthName3") or ""

                if target_gender == "women" and looks_mens(name, c2, c3):
                    stats.add("search_mens_cut"); continue

                tops_ok = is_tops_by_category(c2, c3) or any(t in name for t in TOPS_TERMS)
                dress_ok = allow_dress and is_dress(name, c2, c3)
                if not (tops_ok or dress_ok):
                    stats.add("search_not_tops_dress"); continue
                if not is_apparel(name):
                    stats.add("search_not_apparel"); continue

                detected = detect_label_from_title(name) or hint_label_from_media(item) or label

                if labels_filter and detected not in labels_filter:
                    stats.add("search_label_filter"); continue
                if saved.get(detected, 0) >= per_label_target:
                    stats.add("search_label_full"); continue

                try_save_item(item, detected, known_ids, known_hashes, saved, stats, labels_filter)

    print("\n[done] 카테고리 + 검색(v2) 복합 수집 완료. 저장:", SAVE_ROOT.as_posix())
    print("       manifest:", MANIFEST.as_posix())
    print("       per-label:", saved)
    stats.dump()

# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-label", type=int, default=20, help="라벨별 목표 수")
    ap.add_argument("--cat-pages", type=int, default=30, help="카테고리 탐색 페이지")
    ap.add_argument("--search-pages", type=int, default=10, help="검색 탐색 페이지")
    ap.add_argument("--no-category", action="store_true", help="카테고리 수집 비활성화")
    args = ap.parse_args()

    collect_api(
        per_label_target=args.per_label,
        labels_filter=None,
        max_pages=args.cat_pages,
        search_pages=args.search_pages,
        use_category_for=() if args.no_category else ("solid","stripe","plaid"),
        use_search_for=("floral","polka_dot"),
        search_config=SEARCH_CONFIG
    )
