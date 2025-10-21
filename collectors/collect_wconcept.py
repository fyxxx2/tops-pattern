# collectors/collect_wconcept_api.py
# W컨셉 카테고리 API 크롤러 (남/여 상의 여러 카테고리 순회 + 실측 Payload)
# 저장: data/raw/tops/<label>/wconcept_<itemCd>.jpg
# manifest: data/raw/tops/_manifest_wconcept.csv
import io, time, json, re
from typing import Optional, Dict, List, Tuple
import requests
from PIL import Image
import imagehash

from crawler_utils import (
    SAVE_ROOT, fetch_image_bytes, ensure_min_side,
    is_apparel, load_manifest, append_manifest,
    polite_sleep, MIN_SIDE, PHASH_THRESHOLD, USE_PHASH,
    robots_allowed, HEADERS
)

SITE = "wconcept"
MANIFEST = SAVE_ROOT / "_manifest_wconcept.csv"

# ─────────────────────────────────────────────────────────
# 0) 카테고리 목록 (네가 준 URL들 반영)  /display/api/v1/category/products/<cat>/<sub1>/<sub2>
# name, cat_id, sub1, sub2
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
    # ("women-dress","M33439436","001","005"),  # 원피스(상의 아님) 제외
]

def build_url(cat_id: str, sub1: str, sub2: Optional[str]=None) -> str:
    if sub2:
        return f"https://api-display.wconcept.co.kr/display/api/v1/category/products/{cat_id}/{sub1}/{sub2}"
    return f"https://api-display.wconcept.co.kr/display/api/v1/category/products/{cat_id}/{sub1}"

# ─────────────────────────────────────────────────────────
# 1) 헤더/페이로드 (DevTools에서 캡처한 값 사용)
DISPLAY_API_KEY = "VWmkUPgs6g2fviPZ5JQFQ3pERP4tIXv/J2jppLqSRBk="  # 최신값 필요 시 교체

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

def make_payload(page_no: int, page_size: int = 60) -> Dict:
    # 네가 캡처한 구조 그대로
    return {
        "custNo": "",
        "gender": "All",
        "sort": "WCK",          # 정렬: WCK(사이트 기본값). 필요시 NEW/POPULAR 등 교체
        "pageNo": page_no,      # <== 중요: pageNo
        "pageSize": page_size,  # <== 중요: pageSize
        "bcds": [],
        "colors": [],
        "benefits": [],
        "discounts": [],
        "domainType": "pc",
        "shopCds": [],
        "status": ["01"]        # 판매중
    }

# ─────────────────────────────────────────────────────────
# 2) 상의/패턴 설정
LABELS = ["solid","stripe","plaid","polka_dot","floral"]
TOPS_TERMS = {"티셔츠","셔츠","블라우스","니트","스웨터","가디건","맨투맨","후드","탑","베스트","조끼","카디건","티","니트탑"}
TOPS_CATEGORIES_2 = {"니트","셔츠","블라우스","티셔츠","가디건","맨투맨","후드","베스트","스웨터","탑"}
TOPS_CATEGORIES_3 = {"풀오버","가디건","셔츠","블라우스","티셔츠","후드","맨투맨","베스트","니트","탑"}

PATTERN_SYNONYM = {
    "solid":     ["무지","솔리드","plain","basic","solid"],
    "stripe":    ["스트라이프","줄무늬","단가라","보더","stripe","striped","pinstripe","pin-striped","세로줄","가로줄"],
    "plaid":     ["체크","타탄","글렌체크","하운드투스","깅엄","버버리체크","플래드","check","checked","plaid","tartan","gingham","houndstooth"],
    "polka_dot": ["도트","땡땡이","물방울","polka dot","polka-dot","dot","dotted","polkadot"],
    "floral":    ["플로럴","플라워","꽃무늬","꽃","보타닉","botanic","floral","flower","ditsy","bouquet"],
}

def normalize_product_url(webViewUrl: Optional[str]) -> Optional[str]:
    if not webViewUrl:
        return None
    u = webViewUrl.strip()
    if u.startswith("/"):
        return "https://www.wconcept.co.kr" + u
    if u.startswith("http"):
        return u
    return "https://www.wconcept.co.kr/" + u.lstrip("/")

def is_tops_by_category(cat2: str, cat3: str) -> bool:
    c2 = (cat2 or "").strip()
    c3 = (cat3 or "").strip()
    return (c2 in TOPS_CATEGORIES_2) or (c3 in TOPS_CATEGORIES_3)

# ─────────────────────────────────────────────────────────
# 2-1) 제목기반 라벨 감지 정확도 강화 (floral/polka_dot 과검 방지)
PATTERN_CARRIERS = [
    "패턴", "프린트", "프린팅", "올오버", "올 오버", "전체",
    "나염", "자카드", "자수", "텍스타일", "패브릭", "패치워크"
]
NEG_CONTEXT_FLORAL = [
    "한송이", "한 송이", "꽃 단추", "꽃단추", "꽃 버튼", "브로치", "코사지",
    "플라워 포인트", "꽃 포인트", "자수 포인트", "프린트 포인트", "로고", "레터링",
    "스티치", "포켓", "트리밍", "파이핑"
]
NEG_CONTEXT_DOT = [
    "도트 버튼", "도트 단추", "버튼", "단추", "브로치", "코사지",
    "도트 포인트", "자수 포인트", "프린트 포인트", "로고", "레터링",
    "스티치", "포켓", "파이핑", "리벳"
]
DOT_NEAR_NEG_RE = re.compile(r"(도트|polka ?dot|polka-dot|땡땡이).{0,6}(버튼|단추|브로치|코사지)", re.IGNORECASE)
FLORAL_NEAR_NEG_RE = re.compile(r"(플라워|꽃|floral|flower).{0,6}(버튼|단추|브로치|코사지|포인트)", re.IGNORECASE)

def _any_in(name: str, words: List[str]) -> bool:
    ln = name.lower()
    return any(w.lower() in ln for w in words)

def detect_label_from_title(name: str) -> Optional[str]:
    """제목 문자열만으로 라벨 추정 (정확도 강화판)"""
    if not name:
        return None
    lname = name.lower()

    # stripe / plaid → 상대적으로 오검이 적으므로 관대
    if _any_in(lname, PATTERN_SYNONYM["stripe"]):
        return "stripe"
    if _any_in(lname, PATTERN_SYNONYM["plaid"]):
        return "plaid"

    carriers = _any_in(lname, PATTERN_CARRIERS) or _any_in(lname, ["allover", "all-over"])

    # polka_dot (장식 문맥 제외 + 점수제)
    if _any_in(lname, PATTERN_SYNONYM["polka_dot"]):
        if DOT_NEAR_NEG_RE.search(name) or _any_in(lname, NEG_CONTEXT_DOT):
            pass
        else:
            score = 1
            if carriers: score += 1
            if _any_in(lname, ["올오버", "올 오버", "전체", "allover", "all-over"]): score += 1
            if score >= 2:
                return "polka_dot"

    # floral (장식 문맥 제외 + 점수제)
    if _any_in(lname, PATTERN_SYNONYM["floral"]):
        if FLORAL_NEAR_NEG_RE.search(name) or _any_in(lname, NEG_CONTEXT_FLORAL):
            pass
        else:
            score = 1
            if carriers: score += 1
            if _any_in(lname, ["올오버", "올 오버", "전체", "allover", "all-over"]): score += 1
            if score >= 2:
                return "floral"

    # solid → 가장 마지막에만 (다른 패턴 신호 없을 때)
    if _any_in(lname, PATTERN_SYNONYM["solid"]) and not carriers:
        return "solid"

    return None

# ─────────────────────────────────────────────────────────
# 3) 페이지 순회(중복 페이지 감지로 조기 종료)
def fetch_page(base_url: str, page_no: int, page_size: int=60) -> Optional[dict]:
    try:
        r = requests.post(base_url, headers=make_headers(), data=json.dumps(make_payload(page_no, page_size)), timeout=20)
        if r.status_code != 200:
            print(f"[http {r.status_code}] {base_url}")
            return None
        return r.json()
    except Exception as e:
        print("[fetch error]", e)
        return None

def iterate_category(base_url: str, max_pages:int=10, page_size:int=60) -> List[dict]:
    out: List[dict] = []
    first = fetch_page(base_url, page_no=1, page_size=page_size)
    if not first:
        print(f"[probe] 첫 페이지 로드 실패: {base_url}")
        return out

    pl = first.get("data", {}).get("productList", {})
    total_pages = int(pl.get("totalPages", 1))
    content = pl.get("content", []) or []
    out.extend(content)
    print(f"[api] p1/{total_pages} → {len(content)} items")

    prev_ids = {str(i.get("itemCd")) for i in content if i.get("itemCd")}
    same_page_repeat = 0

    last_page = min(total_pages, max_pages)
    for p in range(2, last_page + 1):
        polite_sleep()
        d = fetch_page(base_url, page_no=p, page_size=page_size)
        if not d:
            print(f"[api] p{p} 실패 → 중단")
            break
        pl = d.get("data", {}).get("productList", {})
        content = pl.get("content", []) or []
        print(f"[api] p{p}/{total_pages} → {len(content)} items")
        out.extend(content)

        curr_ids = {str(i.get("itemCd")) for i in content if i.get("itemCd")}
        # 연속 페이지가 80% 이상 동일하면 2회 반복 시 중단
        if curr_ids and prev_ids and len(curr_ids & prev_ids) / max(len(curr_ids), 1) >= 0.8:
            same_page_repeat += 1
        else:
            same_page_repeat = 0
        prev_ids = curr_ids
        if same_page_repeat >= 2:
            print(f"[api] 연속 페이지 중복률 높음 → p{p}에서 조기 종료")
            break

    return out

# ─────────────────────────────────────────────────────────
# 4) 스킵 카운터
class SkipStats:
    def __init__(self):
        self.c: Dict[str, int] = {}
    def add(self, reason: str):
        self.c[reason] = self.c.get(reason, 0) + 1
    def dump(self):
        print("[skip-stats]", self.c)

# ─────────────────────────────────────────────────────────
# 5) 메인 수집
def collect_api(per_label_target=20, labels_filter=None, max_pages=10):
    known_ids, known_hashes = load_manifest(MANIFEST)
    print(f"[{SITE}] manifest ids={len(known_ids)}, phash={len(known_hashes)}")

    saved = {lb: 0 for lb in LABELS}
    seen = set()
    stats = SkipStats()

    for cat_name, cat_id, sub1, sub2 in CATEGORIES:
        base_url = build_url(cat_id, sub1, sub2)
        print(f"[use] endpoint={base_url} ({cat_name})")

        contents = iterate_category(base_url, max_pages=max_pages, page_size=60)
        if not contents:
            continue

        for item in contents:
            if all(saved[lb] >= per_label_target for lb in LABELS):
                break

            itemCd = str(item.get("itemCd") or "").strip()
            if not itemCd:
                stats.add("no_id"); continue
            if itemCd in seen:
                stats.add("dup_id"); continue
            seen.add(itemCd)

            name = (item.get("itemName") or "").strip()
            cat2 = item.get("categoryDepthName2") or ""
            cat3 = item.get("categoryDepthName3") or ""
            if not (is_tops_by_category(cat2, cat3) or any(t in name for t in TOPS_TERMS)):
                stats.add("not_tops"); continue
            if not is_apparel(name):
                stats.add("not_apparel"); continue

            # ✅ 제목에서 패턴 감지 (정확도 강화판 사용)
            detected = detect_label_from_title(name)
            if not detected:
                stats.add("no_pattern"); continue
            if labels_filter and detected not in labels_filter:
                stats.add("label_filter"); continue
            if saved[detected] >= per_label_target:
                stats.add("label_full"); continue

            detail_url = normalize_product_url(item.get("webViewUrl"))
            img_url = item.get("imageUrlMobile") or ""
            if not detail_url or not img_url:
                stats.add("no_url"); continue
            if not robots_allowed(detail_url):
                stats.add("robots_block"); continue

            raw = fetch_image_bytes(img_url)
            if not raw:
                stats.add("img_fetch_fail"); continue

            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                stats.add("img_open_fail"); continue
            if not ensure_min_side(img, MIN_SIDE):
                stats.add("too_small"); continue

            phx = imagehash.phash(img).__str__() if USE_PHASH else None
            if phx and any(bin(int(phx,16)^int(hx,16)).count("1") <= PHASH_THRESHOLD for hx,_,_ in known_hashes):
                stats.add("phash_dup"); continue

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
                "img_url": img_url,
                "file_path": str(out_path.as_posix()),
                "width": img.size[0], "height": img.size[1],
                "phash_hex": phx or "",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })

            if phx:
                known_hashes.append((phx, detected, str(out_path)))
            saved[detected] += 1
            print(f"  ✔ {SITE} | {detected} | {name[:60]}... → {fname} ({saved[detected]}/{per_label_target})")

    print("\n[done] W컨셉(API) 수집 완료. 저장:", SAVE_ROOT.as_posix())
    print("       manifest:", MANIFEST.as_posix())
    print("       per-label:", saved)
    stats.dump()

if __name__ == "__main__":
    # 먼저 작게 검증 후 수치 올리기
    collect_api(per_label_target=20, labels_filter=None, max_pages=10)
