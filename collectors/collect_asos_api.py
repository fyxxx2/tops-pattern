# collectors/collect_asos_api.py
# ASOS 검색 JSON API 전용 크롤러 (여성군 + floral/polka_dot, 카테고리 제한 해제, 대량 수집)
# 저장: data/raw/tops/<label>/asos_<productId>.jpg
# manifest: data/raw/tops/_manifest_asos.csv

import io, time, json, random
from typing import Dict, List, Optional
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
import imagehash

# ─────────────────────────────────────────────────────────
# 프로젝트 공용 유틸(있으면 사용; 없으면 아래 대체 사용)
try:
    from crawler_utils import (
        SAVE_ROOT, ensure_min_side, load_manifest, append_manifest,
        MIN_SIDE, PHASH_THRESHOLD, USE_PHASH
    )
except Exception:
    SAVE_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "tops"
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    MIN_SIDE = 400
    USE_PHASH = True
    PHASH_THRESHOLD = 6

    def ensure_min_side(img: Image.Image, min_side: int = 400) -> bool:
        return min(img.size) >= min_side

    def load_manifest(manifest_path: Path):
        ids, phashes = set(), []
        if manifest_path.exists():
            import csv
            with open(manifest_path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    ids.add(str(row.get("product_id","")).strip())
                    hx = row.get("phash_hex","").strip()
                    if hx:
                        phashes.append((hx, row.get("label",""), row.get("file_path","")))
        return ids, phashes

    def append_manifest(manifest_path: Path, row: Dict):
        import csv
        write_header = not manifest_path.exists()
        with open(manifest_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "site","product_id","label","title","url","img_url","file_path",
                "width","height","phash_hex","created_at"
            ])
            if write_header:
                w.writeheader()
            w.writerow(row)

# ─────────────────────────────────────────────────────────
SITE = "asos"
MANIFEST = SAVE_ROOT / "_manifest_asos.csv"

# 여성군 강제 — attribute_1047:6132 = Women
WOMEN_REFINE = "attribute_1047:6132"

LABELS = ["polka_dot","floral"]
SEARCH_KEYWORDS = {
    "polka_dot": [
        "polka dot", "polkadot", "polka-dot", "dot print", "spot print", "spotted", "micro dot", "mini dot"
    ],
    "floral": [
        "floral", "florals", "floral print", "flower print", "liberty", "botanical", "ditsy",
        "rose print", "bouquet", "tropical", "hawaiian", "garden print"
    ],
}

ASOS_SEARCH_ENDPOINT = "https://www.asos.com/api/product/search/v2/"

# ASOS 전용 이미지 최소 변 기준(완화)
ASOS_MIN_SIDE = 256

# ─────────────────────────────────────────────────────────
def make_headers() -> Dict[str,str]:
    return {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/141.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.asos.com/",
        "Origin": "https://www.asos.com",
    }

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=6, connect=6, read=6,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32))
    s.headers.update(make_headers())
    return s

SESSION = _build_session()

DEFAULT_PARAMS = {
    "country": "KR",
    "lang": "en-GB",
    "store": "ROW",
    "currency": "USD",
    "sizeSchema": "UK",
    "offset": 0,
    "limit": 72,          # 최대 72
    "q": "",
    "refine": WOMEN_REFINE,  # 여성군만
}

def normalize_img(url: str) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    if u.startswith("//"):
        u = "https:" + u
    elif u.startswith("/"):
        u = "https://www.asos.com" + u
    elif not u.startswith("http"):
        u = "https://" + u
    return u

def constrain_asos_img(url: str, wid: int = 1000) -> Optional[str]:
    u = normalize_img(url)
    if not u:
        return None
    if "images.asos-media.com" in u:
        sep = "&" if "?" in u else "?"
        if "wid=" not in u:
            u = f"{u}{sep}wid={wid}&fit=constrain"
    return u

def fetch_asos_image_bytes(url: str, tries: int = 5) -> Optional[bytes]:
    big  = constrain_asos_img(url, wid=1000) or url
    mid  = constrain_asos_img(url, wid=800)  or big
    last = constrain_asos_img(url, wid=600)  or mid
    chain = [big, big, mid, mid, last]
    for i, use in enumerate(chain, 1):
        try:
            r = SESSION.get(use, timeout=(8, 45), stream=True)
            if r.status_code == 200 and r.content:
                return r.content
            else:
                print(f"  [img http {r.status_code}] {use}")
        except requests.exceptions.ReadTimeout:
            print(f"  [img timeout] {use} (try {i}/{len(chain)})")
        except Exception as e:
            print(f"  [img error] {e} (try {i}/{len(chain)})")
        time.sleep(0.5 * i)
    return None

def select_best_image(p: dict) -> Optional[str]:
    cand = []
    if p.get("imageUrl"):
        cand.append(p["imageUrl"])
    for a in (p.get("alternateImageUrls") or []):
        if a: cand.append(a)
    for c in cand:
        u = normalize_img(c)
        if u:
            return u
    return None

def request_with_retry(params: Dict, max_retry=3, timeout=25) -> Optional[dict]:
    for i in range(max_retry):
        try:
            r = SESSION.get(ASOS_SEARCH_ENDPOINT, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            else:
                print(f"  [http {r.status_code}] offset={params.get('offset')} q='{params.get('q')}'")
        except requests.exceptions.ReadTimeout:
            print(f"  [timeout] offset={params.get('offset')} q='{params.get('q')}' (try {i+1}/{max_retry})")
        except Exception as e:
            print(f"  [fetch err] {e} (try {i+1}/{max_retry})")
        time.sleep(1.0 + 0.5*random.random())
    return None

class SkipStats:
    def __init__(self):
        self.c: Dict[str, int] = {}
    def add(self, reason: str):
        self.c[reason] = self.c.get(reason, 0) + 1
    def dump(self, prefix="[skip-stats]"):
        print(prefix, self.c)

# ─────────────────────────────────────────────────────────
def collect_asos_search(per_label_target=300, max_pages=60):
    """
    per_label_target: 라벨별 저장 목표 수 (디폴트 300)
    max_pages: 키워드별 최대 페이지 수 (72 * max_pages 아이템 조회)
    """
    known_ids, known_hashes = load_manifest(MANIFEST)
    print(f"[{SITE}] manifest ids={len(known_ids)}, phash={len(known_hashes)}")

    saved = {lb: 0 for lb in LABELS}
    seen: set = set()
    stats = SkipStats()

    for label in LABELS:
        kws = SEARCH_KEYWORDS[label]
        print(f"\n[search] label={label} → keywords={kws}")
        if saved[label] >= per_label_target:
            continue

        for kw in kws:
            if saved[label] >= per_label_target:
                break

            total = 0
            for page in range(max_pages):
                if saved[label] >= per_label_target:
                    break

                params = DEFAULT_PARAMS.copy()
                params["q"] = kw
                params["offset"] = page * params["limit"]

                resp = request_with_retry(params)
                if not resp or not isinstance(resp, dict):
                    if page == 0:
                        print(f"  [warn] first page empty for '{kw}' → stop this keyword")
                    break

                products = resp.get("products") or resp.get("items") or []
                if not products:
                    if page == 0:
                        print(f"  [warn] no products for '{kw}'")
                    break

                print(f"  [page {page+1}] got {len(products)} items for '{kw}'")
                total += len(products)

                for p in products:
                    if saved[label] >= per_label_target:
                        break

                    pid = str(p.get("id") or "").strip()
                    if not pid:
                        stats.add("no_id"); continue
                    if pid in seen or pid in known_ids:
                        stats.add("dup_id"); continue
                    seen.add(pid)

                    name = (p.get("name") or "").strip()

                    # 이미지
                    img_url = select_best_image(p)
                    if not img_url:
                        stats.add("no_img_url"); continue

                    raw = fetch_asos_image_bytes(img_url)
                    if not raw:
                        stats.add("img_fetch_fail"); continue

                    try:
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                    except Exception:
                        stats.add("img_open_fail"); continue
                    if not ensure_min_side(img, ASOS_MIN_SIDE):
                        stats.add("too_small"); continue

                    phx = imagehash.phash(img).__str__() if USE_PHASH else ""
                    if phx and any(bin(int(phx,16)^int(hx,16)).count("1") <= PHASH_THRESHOLD for hx,_,_ in known_hashes):
                        stats.add("phash_dup"); continue

                    out_dir = (SAVE_ROOT / label)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fname = f"asos_{pid}.jpg"
                    out_path = out_dir / fname
                    with open(out_path, "wb") as f:
                        f.write(raw)

                    product_url = p.get("url") or ""
                    if product_url and product_url.startswith("/"):
                        product_url = "https://www.asos.com" + product_url

                    append_manifest(MANIFEST, {
                        "site": SITE,
                        "product_id": pid,
                        "label": label,
                        "title": name,
                        "url": product_url,
                        "img_url": img_url,
                        "file_path": str(out_path.as_posix()),
                        "width": img.size[0], "height": img.size[1],
                        "phash_hex": phx or "",
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    if phx:
                        known_hashes.append((phx, label, str(out_path)))
                    saved[label] += 1
                    print(f"   ✔ {SITE} | {label} | {name[:60]}... → {fname} ({saved[label]}/{per_label_target})")

                    time.sleep(0.12)  # 이미지 요청 간격(약간의 완화)

                # **종료 조건**: 마지막 페이지는 limit보다 적게 오면 끝
                if len(products) < params["limit"]:
                    break

                time.sleep(0.35 + 0.25*random.random())

            if total == 0:
                continue

    print("\n[done] ASOS 검색 수집 완료. 저장:", SAVE_ROOT.as_posix())
    print("       manifest:", MANIFEST.as_posix())
    print("       per-label:", saved)
    stats.dump()

if __name__ == "__main__":
    collect_asos_search(per_label_target=300, max_pages=60) #300개를 60페이지에서 추출
