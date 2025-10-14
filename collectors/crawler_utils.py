# collectors/crawler_utils.py
from pathlib import Path
from typing import List, Tuple, Optional
from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser
import re, csv, time, random, requests

from bs4 import BeautifulSoup
from PIL import Image
import imagehash

# ====== 경로 설정 (collectors → 상위 루트 → data/raw/tops) ======
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVE_ROOT = PROJECT_ROOT / "data" / "raw" / "tops"
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

# ====== 요청/딜레이/헤더 ======
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.wconcept.co.kr/",
    "Cache-Control": "no-cache",
}

REQ_SLEEP_RANGE = (1.0, 2.0)
PAGE_SLEEP_RANGE = (1.5, 3.0)

def polite_sleep(a=REQ_SLEEP_RANGE[0], b=REQ_SLEEP_RANGE[1]):
    time.sleep(random.uniform(a, b))

# ====== 이미지/품질/중복 ======
MIN_SIDE = 700          # 더 엄격히 하려면 800~900
USE_PHASH = True
PHASH_THRESHOLD = 5     # 더 엄격히: 4

# ====== 필터(상의/라벨) ======
APPAREL_WORDS = {
    "top","tee","t-shirt","shirt","blouse","sweater","hoodie","cardigan",
    "맨투맨","티셔츠","셔츠","블라우스","니트","후드","가디건","상의","아우터",
    "카라","피케","폴로","집업","라글란","크루넥","브이넥","재킷","자켓","점퍼"
}
PATTERN_KEYWORDS = {
    "stripe":    ["stripe","striped","pinstripe","vertical stripe","스트라이프"],
    "plaid":     ["plaid","check","checked","gingham","tartan","체크","격자"],
    "polka_dot": ["polka dot","polkadot","dot print","dots","도트"],
    "floral":    ["floral","floral print","flower pattern","botanical","플로럴","꽃무늬"],
    "solid":     ["solid","plain","basic","무지","솔리드"]
}
LABELS_ORDER = ["stripe","plaid","polka_dot","floral","solid"]

def is_apparel(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in APPAREL_WORDS)

def guess_label(text: str, fallback_label: Optional[str]=None) -> Optional[str]:
    t = (text or "").lower()
    hits = [lb for lb, kws in PATTERN_KEYWORDS.items() if any(k in t for k in kws)]
    if hits:
        for lb in LABELS_ORDER:
            if lb in hits:
                return lb
    return fallback_label

# ====== robots 허용(완화) ======
def _is_sitemap_url(url: str) -> bool:
    u = urlparse(url)
    return ("sitemap" in u.path.lower()) or u.path.lower().endswith(".xml")

# << 여기 핵심: ALLOW_HOSTS 에 display.wconcept.co.kr 포함 >>
ALLOW_HOSTS = {
    "www.wconcept.co.kr", "wconcept.co.kr",
    "display.wconcept.co.kr",  # sitemap 서버
}

def robots_allowed(url: str) -> bool:
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = f"{base}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        r = requests.get(robots_url, headers=HEADERS, timeout=10)
        if r.status_code == 200 and r.text.strip():
            rp.parse(r.text.splitlines())
        else:
            rp.set_url(robots_url)
            rp.read()
        ua_ok = rp.can_fetch(HEADERS["User-Agent"], url)
        star_ok = rp.can_fetch("*", url)
        return bool(ua_ok or star_ok)
    except Exception:
        host = parsed.netloc.lower()
        if _is_sitemap_url(url):
            return True
        if host in ALLOW_HOSTS:
            return True
        print(f"[robots-warning] fallback allow: {url}")
        return True

# ====== HTTP ======
def get_html(url: str) -> Optional[str]:
    try:
        if not _is_sitemap_url(url):
            if not robots_allowed(url):
                print(f"[robots] Not allowed: {url}")
                return None
        res = requests.get(url, headers=HEADERS, timeout=20)
        if res.status_code != 200:
            print(f"[HTTP {res.status_code}] {url}")
            return None
        return res.text
    except Exception as e:
        print("[req error]", e)
        return None

def fetch_image_bytes(url: str) -> Optional[bytes]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print("  [img error]", e)
        return None

# ====== 이미지/파일 ======
def ensure_min_side(img: Image.Image, min_side=MIN_SIDE) -> bool:
    w, h = img.size
    return min(w, h) >= min_side

def to_product_id(url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", urlparse(url).path).strip("_") or "unknown"

# ====== manifest ======
def load_manifest(manifest_path: Path) -> Tuple[set, List[Tuple[str,str,str]]]:
    known_ids, known_hashes = set(), []
    if manifest_path.exists():
        with open(manifest_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                pid = row.get("product_id")
                if pid:
                    known_ids.add(pid)
                ph = row.get("phash_hex")
                if ph:
                    known_hashes.append((ph, row.get("label",""), row.get("file_path","")))
    return known_ids, known_hashes

def append_manifest(manifest_path: Path, row: dict):
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "site","product_id","label","title","url","img_url","file_path",
            "width","height","phash_hex","created_at"
        ])
        if write_header:
            w.writeheader()
        w.writerow(row)

# ====== sitemap helpers ======
def parse_sitemap_index(xml_text: str) -> List[str]:
    soup = BeautifulSoup(xml_text, "xml")
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]

def parse_sitemap_urls(xml_text: str) -> List[str]:
    soup = BeautifulSoup(xml_text, "xml")
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]