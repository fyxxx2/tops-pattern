# collectors/collect_wconcept.py — W컨셉 단일 크롤러(사이트맵 + 루트 링크 보강)
# 저장: 루트/data/raw/tops/<label>/wconcept_<id>.jpg
# manifest: 루트/data/raw/tops/_manifest_wconcept.csv

import io, time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from PIL import Image
import imagehash

from crawler_utils import (
    SAVE_ROOT, get_html, fetch_image_bytes, ensure_min_side,
    is_apparel, guess_label, load_manifest, append_manifest,
    polite_sleep, parse_sitemap_index, parse_sitemap_urls,
    to_product_id, MIN_SIDE, PHASH_THRESHOLD, USE_PHASH, robots_allowed
)

SITE = "wconcept"
MANIFEST = SAVE_ROOT / "_manifest_wconcept.csv"

# 네가 준 링크(사이트맵 서버)
SITEMAP_HOME  = "https://display.wconcept.co.kr"
SITEMAP_INDEX = f"{SITEMAP_HOME}/sitemap.xml"

MAX_SITEMAPS = 25
MAX_PRODUCTS_PER_MAP = 800
TOTAL_PRODUCT_LIMIT = 8000

def is_product_url(u: str) -> bool:
    return "/product/" in u.lower()

def extract_title_img(detail_html: str, detail_url: str):
    soup = BeautifulSoup(detail_html, "html.parser")
    # title
    title = None
    m = soup.find("meta", property="og:title")
    if m and m.get("content"):
        title = m["content"].strip()
    elif soup.title:
        title = soup.title.get_text(" ", strip=True)
    # image
    img_url = None
    m2 = soup.find("meta", property="og:image")
    if m2 and m2.get("content"):
        img_url = m2["content"]
    else:
        imgtag = soup.find("img")
        if imgtag and imgtag.get("src"):
            src = imgtag.get("src")
            img_url = src if src.startswith("http") else urljoin(detail_url, src)
    return title, img_url

def collect(per_label_target=60, labels_filter=None):
    known_ids, known_hashes = load_manifest(MANIFEST)
    print(f"[{SITE}] manifest ids={len(known_ids)}, phash={len(known_hashes)}")

    saved = {lb: 0 for lb in ["solid","stripe","plaid","polka_dot","floral"]}
    total_processed = 0

    # 1) 사이트맵 인덱스
    idx_xml = get_html(SITEMAP_INDEX)
    sitemap_urls = parse_sitemap_index(idx_xml) if idx_xml else []
    if not sitemap_urls:
        print("[warn] sitemap index 비어있음:", SITEMAP_INDEX)

    # product 관련 맵 우선
    maps = [u for u in sitemap_urls if "product" in u.lower()] or sitemap_urls
    maps = maps[:MAX_SITEMAPS]

    # 2) 각 sitemap에서 product 상세 URL 수집
    all_prod = []
    for sm in maps:
        xml = get_html(sm); polite_sleep()
        if not xml:
            print("[skip] sitemap 로드 실패:", sm)
            continue
        urls = parse_sitemap_urls(xml)
        prod_urls = [u for u in urls if is_product_url(u)]
        prod_urls = prod_urls[:MAX_PRODUCTS_PER_MAP]
        print(f"  [sitemap] {sm} → products: {len(prod_urls)}")
        all_prod.extend(prod_urls)

    # 3) display.wconcept.co.kr 루트 페이지에서 추가로 a[href] 스캔(보강)
    html0 = get_html(SITEMAP_HOME)
    if html0:
        s0 = BeautifulSoup(html0, "html.parser")
        for a in s0.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            full = href if href.startswith("http") else urljoin(SITEMAP_HOME, href)
            if is_product_url(full):
                all_prod.append(full)

    # 4) 중복 제거
    prod_urls = list(dict.fromkeys(all_prod))
    print("[debug] 총 상품 URL 수:", len(prod_urls))

    for purl in prod_urls:
        if total_processed >= TOTAL_PRODUCT_LIMIT:
            break

        pid = to_product_id(purl)
        if pid in known_ids:
            continue

        if not robots_allowed(purl):
            continue

        html2 = get_html(purl); polite_sleep()
        if not html2:
            continue

        title, img_url = extract_title_img(html2, purl)
        if not title or not is_apparel(title):
            continue

        label = guess_label(title)
        if labels_filter and label not in labels_filter:
            continue
        if not label or saved[label] >= per_label_target:
            continue

        if not img_url:
            continue
        raw = fetch_image_bytes(img_url)
        if not raw:
            continue

        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            continue
        if not ensure_min_side(img, MIN_SIDE):
            continue

        phx = imagehash.phash(img).__str__() if USE_PHASH else None
        if phx:
            # 유사 이미지 중복 제거
            if any(bin(int(phx,16)^int(hx,16)).count("1") <= PHASH_THRESHOLD for hx,_,_ in known_hashes):
                continue

        out_dir = SAVE_ROOT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{SITE}_{pid}.jpg"
        out_path = out_dir / fname
        with open(out_path, "wb") as f:
            f.write(raw)

        append_manifest(MANIFEST, {
            "site": SITE,
            "product_id": pid,
            "label": label,
            "title": title,
            "url": purl,
            "img_url": img_url,
            "file_path": str(out_path.as_posix()),
            "width": img.size[0], "height": img.size[1],
            "phash_hex": phx or "",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        known_ids.add(pid)
        if phx:
            known_hashes.append((phx, label, str(out_path)))

        saved[label] += 1
        total_processed += 1
        print(f"  ✔ {SITE} | {label} | {title[:60]}... → {fname} ({saved[label]}/{per_label_target})")

    print("\n[done] W컨셉 수집 완료. 저장:", SAVE_ROOT.as_posix())
    print("       manifest:", MANIFEST.as_posix())

if __name__ == "__main__":
    # 먼저 소량(60)으로 검증 후, OK면 200~300으로 올려 수집
    collect(per_label_target=60, labels_filter=None)