"""Price extraction. Tries several strategies so it survives layout changes."""
import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class ScrapeError(Exception):
    """Raised when a price can't be fetched; message is safe to show the user."""


@dataclass
class Product:
    title: str
    price: float


def parse_price(text):
    """'₹1,299.00' / 'Rs. 1.299,00' / '$19.99' -> float, or None."""
    if text is None:
        return None
    s = re.sub(r"[^\d.,]", "", str(text))
    if not s:
        return None
    # Decide which symbol is the decimal separator.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):          # 1.299,00
            s = s.replace(".", "").replace(",", ".")
        else:                                    # 1,299.00
            s = s.replace(",", "")
    elif "," in s:
        head, _, tail = s.rpartition(",")
        s = head.replace(",", "") + "." + tail if len(tail) == 2 else s.replace(",", "")
    try:
        value = float(s)
    except ValueError:
        return None
    return value if value > 0 else None


ASIN_RE = re.compile(r"(?:/dp/|/gp/product/|/gp/aw/d/|/product/)([A-Z0-9]{10})(?:[/?&%]|$)")


def clean_amazon_url(url):
    """Reduce any Amazon link (incl. /sspa/click sponsored links) to https://host/dp/ASIN."""
    p = urlparse(url)
    if "amazon." not in p.netloc:
        return url
    candidates = [unquote(url), p.path]
    inner = parse_qs(p.query).get("url")
    if inner:
        candidates.insert(0, unquote(inner[0]))
    for c in candidates:
        m = ASIN_RE.search(c)
        if m:
            return f"{p.scheme}://{p.netloc}/dp/{m.group(1)}"
    if "/sspa/click" in p.path:
        raise ScrapeError(
            "That is a sponsored-ad redirect link and the product part was cut off when it was copied. "
            "Open the product page and copy the link from the address bar - it should contain /dp/ and a 10-character code."
        )
    return url


def title_from_url(url):
    """Best-effort readable name when the page title is unknown (manual entry)."""
    p = urlparse(url)
    m = re.search(r"(?:^|/)([^/]+)/(?:dp|gp/product)/", unquote(url))
    if m and "-" in m.group(1):
        return m.group(1).replace("-", " ")
    m = ASIN_RE.search(url)
    if m:
        return f"Amazon product {m.group(1)}"
    return p.netloc.replace("www.", "") + " product"


def validate_url(url):
    url = (url or "").strip()
    if url and not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    p = urlparse(url)
    if not p.netloc or "." not in p.netloc:
        raise ScrapeError("That doesn't look like a product link. Paste the full address of the product page.")
    return clean_amazon_url(url)


def _jsonld_product(soup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                stack.extend(node.get("@graph", []) if isinstance(node.get("@graph"), list) else [])
                types = node.get("@type")
                types = types if isinstance(types, list) else [types]
                if "Product" in types:
                    offers = node.get("offers")
                    offers = offers if isinstance(offers, list) else [offers]
                    for o in offers:
                        if isinstance(o, dict):
                            price = parse_price(o.get("price") or o.get("lowPrice"))
                            if price:
                                return node.get("name"), price
            elif isinstance(node, list):
                stack.extend(node)
    return None, None


AMAZON_SELECTORS = [
    "#corePrice_feature_div .a-price .a-offscreen",
    "#corePriceDisplay_desktop_feature_div .a-price .a-offscreen",
    "#apex_desktop .a-price .a-offscreen",
    "span.priceToPay .a-offscreen",
    "#priceblock_dealprice",
    "#priceblock_ourprice",
    "#price_inside_buybox",
    ".a-price .a-offscreen",
    ".a-price-whole",
]
META_SELECTORS = [
    ("meta", {"property": "product:price:amount"}),
    ("meta", {"property": "og:price:amount"}),
    ("meta", {"itemprop": "price"}),
]


def extract_product(html):
    """Return Product parsed from page HTML, or raise ScrapeError."""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True).lower()
    if "enter the characters you see below" in text or "automated access" in text or "robot check" in text:
        raise ScrapeError(
            "The shop showed a bot-check (CAPTCHA) instead of the product page. "
            "Enter the price manually below, or try again later."
        )

    title_el = soup.select_one("#productTitle") or soup.find("meta", property="og:title") or soup.title
    title = None
    if title_el is not None:
        title = (title_el.get("content") if title_el.name == "meta" else title_el.get_text(strip=True)) or None

    ld_title, ld_price = _jsonld_product(soup)
    if ld_price:
        return Product(ld_title or title or "Product", ld_price)

    for sel in AMAZON_SELECTORS:
        el = soup.select_one(sel)
        if el:
            price = parse_price(el.get_text())
            if price:
                return Product(title or "Product", price)

    for name, attrs in META_SELECTORS:
        el = soup.find(name, attrs=attrs)
        if el and parse_price(el.get("content")):
            return Product(title or "Product", parse_price(el.get("content")))

    # Embedded JSON blobs used by Amazon / Flipkart / generic shops
    for pat in (
        r'"priceAmount"\s*:\s*"?([\d.,]+)',
        r'"displayPrice"\s*:\s*"[^\d"]*([\d.,]+)"',
        r'id="twister-plus-price-data-price"[^>]*value="([\d.,]+)"',
        r'"price"\s*:\s*"?([\d]{2,}[\d.,]*)"?\s*,\s*"(?:priceCurrency|currency)',
    ):
        m = re.search(pat, html)
        if m and parse_price(m.group(1)):
            return Product(title or "Product", parse_price(m.group(1)))

    raise ScrapeError(
        "Couldn't find a price on that page. The shop may be hiding it from scripts - "
        "enter the price yourself and it will still be tracked."
    )


ALT_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


def _fetch_browser(url):
    """Optional: real headless Chromium (pip install playwright && playwright install chromium)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_context(locale="en-IN", user_agent=ALT_AGENTS[0]).new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(1500)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return None


def fetch_product(url, timeout=15):
    import os
    import time
    last_error = None
    session = requests.Session()
    for attempt, agent in enumerate(ALT_AGENTS):
        headers = {**HEADERS, "User-Agent": agent, "Upgrade-Insecure-Requests": "1",
                   "Referer": "https://www.google.com/"}
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            last_error = ScrapeError(f"Couldn't reach the page ({e.__class__.__name__}). Check your internet connection.")
            continue
        if resp.status_code == 404:
            raise ScrapeError("The shop says that page doesn't exist (HTTP 404). Check the link.")
        if resp.status_code >= 400:
            last_error = ScrapeError(f"The shop blocked the request (HTTP {resp.status_code}).")
            time.sleep(1 + attempt)
            continue
        try:
            return extract_product(resp.text)
        except ScrapeError as e:
            last_error = e
            time.sleep(1 + attempt)

    if os.environ.get("ECOMTRACK_BROWSER", "1") == "1":
        html = _fetch_browser(url)
        if html:
            try:
                return extract_product(html)
            except ScrapeError as e:
                last_error = e
    raise last_error or ScrapeError("Couldn't fetch the page.")
