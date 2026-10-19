import csv
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT
);
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    price REAL NOT NULL,
    source TEXT NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS idx_prices_product ON prices(product_id, ts);
"""


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(path):
    with connect(path) as c:
        c.executescript(SCHEMA)
        cols = [r["name"] for r in c.execute("PRAGMA table_info(prices)")]
        if "source" not in cols:   # upgrade databases created by the previous version
            c.execute("ALTER TABLE prices ADD COLUMN source TEXT NOT NULL DEFAULT 'live'")


def get_or_create_product(conn, url, title=None):
    row = conn.execute("SELECT * FROM products WHERE url=?", (url,)).fetchone()
    if row:
        if title and not row["title"]:
            conn.execute("UPDATE products SET title=? WHERE id=?", (title, row["id"]))
        return row["id"]
    return conn.execute("INSERT INTO products(url,title) VALUES(?,?)", (url, title)).lastrowid


def add_price(conn, product_id, price, ts=None, source="live"):
    ts = (ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO prices(product_id,ts,price,source) VALUES(?,?,?,?)", (product_id, ts, price, source))


def history(conn, product_id):
    rows = conn.execute("SELECT ts, price, source FROM prices WHERE product_id=? ORDER BY ts", (product_id,)).fetchall()
    return [(datetime.strptime(r["ts"], "%Y-%m-%d %H:%M:%S"), r["price"], r["source"]) for r in rows]


def list_products(conn):
    return conn.execute(
        """SELECT p.id, p.url, p.title,
                  (SELECT price FROM prices WHERE product_id=p.id ORDER BY ts DESC LIMIT 1) AS last_price,
                  (SELECT COUNT(*) FROM prices WHERE product_id=p.id) AS n
           FROM products p ORDER BY p.id DESC"""
    ).fetchall()


def import_legacy_csv(conn, csv_path):
    """One-time import of the old prices.csv, skipping malformed rows."""
    from scraper import clean_amazon_url
    path = Path(csv_path)
    if not path.exists() or conn.execute("SELECT 1 FROM products LIMIT 1").fetchone():
        return 0
    imported = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url, raw_ts, raw_price = row.get("url"), row.get("timestamp"), row.get("price")
            if not url:      # older rows had only timestamp,price
                continue
            try:
                price = float(raw_price)
                ts = datetime.fromisoformat(raw_ts.strip())
            except (TypeError, ValueError):
                continue
            pid = get_or_create_product(conn, clean_amazon_url(url))
            add_price(conn, pid, price, ts)
            imported += 1
    conn.commit()
    return imported


DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


def parse_history_text(text):
    """Parse pasted/uploaded lines like '2026-08-01, 5,990' or '01/08/2026;5990'.
    Returns (rows[(datetime, price)], skipped_line_count)."""
    from scraper import parse_price
    import re
    rows, skipped = [], 0
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("date", "timestamp")):
            continue
        when = price = None
        if re.search(r"[;\t|]", line):
            splits = [tuple(p.strip().strip('"') for p in re.split(r"[;\t|]", line, maxsplit=1))]
        else:   # commas may also appear inside the price (5,990) or date (Sep 1, 2026): try each one
            splits = [(line[:i].strip().strip('"'), line[i + 1:].strip().strip('"'))
                      for i, ch in enumerate(line) if ch == ","]
        for raw_date, raw_price in splits:
            for fmt in DATE_FORMATS:
                try:
                    when = datetime.strptime(raw_date, fmt)
                    break
                except ValueError:
                    pass
            price = parse_price(raw_price) if when else None
            if when and price:
                break
            when = price = None
        if when and price:
            rows.append((when, price))
        else:
            skipped += 1
    return rows, skipped


def make_sample_history(current_price, product_id, days=120, step=4):
    """Clearly-labelled simulated history (random walk ending near the current price)."""
    import random
    rng = random.Random(product_id * 7919)
    now = datetime.now()
    points, price = [], current_price * rng.uniform(1.05, 1.25)
    n = days // step
    for i in range(n, 0, -1):
        drift = (current_price - price) / max(i, 1) * 0.35
        price = max(current_price * 0.6, price * (1 + rng.uniform(-0.05, 0.04)) + drift)
        points.append((now - timedelta(days=i * step), round(price)))
    return points
