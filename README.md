# EcomTrack

Price tracker and buying adviser. Paste a product link, it records the price, charts the history (highest / lowest / average) and says **Buy now** or **Wait**.

## Run
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py                                        # http://127.0.0.1:5000
```
Optional hourly auto-refresh of all tracked products: `ENABLE_SCHEDULER=1 python app.py`
Tests: `pip install -r requirements-dev.txt` then `pytest`

## Getting history
A new product starts with one price. To get advice straight away: paste/upload past prices on the product page (`date, price` per line), or use "Try with sample history" (clearly marked as simulated). For real history of Amazon items, copy the graph values from Keepa.

## Optional: harder-to-scrape shops
If a shop hides prices from plain requests, install a real headless browser and the app will use it automatically as a fallback:
`pip install playwright` then `playwright install chromium`

## Notes
- Data is stored in `ecomtrack.db` (SQLite). Your old `prices.csv` is imported once on first run (malformed rows skipped).
- Amazon and other big shops often block scripts (CAPTCHA / HTTP 503). When that happens the app tells you and lets you type the price manually, so it never just crashes.
- Advice uses a simple trend + average/lowest comparison (no Prophet/Selenium needed).
