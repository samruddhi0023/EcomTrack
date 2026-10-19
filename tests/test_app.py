import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
import pytest
from app import create_app
from scraper import Product, ScrapeError, extract_product, parse_price, clean_amazon_url
import forecast

AMAZON_HTML = '<html><body><span id="productTitle"> Test Guitar </span><div id="corePrice_feature_div"><span class="a-price"><span class="a-offscreen">₹4,999.00</span></span></div></body></html>'
LD_HTML = '<html><head><script type="application/ld+json">{"@type":"Product","name":"Shoe","offers":{"price":"59.90"}}</script></head></html>'


def test_parse_price():
    assert parse_price("₹1,299.00") == 1299.0
    assert parse_price("1.299,50") == 1299.5
    assert parse_price("$19.99") == 19.99
    assert parse_price("abc") is None


def test_extract():
    assert extract_product(AMAZON_HTML) == Product("Test Guitar", 4999.0)
    assert extract_product(LD_HTML) == Product("Shoe", 59.9)
    with pytest.raises(ScrapeError):
        extract_product("<html><body>Enter the characters you see below</body></html>")
    with pytest.raises(ScrapeError):
        extract_product("<html><body>nothing</body></html>")


def test_clean_url():
    u = "https://www.amazon.in/sspa/click?ie=UTF8&url=%2FFoo-Bar%2Fdp%2FB0DJ35MC78%2Fref%3Dsr_1_1"
    assert clean_amazon_url(u) == "https://www.amazon.in/dp/B0DJ35MC78"


def test_forecast():
    t = datetime(2026, 1, 1)
    ts = [t + timedelta(days=i) for i in range(5)]
    assert "Not enough" in forecast.advise(ts[:2], [10, 9])[0]
    assert forecast.advise(ts, [100, 90, 80, 70, 60])[0].startswith("Buy now")   # at the low
    assert forecast.advise(ts, [60, 70, 80, 90, 100])[0].startswith("Buy now")   # rising: buy before it climbs
    assert forecast.advise(ts, [60, 100, 100, 100, 99])[0].startswith(("Wait", "Buy now"))


@pytest.fixture
def client(tmp_path):
    prices = iter([100.0, 95.0, 90.0, 99.0])
    def fake(url):
        try: return Product("Widget", next(prices))
        except StopIteration: raise ScrapeError("blocked")
    app = create_app(str(tmp_path / "t.db"), fetcher=fake)
    app.config["TESTING"] = True
    return app.test_client()


def test_flow(client):
    assert client.get("/").status_code == 200
    r = client.post("/track", data={"url": "example.com/p"}, follow_redirects=True)
    assert b"Widget" in r.data and b"Not enough" in r.data and b"Needs history" in r.data
    for _ in range(2):
        client.post("/product/1/refresh")
    r = client.get("/product/1")
    assert r.status_code == 200 and b"3 records" in r.data
    client.post("/product/1/refresh")                      # 4th ok
    r = client.post("/product/1/refresh", follow_redirects=True)   # blocked -> message, no crash
    assert b"blocked" in r.data
    r = client.post("/track", data={"url": "other.com/x"})  # fetch fails -> manual form
    assert b"Enter the price yourself" in r.data
    r = client.post("/track", data={"url": "other.com/x", "manual_price": "49.5"}, follow_redirects=True)
    assert b"49.50" in r.data
    assert client.post("/product/1/delete", follow_redirects=True).status_code == 200
    assert client.get("/product/1").status_code == 404


def test_history_and_sample(client):
    client.post("/track", data={"url": "example.com/p", "manual_price": "5390"}, follow_redirects=True)
    r = client.post("/product/1/history", data={"history": "2026-08-01, 5,990\n01/09/2026;5490\nnonsense"}, follow_redirects=True)
    assert b"Imported 2 past prices" in r.data and b"Skipped 1" in r.data
    assert b"Needs history" not in r.data            # 3 records -> real advice
    r = client.post("/product/1/sample", follow_redirects=True)
    assert b"simulated sample history" in r.data
    r = client.post("/product/1/clear-sample", follow_redirects=True)
    assert b"includes simulated" not in r.data
    r = client.post("/product/1/history", data={"history": "garbage"}, follow_redirects=True)
    assert b"No prices found" in r.data


def test_bad_sponsored_link(client):
    r = client.post("/track", data={"url": "https://www.amazon.in/sspa/click?ie=UTF8&spc=MTo3Mzc0"}, follow_redirects=True)
    assert b"sponsored-ad redirect link" in r.data


def test_money_format():
    from app import money
    assert money(123456, "https://amazon.in/dp/X") == "\u20b91,23,456"
    assert money(5390, "https://amazon.in/dp/X") == "\u20b95,390"
    assert money(1234.5, "https://shop.com/x") == "$1,234.50"
