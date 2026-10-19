import logging
import os
from urllib.parse import urlparse

from flask import Flask, abort, flash, redirect, render_template, request, url_for

import db
import forecast
from scraper import ScrapeError, fetch_product, parse_price, title_from_url, validate_url

BASE = os.path.dirname(os.path.abspath(__file__))


def currency_symbol(url):
    host = urlparse(url or "").netloc.lower()
    if host.endswith((".in", ".co.in")) or "flipkart" in host or "myntra" in host:
        return "₹"
    if host.endswith(".co.uk"):
        return "£"
    return "$"


def money(value, url=None):
    """Format with the shop's currency; Indian digit grouping (1,23,456) for rupees."""
    if value is None:
        return "-"
    sym = currency_symbol(url)
    whole, frac = f"{abs(value):.2f}".split(".")
    if sym == "₹" and len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    else:
        whole = f"{int(whole):,}"
    return f"{sym}{whole}" + ("" if frac == "00" else f".{frac}")


def sparkline(prices, w=120, h=36):
    if len(prices) < 2:
        return ""
    lo, hi = min(prices), max(prices)
    span = (hi - lo) or 1
    pts = [f"{i / (len(prices) - 1) * w:.1f},{h - 3 - (p - lo) / span * (h - 6):.1f}" for i, p in enumerate(prices)]
    return " ".join(pts)


def create_app(db_path=None, fetcher=fetch_product):
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-change-me")
    app.config["DB_PATH"] = db_path or os.path.join(BASE, "ecomtrack.db")
    app.config["FETCHER"] = fetcher
    app.jinja_env.filters["money"] = money
    logging.basicConfig(level=logging.INFO)

    db.init(app.config["DB_PATH"])
    with db.connect(app.config["DB_PATH"]) as c:
        n = db.import_legacy_csv(c, os.path.join(BASE, "prices.csv"))
        if n:
            logging.info("Imported %d rows from legacy prices.csv", n)

    def conn():
        return db.connect(app.config["DB_PATH"])

    def cards():
        out = []
        with conn() as c:
            for p in db.list_products(c):
                hist = db.history(c, p["id"])
                prices = [h[1] for h in hist]
                if not prices:
                    continue
                _, _, verdict = forecast.advise([h[0] for h in hist], prices)
                change = None
                if len(prices) > 1 and prices[-2]:
                    change = (prices[-1] - prices[-2]) / prices[-2] * 100
                out.append({"id": p["id"], "url": p["url"], "title": p["title"] or title_from_url(p["url"]),
                            "price": prices[-1], "spark": sparkline(prices[-30:]), "change": change,
                            "verdict": verdict, "n": len(prices), "low": min(prices)})
        return out

    @app.route("/")
    def home():
        return render_template("index.html", products=cards())

    @app.route("/track", methods=["POST"])
    def track():
        try:
            url = validate_url(request.form.get("url"))
        except ScrapeError as e:
            flash(str(e), "error")
            return redirect(url_for("home"))

        manual = request.form.get("manual_price", "").strip()
        manual_title = request.form.get("title", "").strip() or None
        title = manual_title
        try:
            if manual:
                price = parse_price(manual)
                if not price:
                    raise ScrapeError("Enter the price as a positive number, like 5390.")
                title = title or title_from_url(url)
            else:
                product = app.config["FETCHER"](url)
                price, title = product.price, product.title
        except ScrapeError as e:
            flash(str(e), "error")
            return render_template("index.html", products=cards(), url=url, ask_manual=True)

        with conn() as c:
            pid = db.get_or_create_product(c, url, title)
            db.add_price(c, pid, price, source="manual" if manual else "live")
        flash("Now tracking this product. Add past prices below to get advice right away.", "ok")
        return redirect(url_for("product", pid=pid))

    @app.route("/product/<int:pid>")
    def product(pid):
        with conn() as c:
            row = c.execute("SELECT * FROM products WHERE id=?", (pid,)).fetchone()
            if not row:
                abort(404)
            hist = db.history(c, pid)
        if not hist:
            abort(404)
        times, prices = [h[0] for h in hist], [h[1] for h in hist]
        advice, predicted, verdict = forecast.advise(times, prices)
        points = [{"t": int(t.timestamp() * 1000), "p": p, "s": s} for t, p, s in hist]
        lo_i = min(range(len(prices)), key=prices.__getitem__)
        return render_template(
            "result.html", product=row, title=row["title"] or title_from_url(row["url"]),
            price=prices[-1], advice=advice, predicted=predicted, verdict=verdict,
            stats=forecast.stats(prices), points=points, count=len(prices),
            has_demo=any(h[2] == "demo" for h in hist), lowest_when=times[lo_i],
            history=list(reversed(hist))[:30], sym=currency_symbol(row["url"]),
            is_amazon="amazon." in row["url"],
            asin=row["url"].rsplit("/", 1)[-1] if "/dp/" in row["url"] else None,
        )

    @app.route("/product/<int:pid>/refresh", methods=["POST"])
    def refresh(pid):
        with conn() as c:
            row = c.execute("SELECT * FROM products WHERE id=?", (pid,)).fetchone()
            if not row:
                abort(404)
            try:
                p = app.config["FETCHER"](row["url"])
                db.add_price(c, pid, p.price)
                flash("Price updated.", "ok")
            except ScrapeError as e:
                flash(str(e) + " You can add today's price by hand.", "error")
        return redirect(url_for("product", pid=pid))

    @app.route("/product/<int:pid>/add", methods=["POST"])
    def add_manual(pid):
        price = parse_price(request.form.get("manual_price"))
        if not price:
            flash("Enter the price as a positive number, like 5390.", "error")
        else:
            with conn() as c:
                db.add_price(c, pid, price, source="manual")
            flash("Price added.", "ok")
        return redirect(url_for("product", pid=pid))

    @app.route("/product/<int:pid>/history", methods=["POST"])
    def import_history(pid):
        text = request.form.get("history", "")
        f = request.files.get("history_file")
        if f and f.filename:
            text += "\n" + f.read().decode("utf-8", errors="replace")
        rows, skipped = db.parse_history_text(text)
        if not rows:
            flash("No prices found. Use one per line, like: 2026-08-01, 5990", "error")
        else:
            with conn() as c:
                for when, price in rows:
                    db.add_price(c, pid, price, when, source="import")
            msg = f"Imported {len(rows)} past prices."
            if skipped:
                msg += f" Skipped {skipped} line(s) that weren't in 'date, price' form."
            flash(msg, "ok")
        return redirect(url_for("product", pid=pid))

    @app.route("/product/<int:pid>/sample", methods=["POST"])
    def sample(pid):
        with conn() as c:
            hist = db.history(c, pid)
            if not hist:
                abort(404)
            for when, price in db.make_sample_history(hist[-1][1], pid):
                db.add_price(c, pid, price, when, source="demo")
        flash("Added simulated sample history so you can see how advice and the chart work. It is not real data.", "ok")
        return redirect(url_for("product", pid=pid))

    @app.route("/product/<int:pid>/clear-sample", methods=["POST"])
    def clear_sample(pid):
        with conn() as c:
            c.execute("DELETE FROM prices WHERE product_id=? AND source='demo'", (pid,))
        flash("Sample data removed.", "ok")
        return redirect(url_for("product", pid=pid))

    @app.route("/product/<int:pid>/delete", methods=["POST"])
    def delete(pid):
        with conn() as c:
            c.execute("DELETE FROM products WHERE id=?", (pid,))
        flash("Product removed.", "ok")
        return redirect(url_for("home"))

    if os.environ.get("ENABLE_SCHEDULER") == "1" and (
        not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    ):
        from apscheduler.schedulers.background import BackgroundScheduler

        def refresh_all():
            with conn() as c:
                for r in db.list_products(c):
                    try:
                        db.add_price(c, r["id"], app.config["FETCHER"](r["url"]).price)
                        c.commit()
                    except ScrapeError as e:
                        logging.warning("Scheduled refresh failed for %s: %s", r["url"], e)

        sched = BackgroundScheduler()
        sched.add_job(refresh_all, "interval", hours=1)
        sched.start()

    return app


if __name__ == "__main__":
    create_app().run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
