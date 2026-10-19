"""Lightweight buy/wait advice from price history (no heavy ML dependency)."""


def stats(prices):
    return {"highest": max(prices), "lowest": min(prices), "average": sum(prices) / len(prices)}


def advise(timestamps, prices):
    """timestamps: list[datetime], prices: list[float] (oldest first).
    Returns (advice, predicted_next_price or None)."""
    n = len(prices)
    current = prices[-1]
    if n < 3:
        return "Not enough history yet - add past prices or check back after a few more refreshes.", None, "info"

    t0 = timestamps[0]
    x = [(t - t0).total_seconds() / 86400 for t in timestamps]
    y = [float(p) for p in prices]
    if x[-1] - x[0] < 1e-6:
        return "Not enough history yet - add past prices or check back after a few more refreshes.", None, "info"

    mx, my = sum(x) / n, sum(y) / n
    var = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / var
    intercept = my - slope * mx
    horizon = max((x[-1] - x[0]) / max(n - 1, 1), 1.0)   # one average interval ahead
    predicted = float(slope * (x[-1] + horizon) + intercept)
    low, avg = min(prices), sum(prices) / n

    if current <= low * 1.02:
        advice, verdict = "Buy now - at or near the lowest price recorded.", "buy"
    elif predicted < current * 0.98 and current > avg:
        advice, verdict = "Wait - the price is above average and trending down.", "wait"
    elif current < avg * 0.95:
        advice, verdict = "Good time to buy - the price is below its average.", "buy"
    elif predicted < current:
        advice, verdict = "Wait - the price may drop a little.", "wait"
    else:
        advice, verdict = "Buy now - the price is unlikely to drop.", "buy"
    return advice, predicted, verdict
