from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd#handel structured data
from prophet import Prophet #design time series and handle missing data
import json
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

CSV_FILE = 'prices.csv'

def init_csv():
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['timestamp', 'url', 'price'])
        df.to_csv(CSV_FILE, index=False)
    except pd.errors.ParserError:
        logging.error("Error parsing CSV file. Please check the file format.")
        df = pd.DataFrame(columns=['timestamp', 'url', 'price'])
        df.to_csv(CSV_FILE, index=False)

def save_price_to_csv(price, url):
    df = pd.read_csv(CSV_FILE)
    price = float(price.replace(',', ''))  # Remove commas and convert to float
    new_entry = pd.DataFrame({'timestamp': [datetime.now()], 'url': [url], 'price': [price]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    logging.debug(f"Saved price {price} for URL {url}")

def load_data_from_csv(url):
    df = pd.read_csv(CSV_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True, errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    data = df[df['url'] == url]
    logging.debug(f"Loaded data: {data}")
    return data



# Initialize the CSV file
init_csv()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/track', methods=['POST'])
def track():
    url = request.form['url']
    price = get_product_price(url)
    if price != "Price not found":
        save_price_to_csv(price, url)
        advice, stats = generate_advice(url)
        chart_data = get_chart_data(url)
        return render_template('result.html', price=price, advice=advice, stats=stats, chart_data=json.dumps(chart_data))
    else:
        return 'Price not found. Please check the URL or try again later.'

def get_product_price(url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    logging.debug(soup.prettify())  # Log the HTML content
    
    # Try different possible IDs and classes for the price element
    price = soup.find('span', {'id': 'priceblock_ourprice'})
    if not price:
        price = soup.find('span', {'id': 'priceblock_dealprice'})
    if not price:
        price = soup.find('span', {'class': 'a-price-whole'})
    if not price:
        price = soup.find('span', {'class': 'a-offscreen'})
    if not price:
        price = soup.find('span', {'class': 'a-price'})
    if not price:
        price = soup.find('span', {'class': 'a-size-medium a-color-price'})
    
    if price:
        logging.debug(f"Price found: {price.text.strip()}")
        return price.text.strip()
    
    # If price not found, try using Selenium
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    
    try:
        price = driver.find_element(By.ID, 'priceblock_ourprice')
    except:
        try:
            price = driver.find_element(By.ID, 'priceblock_dealprice')
        except:
            try:
                price = driver.find_element(By.CLASS_NAME, 'a-price-whole')
            except:
                try:
                    price = driver.find_element(By.CLASS_NAME, 'a-offscreen')
                except:
                    try:
                        price = driver.find_element(By.CLASS_NAME, 'a-price')
                    except:
                        try:
                            price = driver.find_element(By.CLASS_NAME, 'a-size-medium a-color-price')
                        except:
                            price = None
    
    if price:
        return price.text.strip()
    return "Price not found"

def generate_advice(url):
    data = load_data_from_csv(url)
    logging.debug(f"Data for prediction: {data}")
    if len(data) < 2:
        return "Not enough data to make a prediction.", None
    
    df = data.rename(columns={'timestamp': 'ds', 'price': 'y'})
    model = Prophet()
    model.fit(df)
    
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    
    predicted_price = forecast['yhat'].iloc[-1]
    current_price = df['y'].iloc[-1]
    
    logging.debug(f"Predicted price: {predicted_price}, Current price: {current_price}")
    
    advice = "Wait, the price may drop." if predicted_price < current_price else "Buy now, the price is unlikely to drop."
    
    stats = {
        'highest': data['price'].max(),
        'lowest': data['price'].min(),
        'average': data['price'].mean()
    }
    
    return advice, stats

def get_chart_data(url):
    data = load_data_from_csv(url)
    chart_data = [{
        'x': data['timestamp'].dt.strftime('%Y-%m-%d').tolist(),
        'y': data['price'].tolist(),
        'type': 'scatter'
    }]
    return chart_data

def track_price_periodically():
    # Add the URLs you want to track
    urls = [
        'https://www.amazon.in/dp/B08N5WRWNW',
        'https://www.amazon.in/dp/B08N5VSQNG'
    ]
    for url in urls:
        price = get_product_price(url)
        if price != "Price not found":
            save_price_to_csv(price, url)

# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(track_price_periodically, 'interval', hours=1)
scheduler.start()

if __name__ == '__main__':
    app.run(debug=True)
