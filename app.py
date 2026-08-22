#!/usr/bin/env python3
"""
Multi-Asset Consensus Trading Bot – KuCoin Edition
- No Binance (uses KuCoin public API – works from Render)
- ATR broadcast fixed
- Telegram event loop fixed
- CSV download endpoint
"""

import os
import time
import csv
import sqlite3
import logging
import threading
import asyncio
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from dotenv import load_dotenv
from flask import Flask, jsonify, send_file
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

# ---------------------------- CONFIGURATION ----------------------------
class Config:
    SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT,MATIC/USDT").split(',') if s.strip()]
    INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "10000.0"))
    MAX_POSITIONS_GLOBAL = int(os.getenv("MAX_POSITIONS_GLOBAL", "5"))
    MAX_POSITIONS_PER_SYMBOL = int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "1"))
    PER_TRADE_RISK_PCT = float(os.getenv("PER_TRADE_RISK_PCT", "0.02"))
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
    CONSENSUS_THRESHOLD = float(os.getenv("CONSENSUS_THRESHOLD", "0.60"))
    CONSECUTIVE_LOSS_LIMIT = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "3"))
    TRADE_INTERVAL_SECONDS = int(os.getenv("TRADE_INTERVAL_SECONDS", "60"))
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    DB_FILE = os.getenv("DB_FILE", "trades.db")
    CSV_FILE = os.getenv("CSV_FILE", "trades.csv")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "")
    EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
    EXCHANGE_SECRET = os.getenv("EXCHANGE_SECRET", "")
    LIVE_TRADING = bool(EXCHANGE_NAME and EXCHANGE_API_KEY and EXCHANGE_SECRET)

    SOURCE_WEIGHTS = {
        "technical_ma": float(os.getenv("WEIGHT_MA", "0.6")),
        "technical_rsi": float(os.getenv("WEIGHT_RSI", "0.4")),
        "orderbook": float(os.getenv("WEIGHT_ORDERBOOK", "0.0")),   # disabled
        "whale": float(os.getenv("WEIGHT_WHALE", "0.0")),           # disabled
        "sentiment": float(os.getenv("WEIGHT_SENTIMENT", "0.5")),
    }

# ---------------------------- LOGGING ----------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, Config.LOG_LEVEL)
)
logger = logging.getLogger("multi-trader")

# ---------------------------- DATABASE ----------------------------
class TradeDB:
    def __init__(self, db_file=Config.DB_FILE):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                symbol TEXT,
                side TEXT,
                price REAL,
                size REAL,
                fee REAL,
                pnl REAL,
                balance REAL
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                symbol TEXT,
                source TEXT,
                direction INTEGER,
                confidence REAL
            )
        ''')
        self.conn.commit()
        self.lock = threading.Lock()

    def log_trade(self, timestamp, symbol, side, price, size, fee=0.0, pnl=0.0, balance=0.0):
        with self.lock:
            self.cursor.execute(
                "INSERT INTO trades VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
                (timestamp, symbol, side, price, size, fee, pnl, balance)
            )
            self.conn.commit()

    def log_signal(self, timestamp, symbol, source, direction, confidence):
        with self.lock:
            self.cursor.execute(
                "INSERT INTO signals VALUES (NULL, ?, ?, ?, ?, ?)",
                (timestamp, symbol, source, direction, confidence)
            )
            self.conn.commit()

    def get_daily_pnl(self):
        today_start = int(datetime.now().replace(hour=0, minute=0, second=0).timestamp())
        with self.lock:
            self.cursor.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE timestamp >= ?",
                (today_start,)
            )
            return self.cursor.fetchone()[0]

    def close(self):
        self.conn.close()

# ---------------------------- CSV PERFORMANCE LOGGER ----------------------------
class PerformanceLogger:
    def __init__(self, csv_file=Config.CSV_FILE):
        self.csv_file = csv_file
        self.lock = threading.Lock()
        if not os.path.isfile(csv_file):
            with open(csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'side', 'entry_price', 'exit_price',
                    'size', 'pnl', 'pnl_pct', 'status', 'balance_after'
                ])

    def log_trade(self, timestamp, symbol, side, entry_price, exit_price, size, pnl, pnl_pct, status, balance_after):
        with self.lock:
            with open(self.csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, symbol, side, entry_price, exit_price,
                    size, pnl, pnl_pct, status, balance_after
                ])

    def get_summary(self):
        if not os.path.isfile(self.csv_file):
            return None
        try:
            rows = []
            with open(self.csv_file, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            if not rows:
                return None
            total_trades = len(rows)
            pnl_sum = 0.0
            wins = 0
            losses = 0
            win_pnls = []
            loss_pnls = []
            best = -1e9
            worst = 1e9
            for r in rows:
                pnl = float(r['pnl'])
                pnl_sum += pnl
                if pnl > 0:
                    wins += 1
                    win_pnls.append(pnl)
                elif pnl < 0:
                    losses += 1
                    loss_pnls.append(pnl)
                if pnl > best:
                    best = pnl
                if pnl < worst:
                    worst = pnl
            win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
            avg_win = sum(win_pnls) / len(win_pnls) if win_pnls else 0
            avg_loss = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0
            last_balance = float(rows[-1]['balance_after']) if rows else 0
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': pnl_sum,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_trade': best if best != -1e9 else 0,
                'worst_trade': worst if worst != 1e9 else 0,
                'current_balance': last_balance
            }
        except Exception as e:
            logger.error(f"CSV read error: {e}")
            return None

# ---------------------------- MARKET DATA (KuCoin – works on Render) ----------------------------
class MarketData:
    def __init__(self, symbol):
        self.symbol = symbol
        self.base, self.quote = symbol.split('/')
        # KuCoin uses hyphen for symbol
        self.kucoin_symbol = f"{self.base}-{self.quote}"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _fetch_kucoin(self, endpoint, params=None):
        """Generic KuCoin GET request."""
        url = f"https://api.kucoin.com{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.error(f"KuCoin {endpoint} error {resp.status_code}: {resp.text[:100]}")
                return None
            data = resp.json()
            if data.get('code') != '200000':
                logger.error(f"KuCoin API error: {data.get('msg')}")
                return None
            return data.get('data')
        except Exception as e:
            logger.error(f"KuCoin request error: {e}")
            return None

    def get_ohlcv(self, limit=100, timeframe='1hour'):
        """Fetch OHLCV from KuCoin."""
        # KuCoin returns latest candles first; we reverse to have oldest first
        params = {
            "symbol": self.kucoin_symbol,
            "type": timeframe,  # 1hour, 1day, etc.
            "limit": limit
        }
        data = self._fetch_kucoin("/api/v1/market/candles", params)
        if not data:
            return None
        # data is list of [time, open, close, high, low, volume]
        # We need to reverse to chronological order
        data = data[::-1]
        ohlcv = []
        for candle in data:
            try:
                ohlcv.append([
                    float(candle[1]),  # open
                    float(candle[3]),  # high
                    float(candle[4]),  # low
                    float(candle[2]),  # close
                    float(candle[5])   # volume
                ])
            except:
                continue
        if len(ohlcv) < 14:
            return None
        return np.array(ohlcv)

    def get_orderbook(self, limit=20):
        """Disabled – we set weight to 0, but keep method for compatibility."""
        return [], []

    def get_recent_trades(self, limit=100):
        """Disabled – weight to 0."""
        return None

    def get_24h_change(self):
        """Fetch 24h stats from KuCoin."""
        params = {"symbol": self.kucoin_symbol}
        data = self._fetch_kucoin("/api/v1/market/stats", params)
        if not data:
            return 0, 0, 0
        try:
            change = float(data.get('changeRate', 0)) * 100  # KuCoin gives decimal
            volume = float(data.get('vol', 0))
            if change > 2 and volume > 1_000_000:
                return 1, change, volume
            elif change < -2 and volume > 1_000_000:
                return -1, change, volume
            return 0, change, volume
        except:
            return 0, 0, 0

# ---------------------------- SIGNAL CLASSES (using only MA, RSI, Sentiment) ----------------------------
class Signal:
    def __init__(self, direction, confidence, source, timestamp=None):
        self.direction = direction
        self.confidence = confidence
        self.source = source
        self.timestamp = timestamp or int(time.time())

class SignalSource:
    def __init__(self, market_data, db):
        self.market = market_data
        self.db = db
    def fetch(self) -> Optional[Signal]:
        raise NotImplementedError

class MASource(SignalSource):
    def fetch(self):
        ohlcv = self.market.get_ohlcv(limit=100, timeframe='1hour')
        if ohlcv is None or len(ohlcv) < 50:
            return None
        close = ohlcv[:, 3]
        ma20 = np.mean(close[-20:])
        ma50 = np.mean(close[-50:])
        price = close[-1]
        if ma20 > ma50 and price > ma20:
            return Signal(+1, 0.55, "technical_ma")
        elif ma20 < ma50 and price < ma20:
            return Signal(-1, 0.55, "technical_ma")
        return Signal(0, 0.0, "technical_ma")

class RSISource(SignalSource):
    def fetch(self):
        ohlcv = self.market.get_ohlcv(limit=100, timeframe='1hour')
        if ohlcv is None or len(ohlcv) < 20:
            return None
        close = ohlcv[:, 3]
        deltas = np.diff(close)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        if len(gains) == 0 or len(losses) == 0:
            return None
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
        if avg_loss == 0:
            return None
        rsi = 100 - (100 / (1 + avg_gain/avg_loss))
        if rsi < 30:
            return Signal(+1, 0.50, "technical_rsi")
        elif rsi > 70:
            return Signal(-1, 0.50, "technical_rsi")
        return Signal(0, 0.0, "technical_rsi")

class OrderBookSource(SignalSource):
    def fetch(self):
        return None  # disabled

class WhaleSource(SignalSource):
    def fetch(self):
        return None  # disabled

class SentimentSource(SignalSource):
    def fetch(self):
        score, _, _ = self.market.get_24h_change()
        if score == 1:
            return Signal(+1, 0.55, "sentiment")
        elif score == -1:
            return Signal(-1, 0.55, "sentiment")
        return Signal(0, 0.0, "sentiment")

# ---------------------------- CONSENSUS ENGINE ----------------------------
class ConsensusEngine:
    def __init__(self, threshold=Config.CONSENSUS_THRESHOLD, weights=Config.SOURCE_WEIGHTS):
        self.threshold = threshold
        self.weights = weights

    def aggregate(self, signals: List[Signal]) -> Tuple[int, float, List[str]]:
        weighted_sum = 0.0
        total_weight = 0.0
        details = []
        for sig in signals:
            if sig.direction == 0:
                continue
            weight = self.weights.get(sig.source, 0.0)
            if weight == 0:
                continue
            weighted_sum += sig.direction * sig.confidence * weight
            total_weight += sig.confidence * weight
            details.append(f"{sig.source}:{sig.direction} ({sig.confidence:.2f})")
        if total_weight == 0:
            return 0, 0.0, details
        avg = weighted_sum / total_weight
        direction = 1 if avg > self.threshold else (-1 if avg < -self.threshold else 0)
        return direction, abs(avg), details

# ---------------------------- RISK MANAGER (unchanged) ----------------------------
class RiskManager:
    def __init__(self, initial_balance, config):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.config = config
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.lock = threading.Lock()

    def update_balance(self, new_balance):
        with self.lock:
            self.balance = new_balance

    def get_total_positions(self):
        total = 0
        for positions in self.open_positions.values():
            total += len(positions)
        return total

    def can_trade(self, symbol):
        with self.lock:
            if self.daily_pnl < -self.config['MAX_DAILY_LOSS_PCT'] * self.initial_balance:
                return False, "Daily loss limit"
            if self.get_total_positions() >= self.config['MAX_POSITIONS_GLOBAL']:
                return False, "Global max positions"
            if self.open_positions.get(symbol, []) and len(self.open_positions[symbol]) >= self.config['MAX_POSITIONS_PER_SYMBOL']:
                return False, "Max per symbol"
            if self.consecutive_losses >= self.config['CONSECUTIVE_LOSS_LIMIT']:
                return False, "Consecutive loss limit"
            return True, "OK"

    def compute_position_size(self, balance, price, atr):
        risk_amount = balance * self.config['PER_TRADE_RISK_PCT']
        if atr is None or atr == 0:
            atr = price * 0.02
        stop_distance = atr * 2.5
        size = risk_amount / stop_distance
        return min(size, (balance * 0.5) / price)

    def open_position(self, symbol, side, price, size, stop_loss, take_profit):
        with self.lock:
            if symbol not in self.open_positions:
                self.open_positions[symbol] = []
            self.open_positions[symbol].append({
                'side': side,
                'entry': price,
                'size': size,
                'sl': stop_loss,
                'tp': take_profit,
                'open_time': time.time()
            })

    def close_position(self, symbol, index, exit_price):
        with self.lock:
            pos = self.open_positions[symbol].pop(index)
            if pos['side'] == 'buy':
                pnl = (exit_price - pos['entry']) * pos['size']
            else:
                pnl = (pos['entry'] - exit_price) * pos['size']
            return pnl, pos

    def check_sl_tp(self, symbol, current_price):
        with self.lock:
            if symbol not in self.open_positions:
                return 0, None, None
            for i, pos in enumerate(self.open_positions[symbol]):
                if pos['side'] == 'buy':
                    if current_price <= pos['sl']:
                        return self.close_position(symbol, i, current_price) + ('SL',)
                    elif current_price >= pos['tp']:
                        return self.close_position(symbol, i, current_price) + ('TP',)
                else:
                    if current_price >= pos['sl']:
                        return self.close_position(symbol, i, current_price) + ('SL',)
                    elif current_price <= pos['tp']:
                        return self.close_position(symbol, i, current_price) + ('TP',)
            return 0, None, None

    def update_daily_pnl(self, pnl):
        with self.lock:
            self.daily_pnl += pnl
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

# ---------------------------- LIVE BROKER ----------------------------
class LiveBroker:
    def __init__(self, exchange_name, api_key, secret):
        self.enabled = bool(exchange_name and api_key and secret)
        if self.enabled:
            logger.info(f"Live broker initialized for {exchange_name}")
        else:
            logger.info("Live broker disabled – paper trading only.")

    def place_order(self, symbol, side, price, size):
        if not self.enabled:
            logger.info(f"[PAPER] {side} {size:.4f} {symbol} @ ${price:.2f}")
            return {"status": "paper"}
        logger.info(f"[LIVE] {side} {size} {symbol} at {price}")
        return {"status": "live_placeholder"}

# ---------------------------- MULTI-ASSET TRADER ----------------------------
class MultiTrader:
    def __init__(self, symbols, initial_balance, risk_mgr, db, telegram_token, chat_id, live_broker):
        self.symbols = symbols
        self.db = db
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.live_broker = live_broker
        self.risk_mgr = risk_mgr
        self.balance = initial_balance
        self.markets = {sym: MarketData(sym) for sym in symbols}
        self.consensus = ConsensusEngine()
        self.sources = {}
        for sym in symbols:
            self.sources[sym] = [
                MASource(self.markets[sym], db),
                RSISource(self.markets[sym], db),
                SentimentSource(self.markets[sym], db)
            ]
        self.running = True
        self.performance_logger = PerformanceLogger(Config.CSV_FILE)

    def send_alert(self, message):
        if not self.telegram_token or not self.chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"},
                timeout=30
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def get_price_and_atr(self, symbol):
        ohlcv = self.markets[symbol].get_ohlcv(limit=50, timeframe='1hour')
        if ohlcv is None or len(ohlcv) < 20:
            return None, None
        close = ohlcv[:, 3]
        high = ohlcv[:, 1]
        low = ohlcv[:, 2]
        high_curr = high[1:]
        low_curr = low[1:]
        prev_close = close[:-1]
        tr1 = high_curr - low_curr
        tr2 = np.abs(high_curr - prev_close)
        tr3 = np.abs(low_curr - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        return close[-1], atr

    def execute_signal(self, symbol, direction, confidence, price, atr, details, sentiment_score):
        can_trade, _ = self.risk_mgr.can_trade(symbol)
        if not can_trade:
            return
        size = self.risk_mgr.compute_position_size(self.balance, price, atr)
        if size <= 0:
            return
        risk = atr * 2.5
        stop_loss = price - risk if direction == 1 else price + risk
        take_profit = price + risk * 1.5 if direction == 1 else price - risk * 1.5
        side = 'buy' if direction == 1 else 'sell'
        cost = price * size
        if side == 'buy' and self.balance < cost:
            self.send_alert(f"⚠️ Insufficient balance for {symbol}")
            return
        self.risk_mgr.open_position(symbol, side, price, size, stop_loss, take_profit)
        if side == 'buy':
            self.balance -= cost
        else:
            self.balance += price * size
        self.risk_mgr.update_balance(self.balance)
        self.db.log_trade(int(time.time()), symbol, side, price, size, 0.0, 0.0, self.balance)

        details_html = "<br>".join([f"• {d}" for d in details])
        sentiment_label = "🚀 Bullish" if sentiment_score == 1 else ("🔻 Bearish" if sentiment_score == -1 else "⚖️ Neutral")
        msg = (
            f"🔔 <b>{symbol} SIGNAL</b> ({'LIVE' if self.live_broker.enabled else 'PAPER'})\n"
            f"Action: {'🟢 BUY' if direction==1 else '🔴 SELL'}\n"
            f"Entry: ${price:.4f}\n"
            f"TP: ${take_profit:.4f} (+{(risk*1.5/price)*100:.2f}%)\n"
            f"SL: ${stop_loss:.4f} (-{(risk/price)*100:.2f}%)\n"
            f"Risk: {self.risk_mgr.config['PER_TRADE_RISK_PCT']*100:.1f}%\n"
            f"Confidence: {confidence:.2f}\n"
            f"Sentiment: {sentiment_label}\n"
            f"Votes:\n{details_html}\n\n"
            f"<i>Not financial advice.</i>"
        )
        self.send_alert(msg)

    def step(self):
        for symbol in self.symbols:
            price, atr = self.get_price_and_atr(symbol)
            if price is None or atr is None:
                continue
            pnl, status, pos = self.risk_mgr.check_sl_tp(symbol, price)
            if pnl != 0 and pos is not None:
                self.balance += pnl
                self.risk_mgr.update_balance(self.balance)
                self.risk_mgr.update_daily_pnl(pnl)
                pnl_pct = (pnl / (pos['entry'] * pos['size'])) * 100
                self.performance_logger.log_trade(
                    int(time.time()), symbol, pos['side'],
                    pos['entry'], price, pos['size'],
                    pnl, pnl_pct, status, self.balance
                )
                emoji = "✅" if pnl > 0 else "❌"
                self.send_alert(
                    f"{emoji} <b>{symbol} CLOSED</b> ({status})\n"
                    f"Entry: ${pos['entry']:.4f} | Exit: ${price:.4f}\n"
                    f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | Balance: ${self.balance:.2f}"
                )
            if self.risk_mgr.open_positions.get(symbol, []):
                continue
            signals = []
            sentiment = 0
            for src in self.sources[symbol]:
                sig = src.fetch()
                if sig and sig.direction != 0:
                    signals.append(sig)
                    self.db.log_signal(int(time.time()), symbol, sig.source, sig.direction, sig.confidence)
                    if sig.source == "sentiment":
                        sentiment = sig.direction
            if signals:
                direction, conf, details = self.consensus.aggregate(signals)
                if direction != 0 and conf >= Config.CONSENSUS_THRESHOLD:
                    self.execute_signal(symbol, direction, conf, price, atr, details, sentiment)

    def run_loop(self):
        logger.info("Starting multi-asset trading loop. Symbols: %s", self.symbols)
        while self.running:
            try:
                self.step()
                time.sleep(Config.TRADE_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(Config.TRADE_INTERVAL_SECONDS)

# ---------------------------- FLASK APP ----------------------------
app = Flask(__name__)
trader_global = None

@app.route('/')
def health():
    return jsonify({"status": "running", "version": "kucoin", "time": datetime.now().isoformat()})

@app.route('/download')
def download_csv():
    if os.path.exists(Config.CSV_FILE):
        return send_file(Config.CSV_FILE, as_attachment=True, download_name="trades.csv")
    return jsonify({"error": "No trades.csv yet"}), 404

@app.route('/status')
def status():
    if trader_global is None:
        return jsonify({"error": "trader not initialized"})
    return jsonify({
        "balance": trader_global.balance,
        "daily_pnl": trader_global.db.get_daily_pnl(),
        "open_positions": trader_global.risk_mgr.get_total_positions(),
        "running": trader_global.running
    })

# ---------------------------- TELEGRAM BOT ----------------------------
def get_main_keyboard():
    buttons = [
        [KeyboardButton("📊 Status"), KeyboardButton("🔍 Scan")],
        [KeyboardButton("📈 Performance"), KeyboardButton("⏸️ Pause"), KeyboardButton("▶️ Resume")],
        [KeyboardButton("❓ Help")]
    ]
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=False)

async def start(update: Update, context):
    await update.message.reply_text(
        "🤖 <b>Consensus Trader</b>\n\nUse buttons or commands:\n"
        "/status – Account\n/scan – Force scan\n/performance – Stats\n"
        "/pause – Pause\n/resume – Resume\n/help – This\n\n"
        "💾 <a href='https://new-crypto-signals.onrender.com/download'>Download CSV</a>",
        parse_mode='HTML', reply_markup=get_main_keyboard(), disable_web_page_preview=True
    )

async def help_cmd(update: Update, context):
    await update.message.reply_text("Commands: /status, /scan, /performance, /pause, /resume, /help", reply_markup=get_main_keyboard())

async def status_cmd(update: Update, context):
    if trader_global is None:
        await update.message.reply_text("Trader not ready.", reply_markup=get_main_keyboard())
        return
    t = trader_global
    await update.message.reply_text(
        f"📊 <b>Status</b>\nBalance: ${t.balance:.2f}\nDaily PnL: ${t.db.get_daily_pnl():.2f}\n"
        f"Open Positions: {t.risk_mgr.get_total_positions()}\nRunning: {'✅' if t.running else '⏸️'}",
        parse_mode='HTML', reply_markup=get_main_keyboard()
    )

async def performance(update: Update, context):
    if trader_global is None:
        await update.message.reply_text("Trader not ready.", reply_markup=get_main_keyboard())
        return
    summary = trader_global.performance_logger.get_summary()
    if not summary:
        await update.message.reply_text("No trades yet.", reply_markup=get_main_keyboard())
        return
    await update.message.reply_text(
        f"📈 <b>Performance</b>\nTrades: {summary['total_trades']}\nWin Rate: {summary['win_rate']:.1f}%\n"
        f"Total PnL: ${summary['total_pnl']:.2f}\nAvg Win: ${summary['avg_win']:.2f}\n"
        f"Avg Loss: ${summary['avg_loss']:.2f}\nBest: ${summary['best_trade']:.2f}\n"
        f"Worst: ${summary['worst_trade']:.2f}\nBalance: ${summary['current_balance']:.2f}",
        parse_mode='HTML', reply_markup=get_main_keyboard()
    )

async def scan(update: Update, context):
    await update.message.reply_text("🔍 Scanning...", reply_markup=get_main_keyboard())
    if trader_global is None:
        return
    for sym in trader_global.symbols:
        price, atr = trader_global.get_price_and_atr(sym)
        if price is None:
            await update.message.reply_text(f"⚠️ No data for {sym}")
            continue
        signals = []
        for src in trader_global.sources[sym]:
            sig = src.fetch()
            if sig and sig.direction != 0:
                signals.append(sig)
        if signals:
            direction, conf, details = trader_global.consensus.aggregate(signals)
            await update.message.reply_text(
                f"⚖️ {sym}: {'BUY' if direction==1 else 'SELL' if direction==-1 else 'NEUTRAL'}\n"
                f"Confidence: {conf:.2f}\nSources: {' | '.join(details)}"
            )
        else:
            await update.message.reply_text(f"⚠️ No signals for {sym}")
    await update.message.reply_text("✅ Scan complete.", reply_markup=get_main_keyboard())

async def pause(update: Update, context):
    if trader_global:
        trader_global.running = False
    await update.message.reply_text("⏸️ Paused.", reply_markup=get_main_keyboard())

async def resume(update: Update, context):
    if trader_global:
        trader_global.running = True
    await update.message.reply_text("▶️ Resumed.", reply_markup=get_main_keyboard())

async def handle_button(update: Update, context):
    text = update.message.text
    if text == "📊 Status":
        await status_cmd(update, context)
    elif text == "🔍 Scan":
        await scan(update, context)
    elif text == "📈 Performance":
        await performance(update, context)
    elif text == "⏸️ Pause":
        await pause(update, context)
    elif text == "▶️ Resume":
        await resume(update, context)
    elif text == "❓ Help":
        await help_cmd(update, context)

def run_telegram():
    if not Config.TELEGRAM_TOKEN:
        return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app_tg = Application.builder().token(Config.TELEGRAM_TOKEN).build()
    app_tg.add_handler(CommandHandler("start", start))
    app_tg.add_handler(CommandHandler("help", help_cmd))
    app_tg.add_handler(CommandHandler("status", status_cmd))
    app_tg.add_handler(CommandHandler("performance", performance))
    app_tg.add_handler(CommandHandler("scan", scan))
    app_tg.add_handler(CommandHandler("pause", pause))
    app_tg.add_handler(CommandHandler("resume", resume))
    app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_button))
    app_tg.run_polling()

# ---------------------------- MAIN ----------------------------
if __name__ == "__main__":
    # Flask
    flask_thread = threading.Thread(target=app.run, kwargs={'host':'0.0.0.0', 'port':int(os.getenv('PORT', 5000))})
    flask_thread.daemon = True
    flask_thread.start()

    # Telegram
    if Config.TELEGRAM_TOKEN:
        tg_thread = threading.Thread(target=run_telegram)
        tg_thread.daemon = True
        tg_thread.start()
        logger.info("Telegram bot started.")

    # Live broker
    live_broker = LiveBroker(Config.EXCHANGE_NAME, Config.EXCHANGE_API_KEY, Config.EXCHANGE_SECRET)

    # DB & Risk
    db = TradeDB(Config.DB_FILE)
    risk_mgr = RiskManager(Config.INITIAL_BALANCE, {
        'MAX_DAILY_LOSS_PCT': Config.MAX_DAILY_LOSS_PCT,
        'MAX_POSITIONS_GLOBAL': Config.MAX_POSITIONS_GLOBAL,
        'MAX_POSITIONS_PER_SYMBOL': Config.MAX_POSITIONS_PER_SYMBOL,
        'CONSECUTIVE_LOSS_LIMIT': Config.CONSECUTIVE_LOSS_LIMIT,
        'PER_TRADE_RISK_PCT': Config.PER_TRADE_RISK_PCT,
    })

    trader = MultiTrader(
        symbols=Config.SYMBOLS,
        initial_balance=Config.INITIAL_BALANCE,
        risk_mgr=risk_mgr,
        db=db,
        telegram_token=Config.TELEGRAM_TOKEN,
        chat_id=Config.TELEGRAM_CHAT_ID,
        live_broker=live_broker
    )
    trader_global = trader
    trader.run_loop()