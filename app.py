#!/usr/bin/env python3
"""
Multi-Asset Consensus Trading Bot – Fully Fixed
- HTML Telegram messages (no Markdown parse errors)
- Synchronous requests for Telegram (no event loop conflicts)
- TP/SL execution + CSV logging
- Reply keyboard with buttons
- Orderbook empty guard
- No risk-block alerts
"""

import os
import time
import json
import csv
import sqlite3
import logging
import threading
import requests
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

# ---------------------------- CONFIGURATION ----------------------------
class Config:
    SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT,MATIC/USDT,DOGE/USDT,ADA/USDT,DOT/USDT").split(',') if s.strip()]
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
        "orderbook": float(os.getenv("WEIGHT_ORDERBOOK", "0.8")),
        "whale": float(os.getenv("WEIGHT_WHALE", "0.7")),
        "sentiment": float(os.getenv("WEIGHT_SENTIMENT", "0.5")),
    }

# ---------------------------- LOGGING ----------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, Config.LOG_LEVEL)
)
logger = logging.getLogger("multi-trader")

# ---------------------------- DATABASE (SQLite) ----------------------------
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

    def get_trade_count_today(self):
        today_start = int(datetime.now().replace(hour=0, minute=0, second=0).timestamp())
        with self.lock:
            self.cursor.execute(
                "SELECT COUNT(*) FROM trades WHERE timestamp >= ? AND side IN ('buy','sell')",
                (today_start,)
            )
            return self.cursor.fetchone()[0]

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
            df = pd.read_csv(self.csv_file)
            if df.empty:
                return None
            total_trades = len(df)
            wins = df[df['pnl'] > 0]
            losses = df[df['pnl'] < 0]
            win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
            total_pnl = df['pnl'].sum()
            avg_win = wins['pnl'].mean() if not wins.empty else 0
            avg_loss = losses['pnl'].mean() if not losses.empty else 0
            best_trade = df['pnl'].max()
            worst_trade = df['pnl'].min()
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'current_balance': df.iloc[-1]['balance_after'] if not df.empty else 0
            }
        except Exception as e:
            logger.error(f"CSV read error: {e}")
            return None

# ---------------------------- MARKET DATA (FREE APIS) ----------------------------
class MarketData:
    def __init__(self, symbol):
        self.symbol = symbol
        self.base, self.quote = symbol.split('/')

    def get_ohlcv(self, limit=100, timeframe='1h'):
        endpoint = "https://api.binance.com/api/v3/klines"
        params = {"symbol": self.base+self.quote, "interval": timeframe, "limit": limit}
        try:
            resp = requests.get(endpoint, params=params, timeout=10)
            data = resp.json()
            df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume','close_time',
                                             'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume','ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            return df
        except Exception as e:
            logger.error(f"OHLCV error for {self.symbol}: {e}")
            return None

    def get_orderbook(self, limit=20):
        endpoint = "https://api.binance.com/api/v3/depth"
        params = {"symbol": self.base+self.quote, "limit": limit}
        try:
            resp = requests.get(endpoint, params=params, timeout=5)
            data = resp.json()
            bids = [(float(p), float(q)) for p,q in data['bids'][:limit]]
            asks = [(float(p), float(q)) for p,q in data['asks'][:limit]]
            return bids, asks
        except Exception as e:
            logger.error(f"Orderbook error for {self.symbol}: {e}")
            return [], []

    def get_recent_trades(self, limit=100):
        endpoint = "https://api.binance.com/api/v3/trades"
        params = {"symbol": self.base+self.quote, "limit": limit}
        try:
            resp = requests.get(endpoint, params=params, timeout=5)
            data = resp.json()
            trades = []
            for t in data:
                trades.append({
                    'price': float(t['price']),
                    'qty': float(t['qty']),
                    'time': t['time'],
                    'isBuyerMaker': t['isBuyerMaker']
                })
            return trades
        except Exception as e:
            logger.error(f"Recent trades error for {self.symbol}: {e}")
            return None

    def get_24h_change(self):
        endpoint = "https://api.binance.com/api/v3/ticker/24hr"
        params = {"symbol": self.base+self.quote}
        try:
            resp = requests.get(endpoint, params=params, timeout=5)
            data = resp.json()
            change = float(data['priceChangePercent'])
            volume = float(data['quoteVolume'])
            if change > 2 and volume > 1_000_000:
                return 1, change, volume
            elif change < -2 and volume > 1_000_000:
                return -1, change, volume
            return 0, change, volume
        except Exception as e:
            logger.error(f"24h change error for {self.symbol}: {e}")
            return 0, 0, 0

# ---------------------------- SIGNAL CLASSES ----------------------------
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
        df = self.market.get_ohlcv(limit=100, timeframe='1h')
        if df is None or len(df) < 50:
            return None
        close = df['close']
        ma_short = close.rolling(20).mean().iloc[-1]
        ma_long = close.rolling(50).mean().iloc[-1]
        price = close.iloc[-1]
        if pd.isna(ma_short) or pd.isna(ma_long):
            return None
        if ma_short > ma_long and price > ma_short:
            return Signal(+1, 0.55, "technical_ma")
        elif ma_short < ma_long and price < ma_short:
            return Signal(-1, 0.55, "technical_ma")
        return Signal(0, 0.0, "technical_ma")

class RSISource(SignalSource):
    def fetch(self):
        df = self.market.get_ohlcv(limit=100, timeframe='1h')
        if df is None:
            return None
        close = df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        if loss.iloc[-1] == 0:
            return None
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        if pd.isna(last_rsi):
            return None
        if last_rsi < 30:
            return Signal(+1, 0.50, "technical_rsi")
        elif last_rsi > 70:
            return Signal(-1, 0.50, "technical_rsi")
        return Signal(0, 0.0, "technical_rsi")

class OrderBookSource(SignalSource):
    def fetch(self):
        bids, asks = self.market.get_orderbook(limit=20)
        if not bids or not asks or len(bids) < 1 or len(asks) < 1:
            return None
        bid_price, bid_qty = bids[0]
        ask_price, ask_qty = asks[0]
        total_bid_vol = sum(qty for _, qty in bids[:10])
        total_ask_vol = sum(qty for _, qty in asks[:10])
        if total_bid_vol == 0 or total_ask_vol == 0:
            return None
        micro_price = (bid_price * total_ask_vol + ask_price * total_bid_vol) / (total_bid_vol + total_ask_vol)
        mid_price = (bid_price + ask_price) / 2
        if micro_price > mid_price * 1.002:
            return Signal(+1, 0.65, "orderbook")
        elif micro_price < mid_price * 0.998:
            return Signal(-1, 0.65, "orderbook")
        return Signal(0, 0.0, "orderbook")

class WhaleSource(SignalSource):
    def __init__(self, market_data, db, threshold_usd=50000):
        super().__init__(market_data, db)
        self.threshold_usd = threshold_usd
    def fetch(self):
        trades = self.market.get_recent_trades(limit=50)
        if trades is None:
            return None
        bids, asks = self.market.get_orderbook(1)
        if not bids or not asks:
            return None
        current_price = (bids[0][0] + asks[0][0]) / 2
        for t in trades:
            trade_usd = t['price'] * t['qty']
            if trade_usd > self.threshold_usd:
                if not t['isBuyerMaker']:
                    return Signal(+1, 0.70, "whale")
                else:
                    return Signal(-1, 0.70, "whale")
        return Signal(0, 0.0, "whale")

class SentimentSource(SignalSource):
    def fetch(self):
        score, change, volume = self.market.get_24h_change()
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
            weight = self.weights.get(sig.source, 1.0)
            weighted_sum += sig.direction * sig.confidence * weight
            total_weight += sig.confidence * weight
            details.append(f"{sig.source}:{sig.direction} ({sig.confidence:.2f})")
        if total_weight == 0:
            return 0, 0.0, details
        avg = weighted_sum / total_weight
        direction = 1 if avg > self.threshold else (-1 if avg < -self.threshold else 0)
        return direction, abs(avg), details

# ---------------------------- RISK MANAGER (with TP) ----------------------------
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
                return False, f"Daily loss limit reached (${self.daily_pnl:.2f})"
            if self.get_total_positions() >= self.config['MAX_POSITIONS_GLOBAL']:
                return False, f"Global max positions ({self.config['MAX_POSITIONS_GLOBAL']})"
            if self.open_positions.get(symbol, []) and len(self.open_positions[symbol]) >= self.config['MAX_POSITIONS_PER_SYMBOL']:
                return False, f"Max per symbol ({self.config['MAX_POSITIONS_PER_SYMBOL']})"
            if self.consecutive_losses >= self.config['CONSECUTIVE_LOSS_LIMIT']:
                return False, "Consecutive loss limit"
            return True, "OK"

    def compute_position_size(self, balance, price, atr):
        risk_amount = balance * self.config['PER_TRADE_RISK_PCT']
        if atr is None or atr == 0:
            atr = price * 0.02
        stop_distance = atr * 2.5
        size = risk_amount / stop_distance
        max_size = (balance * 0.5) / price
        return min(size, max_size)

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
                        pnl, closed_pos = self.close_position(symbol, i, current_price)
                        return pnl, 'SL', closed_pos
                    elif current_price >= pos['tp']:
                        pnl, closed_pos = self.close_position(symbol, i, current_price)
                        return pnl, 'TP', closed_pos
                else:  # sell
                    if current_price >= pos['sl']:
                        pnl, closed_pos = self.close_position(symbol, i, current_price)
                        return pnl, 'SL', closed_pos
                    elif current_price <= pos['tp']:
                        pnl, closed_pos = self.close_position(symbol, i, current_price)
                        return pnl, 'TP', closed_pos
            return 0, None, None

    def update_daily_pnl(self, pnl):
        with self.lock:
            self.daily_pnl += pnl
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

# ---------------------------- LIVE BROKER (Placeholder) ----------------------------
class LiveBroker:
    def __init__(self, exchange_name, api_key, secret):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.secret = secret
        self.enabled = bool(exchange_name and api_key and secret)
        if self.enabled:
            logger.info(f"Live broker initialized for {exchange_name}")
        else:
            logger.info("Live broker disabled – paper trading only.")

    def place_order(self, symbol, side, price, size, order_type='limit'):
        if not self.enabled:
            logger.info(f"[PAPER] Would place {side} {size} {symbol} at {price}")
            return {"status": "paper", "symbol": symbol, "side": side, "price": price, "size": size}
        logger.info(f"[LIVE] Executing {side} {size} {symbol} at {price} on {self.exchange_name}")
        return {"status": "live_placeholder", "symbol": symbol, "side": side, "price": price, "size": size}

# ---------------------------- MULTI-ASSET TRADER (with synchronous Telegram) ----------------------------
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
                OrderBookSource(self.markets[sym], db),
                WhaleSource(self.markets[sym], db),
                SentimentSource(self.markets[sym], db)
            ]
        self.running = True
        self.last_prices = {}
        self.last_sentiment = {}
        self.performance_logger = PerformanceLogger(Config.CSV_FILE)

    # ---- Synchronous Telegram send via requests ----
    def send_alert(self, message):
        if not self.telegram_token or not self.chat_id:
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    def get_price_and_atr(self, symbol):
        df = self.markets[symbol].get_ohlcv(limit=50, timeframe='1h')
        if df is None or len(df) < 14:
            return None, None
        close = df['close']
        high = df['high']
        low = df['low']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        price = close.iloc[-1]
        return price, atr

    def execute_signal(self, symbol, direction, confidence, price, atr, details, sentiment_score):
        can_trade, reason = self.risk_mgr.can_trade(symbol)
        if not can_trade:
            logger.info(f"Risk block for {symbol}: {reason}")
            return

        size = self.risk_mgr.compute_position_size(self.balance, price, atr)
        if size <= 0:
            return

        risk = atr * 2.5
        stop_loss = price - risk if direction == 1 else price + risk
        take_profit = price + (risk * 1.5) if direction == 1 else price - (risk * 1.5)

        sl_pct = (risk / price) * 100
        tp_pct = ((risk * 1.5) / price) * 100
        risk_pct = self.risk_mgr.config['PER_TRADE_RISK_PCT'] * 100

        sentiment_label = "🚀 Bullish" if sentiment_score == 1 else ("🔻 Bearish" if sentiment_score == -1 else "⚖️ Neutral")
        side = 'buy' if direction == 1 else 'sell'
        action_emoji = "🟢 BUY" if direction == 1 else "🔴 SELL"
        live_tag = "LIVE" if self.live_broker.enabled else "PAPER"

        if self.live_broker.enabled:
            self.live_broker.place_order(symbol, side, price, size)
        else:
            cost = price * size
            if side == 'buy' and self.balance < cost:
                self.send_alert(f"⚠️ Insufficient balance for {symbol} buy")
                return
            self.risk_mgr.open_position(symbol, side, price, size, stop_loss, take_profit)
            if side == 'buy':
                self.balance -= cost
            else:
                self.balance += price * size
            self.risk_mgr.update_balance(self.balance)
            self.db.log_trade(int(time.time()), symbol, side, price, size, 0.0, 0.0, self.balance)

        details_html = "<br>".join([f"• {d}" for d in details])
        msg = (
            f"🔔 <b>{symbol} SIGNAL</b> ({live_tag})\n\n"
            f"<b>Action:</b> {action_emoji}\n"
            f"<b>Entry:</b> ${price:.4f}\n"
            f"<b>Target (TP):</b> ${take_profit:.4f} (+{tp_pct:.2f}%)\n"
            f"<b>Stop (SL):</b> ${stop_loss:.4f} (-{sl_pct:.2f}%)\n\n"
            f"<b>Risk:</b> {risk_pct:.1f}% of portfolio\n"
            f"<b>Confidence:</b> {confidence:.2f}\n"
            f"<b>Sentiment:</b> {sentiment_label}\n\n"
            f"<b>Strategy Votes:</b>\n{details_html}\n\n"
            f"<i>Not financial advice. Trade at your own risk.</i>"
        )
        self.send_alert(msg)

    def step(self):
        for symbol in self.symbols:
            price, atr = self.get_price_and_atr(symbol)
            if price is None or atr is None:
                continue

            # 1. Check SL / TP
            pnl, status, pos = self.risk_mgr.check_sl_tp(symbol, price)
            if pnl != 0 and pos is not None:
                self.balance += pnl
                self.risk_mgr.update_balance(self.balance)
                self.risk_mgr.update_daily_pnl(pnl)
                pnl_pct = (pnl / (pos['entry'] * pos['size'])) * 100
                self.performance_logger.log_trade(
                    timestamp=int(time.time()),
                    symbol=symbol,
                    side=pos['side'],
                    entry_price=pos['entry'],
                    exit_price=price,
                    size=pos['size'],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    status=status,
                    balance_after=self.balance
                )
                emoji = "✅" if pnl > 0 else "❌"
                msg = (
                    f"{emoji} <b>{symbol} POSITION CLOSED</b>\n"
                    f"<b>Reason:</b> {status}\n"
                    f"<b>Entry:</b> ${pos['entry']:.4f}\n"
                    f"<b>Exit:</b> ${price:.4f}\n"
                    f"<b>PnL:</b> ${pnl:.2f} ({pnl_pct:+.2f}%)\n"
                    f"<b>Balance:</b> ${self.balance:.2f}"
                )
                self.send_alert(msg)

            # 2. Skip if already in position
            if self.risk_mgr.open_positions.get(symbol, []):
                continue

            signals = []
            sentiment_score = 0
            for src in self.sources[symbol]:
                sig = src.fetch()
                if sig and sig.direction != 0:
                    signals.append(sig)
                    self.db.log_signal(int(time.time()), symbol, sig.source, sig.direction, sig.confidence)
                    if sig.source == "sentiment":
                        sentiment_score = sig.direction

            if signals:
                direction, conf, details = self.consensus.aggregate(signals)
                if direction != 0 and conf >= Config.CONSENSUS_THRESHOLD:
                    self.execute_signal(symbol, direction, conf, price, atr, details, sentiment_score)

            self.last_prices[symbol] = price
            self.last_sentiment[symbol] = sentiment_score

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
trader_global = None  # will be set in main

@app.route('/')
def health():
    return jsonify({"status": "running", "version": "multi-asset", "time": datetime.now().isoformat()})

@app.route('/status')
def status():
    if trader_global is None:
        return jsonify({"error": "trader not initialized"})
    balance = trader_global.balance
    daily_pnl = trader_global.db.get_daily_pnl()
    total_pos = trader_global.risk_mgr.get_total_positions()
    return jsonify({
        "balance": balance,
        "daily_pnl": daily_pnl,
        "open_positions_total": total_pos,
        "running": trader_global.running
    })

# ---------------------------- TELEGRAM BOT HANDLERS ----------------------------
def get_main_keyboard():
    buttons = [
        [KeyboardButton("📊 Status"), KeyboardButton("🔍 Scan")],
        [KeyboardButton("📈 Performance"), KeyboardButton("⏸️ Pause"), KeyboardButton("▶️ Resume")],
        [KeyboardButton("❓ Help")]
    ]
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=False)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = get_main_keyboard()
    welcome_text = (
        "🤖 <b>Welcome to Multi‑Asset Consensus Trader!</b>\n\n"
        "I scan multiple coins using 5 independent strategies, "
        "aggregate consensus, and execute paper trades with TP/SL.\n\n"
        "Use the buttons below or type commands:\n"
        "📊 /status – Account summary\n"
        "🔍 /scan – Force signal scan\n"
        "📈 /performance – Trade stats\n"
        "⏸️ /pause – Pause trading\n"
        "▶️ /resume – Resume trading\n"
        "❓ /help – Show this menu\n\n"
        "<i>Paper trading only. Not financial advice.</i>"
    )
    await update.message.reply_text(welcome_text, parse_mode='HTML', reply_markup=keyboard)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = get_main_keyboard()
    help_text = (
        "📋 <b>Available Commands</b>\n\n"
        "📊 /status – Show current balance, daily PnL, open positions\n"
        "🔍 /scan – Manually scan all symbols for consensus signals\n"
        "📈 /performance – Show trade win rate, total PnL, best/worst\n"
        "⏸️ /pause – Pause the trading loop\n"
        "▶️ /resume – Resume trading\n"
        "❓ /help – Show this menu\n\n"
        "💡 You can also use the buttons below."
    )
    await update.message.reply_text(help_text, parse_mode='HTML', reply_markup=keyboard)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if trader_global is None:
        await update.message.reply_text("Trader not initialized yet.", reply_markup=get_main_keyboard())
        return
    trader = trader_global
    balance = trader.balance
    daily_pnl = trader.db.get_daily_pnl()
    total_pos = trader.risk_mgr.get_total_positions()
    running = trader.running
    msg = (f"📊 <b>ACCOUNT STATUS</b>\n"
           f"💰 Balance: ${balance:.2f}\n"
           f"📉 Daily PnL: ${daily_pnl:.2f}\n"
           f"📌 Open Positions: {total_pos}\n"
           f"⏳ Running: {'✅' if running else '⏸️'}")
    await update.message.reply_text(msg, parse_mode='HTML', reply_markup=get_main_keyboard())

async def performance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if trader_global is None:
        await update.message.reply_text("Trader not initialized yet.", reply_markup=get_main_keyboard())
        return
    summary = trader_global.performance_logger.get_summary()
    if summary is None:
        await update.message.reply_text("No trades recorded yet.", reply_markup=get_main_keyboard())
        return
    msg = (f"📈 <b>PERFORMANCE SUMMARY</b>\n"
           f"📊 Total Trades: {summary['total_trades']}\n"
           f"🏆 Win Rate: {summary['win_rate']:.1f}%\n"
           f"💰 Total PnL: ${summary['total_pnl']:.2f}\n"
           f"📈 Avg Win: ${summary['avg_win']:.2f}\n"
           f"📉 Avg Loss: ${summary['avg_loss']:.2f}\n"
           f"🌟 Best Trade: ${summary['best_trade']:.2f}\n"
           f"💀 Worst Trade: ${summary['worst_trade']:.2f}\n"
           f"💵 Current Balance: ${summary['current_balance']:.2f}")
    await update.message.reply_text(msg, parse_mode='HTML', reply_markup=get_main_keyboard())

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Scanning all symbols for signals...", reply_markup=get_main_keyboard())
    if trader_global is None:
        await update.message.reply_text("Trader not initialized.", reply_markup=get_main_keyboard())
        return
    trader = trader_global
    for symbol in trader.symbols:
        price, atr = trader.get_price_and_atr(symbol)
        if price is None:
            continue
        signals = []
        for src in trader.sources[symbol]:
            sig = src.fetch()
            if sig and sig.direction != 0:
                signals.append(sig)
        if signals:
            direction, conf, details = trader.consensus.aggregate(signals)
            detail_str = " | ".join(details)
            msg = (f"⚖️ {symbol} Consensus: {'BUY' if direction==1 else 'SELL' if direction==-1 else 'NEUTRAL'}\n"
                   f"📊 Conf: {conf:.2f}\n"
                   f"📋 Sources: {detail_str}")
            await update.message.reply_text(msg, reply_markup=get_main_keyboard())
        else:
            await update.message.reply_text(f"⚠️ No signals for {symbol}", reply_markup=get_main_keyboard())
    await update.message.reply_text("✅ Scan complete.", reply_markup=get_main_keyboard())

async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if trader_global is not None:
        trader_global.running = False
    await update.message.reply_text("⏸️ Trading paused.", reply_markup=get_main_keyboard())

async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if trader_global is not None:
        trader_global.running = True
    await update.message.reply_text("▶️ Trading resumed.", reply_markup=get_main_keyboard())

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

# ---------------------------- MAIN ENTRY ----------------------------
if __name__ == "__main__":
    # Flask
    flask_thread = threading.Thread(target=app.run, kwargs={'host':'0.0.0.0', 'port':int(os.getenv('PORT', 5000))})
    flask_thread.daemon = True
    flask_thread.start()

    # Telegram (only for command handling, we'll use direct HTTP for alerts)
    telegram_app = None
    if Config.TELEGRAM_TOKEN and Config.TELEGRAM_CHAT_ID:
        telegram_app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        telegram_app.add_handler(CommandHandler("start", start))
        telegram_app.add_handler(CommandHandler("help", help_cmd))
        telegram_app.add_handler(CommandHandler("status", status_cmd))
        telegram_app.add_handler(CommandHandler("performance", performance))
        telegram_app.add_handler(CommandHandler("scan", scan))
        telegram_app.add_handler(CommandHandler("pause", pause))
        telegram_app.add_handler(CommandHandler("resume", resume))
        telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_button))
        tg_thread = threading.Thread(target=lambda: telegram_app.run_polling())
        tg_thread.daemon = True
        tg_thread.start()
        logger.info("Telegram bot started with keyboard.")
    else:
        logger.warning("Telegram token or chat ID not set. Bot will run without alerts.")

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

    # Trader
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