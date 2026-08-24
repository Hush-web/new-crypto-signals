#!/usr/bin/env python3
"""
Multi-Asset Consensus Trading Bot – Production with Enhanced Logging
"""

import os
import time
import csv
import sqlite3
import logging
import threading
import asyncio
import requests
import re
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Set, Dict, Any
from dotenv import load_dotenv
from flask import Flask, jsonify, send_file
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.error import Conflict

load_dotenv()

# ---------------------------- CONFIGURATION ----------------------------
class Config:
    SYMBOLS = [s.strip().replace('/', '-') for s in os.getenv("SYMBOLS", "BTC-USDT,ETH-USDT,SOL-USDT,AVAX-USDT,POL-USDT").split(',') if s.strip()]
    INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "10000.0"))
    MAX_POSITIONS_GLOBAL = int(os.getenv("MAX_POSITIONS_GLOBAL", "5"))
    MAX_POSITIONS_PER_SYMBOL = int(os.getenv("MAX_POSITIONS_PER_SYMBOL", "1"))
    PER_TRADE_RISK_PCT = float(os.getenv("PER_TRADE_RISK_PCT", "0.02"))
    MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "0.10"))
    CONSENSUS_THRESHOLD = float(os.getenv("CONSENSUS_THRESHOLD", "0.40"))
    MIN_SOURCES = int(os.getenv("MIN_SOURCES", "1"))
    CONSECUTIVE_LOSS_LIMIT = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "3"))
    VOLATILITY_MIN = float(os.getenv("VOLATILITY_MIN", "0.005"))
    TREND_FILTER = os.getenv("TREND_FILTER", "true").lower() == "true"
    TREND_MA_PERIOD = int(os.getenv("TREND_MA_PERIOD", "200"))
    TRADE_INTERVAL_SECONDS = int(os.getenv("TRADE_INTERVAL_SECONDS", "60"))
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    DB_FILE = os.getenv("DB_FILE", "trades.db")
    CSV_FILE = os.getenv("CSV_FILE", "trades.csv")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    OPTIMIZE_ON_START = os.getenv("OPTIMIZE_ON_START", "true").lower() == "true"
    OPTIMIZE_TRAIN_DAYS = int(os.getenv("OPTIMIZE_TRAIN_DAYS", "60"))
    OPTIMIZE_TEST_DAYS = int(os.getenv("OPTIMIZE_TEST_DAYS", "30"))
    BACKTEST_MONTHS = int(os.getenv("BACKTEST_MONTHS", "12"))

    EXCHANGE_NAME = os.getenv("EXCHANGE_NAME", "")
    EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY", "")
    EXCHANGE_SECRET = os.getenv("EXCHANGE_SECRET", "")
    LIVE_TRADING = bool(EXCHANGE_NAME and EXCHANGE_API_KEY and EXCHANGE_SECRET)

    SOURCE_WEIGHTS = {
        "ma": float(os.getenv("WEIGHT_MA", "0.6")),
        "volume_momentum": float(os.getenv("WEIGHT_VOLUME_MOMENTUM", "0.5")),
        "breakout": float(os.getenv("WEIGHT_BREAKOUT", "0.7")),
        "whale": float(os.getenv("WEIGHT_WHALE", "0.6")),
    }

# ---------------------------- LOGGING ----------------------------
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, Config.LOG_LEVEL)
)
logger = logging.getLogger("multi-trader")

# ---------------------------- HTML SANITIZER ----------------------------
def sanitize_html(text: str) -> str:
    allowed_tags = r'<\/?(b|strong|i|em|u|ins|s|strike|del|a|code|pre)(?:\s[^>]*)?>'
    placeholders = {}
    def replacer(match):
        tag = match.group(0)
        placeholder = f"__TAG_{len(placeholders)}__"
        placeholders[placeholder] = tag
        return placeholder
    temp = re.sub(allowed_tags, replacer, text, flags=re.IGNORECASE)
    temp = temp.replace('<', '&lt;').replace('>', '&gt;')
    for placeholder, tag in placeholders.items():
        temp = temp.replace(placeholder, tag)
    return temp

# ---------------------------- KUCOIN SYMBOL VALIDATOR ----------------------------
def get_kucoin_symbols() -> Set[str]:
    try:
        resp = requests.get("https://api.kucoin.com/api/v2/symbols", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get('code') == '200000':
                symbols = [s['symbol'] for s in data.get('data', []) if s.get('enableTrading') is True]
                return set(symbols)
    except Exception as e:
        logger.error(f"Failed to fetch KuCoin symbols: {e}")
    return set()

# ---------------------------- DATABASE ----------------------------
class TradeDB:
    def __init__(self, db_file=Config.DB_FILE):
        self.conn = sqlite3.connect(db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_tables()
        self.lock = threading.Lock()

    def _init_tables(self):
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
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER UNIQUE,
                symbols TEXT,
                consensus_threshold REAL,
                min_sources INTEGER,
                volatility_min REAL,
                per_trade_risk_pct REAL,
                max_daily_loss_pct REAL,
                max_drawdown REAL,
                max_positions_global INTEGER,
                trend_filter INTEGER,
                weight_ma REAL,
                weight_volume_momentum REAL,
                weight_breakout REAL,
                weight_whale REAL,
                updated_at INTEGER
            )
        ''')
        self.conn.commit()

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

# ---------------------------- SETTINGS MANAGER ----------------------------
class SettingsManager:
    def __init__(self, db: TradeDB):
        self.db = db
        self.lock = threading.Lock()

    def get(self, chat_id: int) -> Optional[Dict[str, Any]]:
        with self.lock:
            self.db.cursor.execute(
                "SELECT * FROM user_settings WHERE chat_id = ?", (chat_id,)
            )
            row = self.db.cursor.fetchone()
            if row:
                return {
                    'symbols': row[2],
                    'consensus_threshold': row[3],
                    'min_sources': row[4],
                    'volatility_min': row[5],
                    'per_trade_risk_pct': row[6],
                    'max_daily_loss_pct': row[7],
                    'max_drawdown': row[8],
                    'max_positions_global': row[9],
                    'trend_filter': bool(row[10]),
                    'weight_ma': row[11],
                    'weight_volume_momentum': row[12],
                    'weight_breakout': row[13],
                    'weight_whale': row[14],
                }
            return None

    def set(self, chat_id: int, key: str, value: Any):
        with self.lock:
            self.db.cursor.execute("SELECT chat_id FROM user_settings WHERE chat_id = ?", (chat_id,))
            if self.db.cursor.fetchone():
                self.db.cursor.execute(
                    f"UPDATE user_settings SET {key} = ?, updated_at = ? WHERE chat_id = ?",
                    (value, int(time.time()), chat_id)
                )
            else:
                defaults = {
                    'symbols': ','.join(Config.SYMBOLS),
                    'consensus_threshold': Config.CONSENSUS_THRESHOLD,
                    'min_sources': Config.MIN_SOURCES,
                    'volatility_min': Config.VOLATILITY_MIN,
                    'per_trade_risk_pct': Config.PER_TRADE_RISK_PCT,
                    'max_daily_loss_pct': Config.MAX_DAILY_LOSS_PCT,
                    'max_drawdown': Config.MAX_DRAWDOWN,
                    'max_positions_global': Config.MAX_POSITIONS_GLOBAL,
                    'trend_filter': 1 if Config.TREND_FILTER else 0,
                    'weight_ma': Config.SOURCE_WEIGHTS['ma'],
                    'weight_volume_momentum': Config.SOURCE_WEIGHTS['volume_momentum'],
                    'weight_breakout': Config.SOURCE_WEIGHTS['breakout'],
                    'weight_whale': Config.SOURCE_WEIGHTS['whale'],
                }
                defaults[key] = value
                self.db.cursor.execute('''
                    INSERT INTO user_settings (
                        chat_id, symbols, consensus_threshold, min_sources,
                        volatility_min, per_trade_risk_pct, max_daily_loss_pct,
                        max_drawdown, max_positions_global, trend_filter,
                        weight_ma, weight_volume_momentum, weight_breakout, weight_whale,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    chat_id,
                    defaults['symbols'],
                    defaults['consensus_threshold'],
                    defaults['min_sources'],
                    defaults['volatility_min'],
                    defaults['per_trade_risk_pct'],
                    defaults['max_daily_loss_pct'],
                    defaults['max_drawdown'],
                    defaults['max_positions_global'],
                    defaults['trend_filter'],
                    defaults['weight_ma'],
                    defaults['weight_volume_momentum'],
                    defaults['weight_breakout'],
                    defaults['weight_whale'],
                    int(time.time())
                ))
            self.db.conn.commit()

    def reset(self, chat_id: int):
        with self.lock:
            self.db.cursor.execute("DELETE FROM user_settings WHERE chat_id = ?", (chat_id,))
            self.db.conn.commit()

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
            max_drawdown = 0
            peak = 0
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
                balance = float(r['balance_after'])
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak if peak > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd
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
                'max_drawdown': max_drawdown,
                'current_balance': last_balance,
                'profit_factor': abs(pnl_sum / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0 else float('inf')
            }
        except Exception as e:
            logger.error(f"CSV read error: {e}")
            return None

# ---------------------------- MARKET DATA (KuCoin) ----------------------------
class MarketData:
    def __init__(self, symbol):
        self.symbol = symbol
        self.base, self.quote = symbol.split('-')
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _fetch_kucoin(self, endpoint, params=None):
        url = f"https://api.kucoin.com{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.error(f"KuCoin {endpoint} {self.symbol} status {resp.status_code}: {resp.text[:100]}")
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
        params = {"symbol": self.symbol, "type": timeframe, "limit": limit}
        data = self._fetch_kucoin("/api/v1/market/candles", params)
        if not data:
            return None
        data = data[::-1]
        ohlcv = []
        for candle in data:
            try:
                ohlcv.append([
                    float(candle[1]),  # open
                    float(candle[2]),  # close
                    float(candle[3]),  # high
                    float(candle[4]),  # low
                    float(candle[5])   # volume
                ])
            except:
                continue
        if len(ohlcv) < 20:
            return None
        return np.array(ohlcv)

    def get_ohlcv_multi_month(self, months=12, timeframe='1hour'):
        if months <= 0:
            months = 1
        now = int(time.time())
        end = now
        start = now - (months * 30 * 24 * 3600)
        all_candles = []
        limit = 1500

        while start < end:
            params = {
                "symbol": self.symbol,
                "type": timeframe,
                "limit": limit,
                "from": start,
                "to": end
            }
            data = self._fetch_kucoin("/api/v1/market/candles", params)
            if not data:
                break
            all_candles.extend(data)
            if len(data) > 0:
                oldest_ts = int(data[0][0])
                end = oldest_ts - 1
            else:
                break
            time.sleep(0.2)

        if not all_candles:
            return None

        all_candles = all_candles[::-1]
        ohlcv = []
        for candle in all_candles:
            try:
                ohlcv.append([
                    float(candle[1]),  # open
                    float(candle[2]),  # close
                    float(candle[3]),  # high
                    float(candle[4]),  # low
                    float(candle[5])   # volume
                ])
            except:
                continue
        if len(ohlcv) < 100:
            return None
        return np.array(ohlcv)

    def get_recent_trades(self, limit=100):
        params = {"symbol": self.symbol, "limit": limit}
        data = self._fetch_kucoin("/api/v1/market/histories", params)
        if not data:
            return None
        trades = []
        for t in data:
            trades.append({
                'price': float(t['price']),
                'size': float(t['size']),
                'side': t['side'],
                'time': t['time']
            })
        return trades

    def get_24h_stats(self):
        params = {"symbol": self.symbol}
        data = self._fetch_kucoin("/api/v1/market/stats", params)
        if not data:
            return None
        return {
            'change': float(data.get('changeRate', 0)) * 100,
            'volume': float(data.get('vol', 0)),
            'high': float(data.get('high', 0)),
            'low': float(data.get('low', 0)),
        }

# ---------------------------- SIGNAL SOURCES ----------------------------
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

    def fetch(self, timeframe='1hour') -> Optional[Signal]:
        raise NotImplementedError

class MASource(SignalSource):
    def fetch(self, timeframe='1hour'):
        ohlcv = self.market.get_ohlcv(limit=100, timeframe=timeframe)
        if ohlcv is None or len(ohlcv) < 50:
            return None
        close = ohlcv[:, 1]
        ma20 = np.mean(close[-20:])
        ma50 = np.mean(close[-50:])
        if ma20 > ma50:
            return Signal(+1, 0.60, "ma")
        elif ma20 < ma50:
            return Signal(-1, 0.60, "ma")
        return Signal(0, 0.0, "ma")

class VolumeMomentumSource(SignalSource):
    def fetch(self, timeframe='1hour'):
        ohlcv = self.market.get_ohlcv(limit=50, timeframe=timeframe)
        if ohlcv is None or len(ohlcv) < 11:
            return None
        close = ohlcv[:, 1]
        volume = ohlcv[:, 4]
        avg_vol = np.mean(volume[-10:-1])
        avg_change = np.mean(np.diff(close[-11:])) / np.mean(close[-11:-1]) if np.mean(close[-11:-1]) != 0 else 0
        curr_vol = volume[-1]
        curr_change = (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0
        if curr_vol > 1.5 * avg_vol and curr_change > 0.005:
            return Signal(+1, 0.55, "volume_momentum")
        elif curr_vol > 1.5 * avg_vol and curr_change < -0.005:
            return Signal(-1, 0.55, "volume_momentum")
        return Signal(0, 0.0, "volume_momentum")

class BreakoutSource(SignalSource):
    def fetch(self, timeframe='1hour'):
        ohlcv = self.market.get_ohlcv(limit=50, timeframe=timeframe)
        if ohlcv is None or len(ohlcv) < 20:
            return None
        high = ohlcv[:, 2]
        low = ohlcv[:, 3]
        close = ohlcv[:, 1]
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        price = close[-1]
        if price > recent_high * 1.002:
            return Signal(+1, 0.65, "breakout")
        elif price < recent_low * 0.998:
            return Signal(-1, 0.65, "breakout")
        return Signal(0, 0.0, "breakout")

class WhaleSource(SignalSource):
    def __init__(self, market_data, db, threshold_usd=50000):
        super().__init__(market_data, db)
        self.threshold_usd = threshold_usd

    def fetch(self, timeframe='1hour'):
        trades = self.market.get_recent_trades(limit=200)
        if trades is None:
            return None
        ohlcv = self.market.get_ohlcv(limit=5, timeframe='1hour')
        if ohlcv is None:
            return None
        price = ohlcv[-1][1]
        buy_volume = 0
        sell_volume = 0
        for t in trades:
            trade_usd = t['price'] * t['size']
            if trade_usd > self.threshold_usd:
                if t['side'] == 'buy':
                    buy_volume += t['size']
                else:
                    sell_volume += t['size']
        if buy_volume > sell_volume * 1.5 and buy_volume > 0:
            return Signal(+1, 0.60, "whale")
        elif sell_volume > buy_volume * 1.5 and sell_volume > 0:
            return Signal(-1, 0.60, "whale")
        return Signal(0, 0.0, "whale")

# ---------------------------- CONSENSUS ENGINE ----------------------------
class ConsensusEngine:
    def __init__(self, threshold=Config.CONSENSUS_THRESHOLD, weights=Config.SOURCE_WEIGHTS, min_sources=Config.MIN_SOURCES):
        self.threshold = threshold
        self.weights = weights
        self.min_sources = min_sources

    def aggregate(self, signals: List[Signal]) -> Tuple[int, float, List[str]]:
        active = [sig for sig in signals if sig.direction != 0]
        if len(active) < self.min_sources:
            return 0, 0.0, []
        sum_dir = 0.0
        total_w = 0.0
        for sig in active:
            w = self.weights.get(sig.source, 1.0)
            sum_dir += sig.direction * w
            total_w += w
        if total_w == 0:
            return 0, 0.0, []
        avg_dir = sum_dir / total_w
        conf_num = 0.0
        conf_den = 0.0
        for sig in active:
            w = self.weights.get(sig.source, 1.0)
            conf_num += sig.confidence * w
            conf_den += w
        avg_conf = conf_num / conf_den if conf_den > 0 else 0.0
        direction = 0
        if avg_dir > self.threshold:
            direction = 1
        elif avg_dir < -self.threshold:
            direction = -1
        details = [f"{sig.source}:{sig.direction} ({sig.confidence:.2f})" for sig in active]
        return direction, avg_conf, details

# ---------------------------- RISK MANAGER ----------------------------
class RiskManager:
    def __init__(self, initial_balance, config):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.config = config
        self.open_positions = {}
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.lock = threading.Lock()

    def update_balance(self, new_balance):
        with self.lock:
            self.balance = new_balance
            if new_balance > self.peak_balance:
                self.peak_balance = new_balance

    def get_drawdown_pct(self):
        with self.lock:
            if self.peak_balance == 0:
                return 0
            return (self.peak_balance - self.balance) / self.peak_balance

    def get_total_positions(self):
        total = 0
        for positions in self.open_positions.values():
            total += len(positions)
        return total

    def can_trade(self, symbol, price, atr, trend_ok, settings):
        max_daily_loss_pct = settings.get('max_daily_loss_pct', Config.MAX_DAILY_LOSS_PCT)
        max_positions_global = settings.get('max_positions_global', Config.MAX_POSITIONS_GLOBAL)
        max_drawdown = settings.get('max_drawdown', Config.MAX_DRAWDOWN)
        volatility_min = settings.get('volatility_min', Config.VOLATILITY_MIN)
        with self.lock:
            if self.daily_pnl < -max_daily_loss_pct * self.initial_balance:
                return False, "Daily loss limit"
            if self.get_total_positions() >= max_positions_global:
                return False, "Global max positions"
            if self.open_positions.get(symbol, []) and len(self.open_positions[symbol]) >= Config.MAX_POSITIONS_PER_SYMBOL:
                return False, "Max per symbol"
            if self.consecutive_losses >= Config.CONSECUTIVE_LOSS_LIMIT:
                return False, "Consecutive loss limit"
            if self.get_drawdown_pct() > max_drawdown:
                return False, f"Max drawdown ({max_drawdown*100:.0f}%) reached"
            if atr is not None and atr > 0:
                volatility = atr / price
                if volatility < volatility_min:
                    return False, f"Volatility too low ({volatility*100:.2f}% < {volatility_min*100:.2f}%)"
                if volatility > 0.10:
                    return False, f"Volatility too high ({volatility*100:.2f}%)"
            if Config.TREND_FILTER and not trend_ok:
                return False, "Trend filter rejected"
            return True, "OK"

    def compute_position_size(self, balance, price, atr, settings):
        per_trade_risk_pct = settings.get('per_trade_risk_pct', Config.PER_TRADE_RISK_PCT)
        risk_amount = balance * per_trade_risk_pct
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

# ---------------------------- MULTI-ASSET TRADER (with enhanced logging) ----------------------------
class MultiTrader:
    def __init__(self, symbols, initial_balance, risk_mgr, db, telegram_token, chat_id, live_broker):
        self.db = db
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.live_broker = live_broker
        self.risk_mgr = risk_mgr
        self.balance = initial_balance
        self.balance_lock = threading.Lock()
        self.settings_manager = SettingsManager(db)

        self.override_settings = None
        if chat_id:
            self.override_settings = self.settings_manager.get(int(chat_id))
        self._init_symbols(symbols)
        self._init_market_data()
        self.running = True
        self.performance_logger = PerformanceLogger(Config.CSV_FILE)
        self.last_prices = {}
        self.heartbeat_counter = 0

        self.optimal_thresholds = {}

        if Config.OPTIMIZE_ON_START:
            logger.info("Starting optimization in background thread...")
            threading.Thread(target=self.run_optimization, daemon=True).start()
        else:
            logger.info("Optimization disabled (OPTIMIZE_ON_START=false)")

    def _init_symbols(self, default_symbols):
        if self.override_settings and 'symbols' in self.override_settings:
            self.symbols = [s.strip() for s in self.override_settings['symbols'].split(',') if s.strip()]
        else:
            self.symbols = default_symbols
        valid = get_kucoin_symbols()
        if valid:
            self.symbols = [s for s in self.symbols if s in valid]
            if not self.symbols:
                self.symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "AVAX-USDT", "POL-USDT"]
                logger.warning("No valid symbols, falling back to defaults")
        logger.info(f"Active symbols: {self.symbols}")

    def _init_market_data(self):
        self.markets = {sym: MarketData(sym) for sym in self.symbols}
        self.sources = {}
        for sym in self.symbols:
            self.sources[sym] = [
                MASource(self.markets[sym], self.db),
                VolumeMomentumSource(self.markets[sym], self.db),
                BreakoutSource(self.markets[sym], self.db),
                WhaleSource(self.markets[sym], self.db),
            ]

    def reload_settings(self):
        if self.chat_id:
            self.override_settings = self.settings_manager.get(int(self.chat_id))
            self._init_symbols(Config.SYMBOLS)
            self._init_market_data()
            logger.info("Settings reloaded and market data updated")

    # ------------------------ WALK-FORWARD OPTIMIZATION (background) ------------------------
    def run_optimization(self):
        for sym in self.symbols:
            try:
                logger.info(f"Running optimization for {sym}...")
                best_th = self.optimize_threshold(sym,
                                                  train_days=Config.OPTIMIZE_TRAIN_DAYS,
                                                  test_days=Config.OPTIMIZE_TEST_DAYS)
                if best_th is not None:
                    self.optimal_thresholds[sym] = best_th
                    logger.info(f"Optimal threshold for {sym}: {best_th:.2f}")
            except Exception as e:
                logger.error(f"Optimization failed for {sym}: {e}")
        logger.info("Optimization complete for all symbols.")

    def optimize_threshold(self, symbol, train_days=60, test_days=30):
        total_days = train_days + test_days
        ohlcv = self.markets[symbol].get_ohlcv_multi_month(months=6, timeframe='1hour')
        if ohlcv is None or len(ohlcv) < total_days * 24 + 100:
            logger.warning(f"Insufficient data for {symbol} optimization.")
            return None
        train_limit = train_days * 24
        test_limit = test_days * 24
        if len(ohlcv) < train_limit + test_limit:
            return None
        train_data = ohlcv[:train_limit]
        test_data = ohlcv[train_limit:train_limit+test_limit]

        thresholds = np.arange(0.2, 0.85, 0.05)
        best_th = 0.4
        best_score = -999
        for th in thresholds:
            train_result = self._run_backtest_on_data(symbol, train_data, threshold=th, months=0)
            if train_result is None or isinstance(train_result, str):
                continue
            test_result = self._run_backtest_on_data(symbol, test_data, threshold=th, months=0)
            if test_result is None or isinstance(test_result, str):
                continue
            score = test_result.get('profit_factor', 0) * test_result.get('win_rate', 0) / 100
            if score > best_score:
                best_score = score
                best_th = th
        return best_th

    def _run_backtest_on_data(self, symbol, ohlcv, threshold, months=0):
        if len(ohlcv) < 50:
            return None
        balance = Config.INITIAL_BALANCE
        pnl_list = []
        wins = 0
        losses = 0
        fee = 0.001
        slippage = 0.0005
        weights = Config.SOURCE_WEIGHTS.copy()
        engine = ConsensusEngine(threshold=threshold, weights=weights, min_sources=Config.MIN_SOURCES)

        for i in range(50, len(ohlcv)-1):
            slice_data = ohlcv[:i+1]
            close = slice_data[:, 1]
            ma20 = np.mean(close[-20:])
            ma50 = np.mean(close[-50:])
            ma_signal = 1 if ma20 > ma50 else (-1 if ma20 < ma50 else 0)

            volume = slice_data[:, 4]
            if len(close) >= 11:
                avg_vol = np.mean(volume[-10:-1])
                avg_change = np.mean(np.diff(close[-11:])) / np.mean(close[-11:-1]) if np.mean(close[-11:-1]) != 0 else 0
                curr_vol = volume[-1]
                curr_change = (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0
                vm_signal = 1 if curr_vol > 1.5 * avg_vol and curr_change > 0.005 else (-1 if curr_vol > 1.5 * avg_vol and curr_change < -0.005 else 0)
            else:
                vm_signal = 0

            high = slice_data[:, 2]
            low = slice_data[:, 3]
            price = close[-1]
            recent_high = np.max(high[-20:])
            recent_low = np.min(low[-20:])
            breakout_signal = 1 if price > recent_high * 1.002 else (-1 if price < recent_low * 0.998 else 0)

            signals = []
            if ma_signal != 0:
                signals.append(Signal(ma_signal, 0.60, "ma"))
            if vm_signal != 0:
                signals.append(Signal(vm_signal, 0.55, "volume_momentum"))
            if breakout_signal != 0:
                signals.append(Signal(breakout_signal, 0.65, "breakout"))
            if not signals:
                continue

            direction, conf, details = engine.aggregate(signals)
            if direction == 0 or conf < threshold:
                continue

            entry_price = price
            if direction == 1:
                entry_price = entry_price * (1 + slippage)
            else:
                entry_price = entry_price * (1 - slippage)

            high_curr = slice_data[:, 2]
            low_curr = slice_data[:, 3]
            prev_close = np.roll(close, 1)[1:]
            tr1 = high_curr[1:] - low_curr[1:]
            tr2 = np.abs(high_curr[1:] - prev_close)
            tr3 = np.abs(low_curr[1:] - prev_close)
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = np.mean(tr[-14:]) if len(tr) >= 14 else 0.02 * price
            risk = atr * 2.5
            stop_loss = entry_price - risk if direction == 1 else entry_price + risk
            take_profit = entry_price + risk * 1.5 if direction == 1 else entry_price - risk * 1.5

            next_high = ohlcv[i+1][2]
            next_low = ohlcv[i+1][3]
            next_close = ohlcv[i+1][1]
            exit_price = next_close
            status = "Close"
            if direction == 1:
                if next_low <= stop_loss:
                    exit_price = stop_loss
                    status = "SL"
                elif next_high >= take_profit:
                    exit_price = take_profit
                    status = "TP"
            else:
                if next_high >= stop_loss:
                    exit_price = stop_loss
                    status = "SL"
                elif next_low <= take_profit:
                    exit_price = take_profit
                    status = "TP"

            if direction == 1:
                exit_price = exit_price * (1 - slippage)
            else:
                exit_price = exit_price * (1 + slippage)

            size = (balance * Config.PER_TRADE_RISK_PCT) / (atr * 2.5)
            if size <= 0:
                continue

            if direction == 1:
                gross_pnl = (exit_price - entry_price) * size
            else:
                gross_pnl = (entry_price - exit_price) * size
            fee_amount = (abs(entry_price * size) + abs(exit_price * size)) * fee
            net_pnl = gross_pnl - fee_amount
            balance += net_pnl
            pnl_list.append(net_pnl)
            if net_pnl > 0:
                wins += 1
            else:
                losses += 1
            if len(pnl_list) > 100:
                break

        total_trades = wins + losses
        if total_trades == 0:
            return None
        win_rate = wins / total_trades * 100
        total_pnl = sum(pnl_list)
        profit_factor = abs(total_pnl / sum([p for p in pnl_list if p < 0])) if any(p < 0 for p in pnl_list) else float('inf')
        if len(pnl_list) > 1:
            mean_pnl = np.mean(pnl_list)
            std_pnl = np.std(pnl_list) if np.std(pnl_list) > 0 else 0.0001
            sharpe = (mean_pnl / std_pnl) * np.sqrt(365*24)
        else:
            sharpe = 0
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'pnl_list': pnl_list,
            'final_balance': balance
        }

    # ------------------------ DYNAMIC WEIGHTS ------------------------
    def get_dynamic_weights(self, symbol, price, atr):
        settings = self.override_settings or {}
        base_weights = {
            'ma': settings.get('weight_ma', Config.SOURCE_WEIGHTS['ma']),
            'volume_momentum': settings.get('weight_volume_momentum', Config.SOURCE_WEIGHTS['volume_momentum']),
            'breakout': settings.get('weight_breakout', Config.SOURCE_WEIGHTS['breakout']),
            'whale': settings.get('weight_whale', Config.SOURCE_WEIGHTS['whale']),
        }
        if atr is None or atr == 0 or price == 0:
            return base_weights
        vol = atr / price
        if vol > 0.03:
            base_weights['breakout'] = min(1.0, base_weights['breakout'] * 1.3)
            base_weights['ma'] = max(0.2, base_weights['ma'] * 0.7)
        elif vol < 0.01:
            base_weights['ma'] = min(1.0, base_weights['ma'] * 1.3)
            base_weights['breakout'] = max(0.2, base_weights['breakout'] * 0.7)
        total = sum(base_weights.values())
        if total > 0:
            for k in base_weights:
                base_weights[k] /= total
        return base_weights

    # ------------------------ MULTI-TIMEFRAME CONSENSUS ------------------------
    def get_multi_tf_signal(self, symbol, price, atr, trend_ok):
        timeframes = ['15min', '1hour', '4hour']
        tf_directions = []
        tf_details = []
        weights = self.get_dynamic_weights(symbol, price, atr)
        settings = self.override_settings or {}
        threshold = settings.get('consensus_threshold', Config.CONSENSUS_THRESHOLD)
        if symbol in self.optimal_thresholds:
            threshold = self.optimal_thresholds[symbol]

        for tf in timeframes:
            signals = []
            for src in self.sources[symbol]:
                sig = src.fetch(timeframe=tf)
                if sig and sig.direction != 0:
                    signals.append(sig)
                    self.db.log_signal(int(time.time()), symbol, sig.source, sig.direction, sig.confidence)
            if signals:
                engine = ConsensusEngine(threshold=threshold, weights=weights, min_sources=Config.MIN_SOURCES)
                direction, conf, details = engine.aggregate(signals)
                if direction != 0 and conf >= threshold:
                    tf_directions.append(direction)
                    tf_details.append(f"{tf}:{direction}")
        if len(tf_directions) >= 2:
            buy_votes = sum(1 for d in tf_directions if d == 1)
            sell_votes = sum(1 for d in tf_directions if d == -1)
            if buy_votes > sell_votes:
                return 1, tf_details
            elif sell_votes > buy_votes:
                return -1, tf_details
        return 0, tf_details

    # ------------------------ SEND ALERT ------------------------
    def send_alert(self, message):
        if not self.telegram_token or not self.chat_id:
            return
        safe_msg = sanitize_html(message)
        try:
            requests.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": safe_msg, "parse_mode": "HTML"},
                timeout=30
            )
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            try:
                requests.post(
                    f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                    json={"chat_id": self.chat_id, "text": message[:500] + "..." if len(message)>500 else message},
                    timeout=30
                )
            except:
                pass

    # ------------------------ PRICE AND ATR ------------------------
    def get_price_and_atr(self, symbol, timeframe='1hour'):
        ohlcv = self.markets[symbol].get_ohlcv(limit=100, timeframe=timeframe)
        if ohlcv is None or len(ohlcv) < 50:
            return None, None, None, None
        close = ohlcv[:, 1]
        high = ohlcv[:, 2]
        low = ohlcv[:, 3]
        high_curr = high[1:]
        low_curr = low[1:]
        prev_close = close[:-1]
        tr1 = high_curr - low_curr
        tr2 = np.abs(high_curr - prev_close)
        tr3 = np.abs(low_curr - prev_close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
        price = close[-1]
        if len(close) >= Config.TREND_MA_PERIOD:
            trend_ma = np.mean(close[-Config.TREND_MA_PERIOD:])
            trend_ok = price > trend_ma
        else:
            trend_ma = None
            trend_ok = True
        return price, atr, trend_ok, trend_ma

    # ------------------------ EXECUTE SIGNAL (with enhanced logging) ------------------------
    def execute_signal(self, symbol, direction, price, atr, tf_details, trend_ok):
        logger.info(f"execute_signal called for {symbol}, direction={direction}, price={price:.2f}")
        settings = self.override_settings or {}
        can_trade, reason = self.risk_mgr.can_trade(symbol, price, atr, trend_ok, settings)
        if not can_trade:
            logger.info(f"Trade blocked for {symbol}: {reason}")
            return

        size = self.risk_mgr.compute_position_size(self.balance, price, atr, settings)
        if size <= 0:
            logger.info(f"Position size zero for {symbol}")
            return
        risk = atr * 2.5
        stop_loss = price - risk if direction == 1 else price + risk
        take_profit = price + risk * 1.5 if direction == 1 else price - risk * 1.5
        side = 'buy' if direction == 1 else 'sell'
        cost = price * size

        with self.balance_lock:
            if side == 'buy':
                if self.balance < cost:
                    self.send_alert(f"⚠️ Insufficient balance for {symbol}")
                    return
                self.balance -= cost
            # For paper shorts: no balance change on entry

        self.risk_mgr.open_position(symbol, side, price, size, stop_loss, take_profit)
        self.db.log_trade(int(time.time()), symbol, side, price, size, 0.0, 0.0, self.balance)
        logger.info(f"Trade opened for {symbol}: {side} {size:.4f} @ {price:.2f}")

        details_html = "<br>".join([sanitize_html(f"• {d}") for d in tf_details])
        msg = (
            f"🔔 <b>{symbol} SIGNAL</b> ({'LIVE' if self.live_broker.enabled else 'PAPER'})\n"
            f"Action: {'🟢 BUY' if direction==1 else '🔴 SELL'}\n"
            f"Entry: ${price:.4f}\n"
            f"TP: ${take_profit:.4f} (+{(risk*1.5/price)*100:.2f}%)\n"
            f"SL: ${stop_loss:.4f} (-{(risk/price)*100:.2f}%)\n"
            f"Risk: {settings.get('per_trade_risk_pct', Config.PER_TRADE_RISK_PCT)*100:.1f}%\n"
            f"Timeframes: {details_html}\n\n"
            f"<i>Not financial advice.</i>"
        )
        self.send_alert(msg)

    # ------------------------ STEP (main loop with enhanced logging) ------------------------
    def step(self):
        for symbol in self.symbols:
            result = self.get_price_and_atr(symbol, timeframe='1hour')
            if result is None or result[0] is None:
                logger.warning(f"Could not get price/ATR for {symbol}, skipping")
                continue
            price, atr, trend_ok, trend_ma = result

            # Check existing positions
            pnl, status, pos = self.risk_mgr.check_sl_tp(symbol, price)
            if pnl != 0 and pos is not None:
                with self.balance_lock:
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

            # Get consensus
            direction, tf_details = self.get_multi_tf_signal(symbol, price, atr, trend_ok)
            if direction != 0:
                logger.info(f"Consensus for {symbol}: direction={direction}, details={tf_details}")
                self.execute_signal(symbol, direction, price, atr, tf_details, trend_ok)
            else:
                # Log occasionally for debugging
                pass

    def run_loop(self):
        logger.info("Starting multi-asset trading loop. Symbols: %s", self.symbols)
        while self.running:
            try:
                self.step()
                self.heartbeat_counter += 1
                if self.heartbeat_counter % 5 == 0:
                    logger.info("Trading loop heartbeat – alive")
                time.sleep(Config.TRADE_INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(Config.TRADE_INTERVAL_SECONDS)

    # ---------------------------- BACKTEST ----------------------------
    def backtest(self, symbol, lookback_days=30, timeframe='1hour'):
        if lookback_days > 60:
            months = lookback_days // 30
            ohlcv = self.markets[symbol].get_ohlcv_multi_month(months=months, timeframe=timeframe)
            if ohlcv is None:
                return "Insufficient data for backtest."
        else:
            ohlcv = self.markets[symbol].get_ohlcv(limit=lookback_days*24, timeframe=timeframe)
            if ohlcv is None or len(ohlcv) < 50:
                return "Insufficient data for backtest."
        threshold = (self.override_settings or {}).get('consensus_threshold', Config.CONSENSUS_THRESHOLD)
        result = self._run_backtest_on_data(symbol, ohlcv, threshold=threshold, months=0)
        if result is None:
            return "No trades generated in backtest period."
        return {
            'symbol': symbol,
            'period': f"{lookback_days} days",
            'total_trades': result['total_trades'],
            'win_rate': result['win_rate'],
            'total_pnl': result['total_pnl'],
            'final_balance': result['final_balance'],
            'max_drawdown': 0,
            'profit_factor': result['profit_factor'],
            'sharpe': result.get('sharpe', 0)
        }

# ---------------------------- FLASK APP ----------------------------
app = Flask(__name__)
trader_global = None

@app.route('/')
def health():
    return jsonify({"status": "running", "version": "final-production", "time": datetime.now().isoformat()})

@app.route('/download')
def download_csv():
    if os.path.exists(Config.CSV_FILE):
        return send_file(Config.CSV_FILE, as_attachment=True, download_name="trades.csv")
    return jsonify({"error": "No trades.csv yet"}), 404

@app.route('/status')
def status():
    if trader_global is None:
        return jsonify({"error": "trader not initialized"})
    with trader_global.balance_lock:
        balance = trader_global.balance
    daily_pnl = trader_global.db.get_daily_pnl()
    total_pos = trader_global.risk_mgr.get_total_positions()
    running = trader_global.running
    drawdown = trader_global.risk_mgr.get_drawdown_pct() * 100
    opt = trader_global.optimal_thresholds
    return jsonify({
        "balance": balance,
        "daily_pnl": daily_pnl,
        "open_positions": total_pos,
        "running": running,
        "drawdown": drawdown,
        "optimized_thresholds": opt
    })

@app.route('/test')
def test_telegram():
    if trader_global is None:
        return jsonify({"error": "trader not initialized"}), 500
    try:
        trader_global.send_alert("🧪 Test message from bot!")
        return jsonify({"status": "sent"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------- TELEGRAM BOT ----------------------------
def get_main_keyboard():
    buttons = [
        [KeyboardButton("📊 Status"), KeyboardButton("🔍 Scan")],
        [KeyboardButton("📈 Performance"), KeyboardButton("⏸️ Pause"), KeyboardButton("▶️ Resume")],
        [KeyboardButton("⚙️ Settings"), KeyboardButton("❓ Help")]
    ]
    return ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=False)

async def ping_cmd(update: Update, context):
    try:
        await update.message.reply_text("🏓 Pong! Bot is alive.", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in ping_cmd: {e}")

async def start(update: Update, context):
    try:
        await update.message.reply_text("⏳ Processing...", reply_markup=get_main_keyboard())
        logger.info(f"Received /start from {update.effective_user.id}")
        await update.message.reply_text(
            "🤖 <b>Consensus Trader (Final Production)</b>\n\n"
            "Commands:\n"
            "/status – Account\n/scan – Force scan\n/performance – Stats\n"
            "/backtest <symbol> – Run 12-month backtest\n"
            "/settings – View/Edit settings\n"
            "/set &lt;key&gt; &lt;value&gt; – Change a setting\n"
            "/reset – Reset to defaults\n"
            "/ping – Liveness check\n"
            "/restartloop – Restart trading loop\n"
            "/pause – Pause\n/resume – Resume\n/help – This\n\n"
            "💾 <a href='https://new-crypto-signals.onrender.com/download'>Download CSV</a>",
            parse_mode='HTML', reply_markup=get_main_keyboard(), disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Error in start: {e}")

async def help_cmd(update: Update, context):
    try:
        await update.message.reply_text("⏳ Processing...", reply_markup=get_main_keyboard())
        await update.message.reply_text(
            "📋 <b>Commands</b>\n"
            "/status – Account\n/scan – Force scan\n/performance – Stats\n"
            "/backtest <symbol> – Run backtest\n"
            "/settings – View/Edit settings\n"
            "/set &lt;key&gt; &lt;value&gt; – Change a setting\n"
            "/reset – Reset to defaults\n"
            "/ping – Liveness check\n"
            "/restartloop – Restart trading loop\n"
            "/pause – Pause\n/resume – Resume\n/help – This",
            parse_mode='HTML', reply_markup=get_main_keyboard()
        )
    except Exception as e:
        logger.error(f"Error in help: {e}")

async def status_cmd(update: Update, context):
    try:
        await update.message.reply_text("⏳ Fetching status...", reply_markup=get_main_keyboard())
        logger.info(f"Received /status from {update.effective_user.id}")

        if trader_global is None:
            await update.message.reply_text("Trader not ready.", reply_markup=get_main_keyboard())
            return

        result_container = {}
        def compute_status():
            try:
                t = trader_global
                drawdown = t.risk_mgr.get_drawdown_pct() * 100
                daily_pnl = t.db.get_daily_pnl()
                total_pos = t.risk_mgr.get_total_positions()
                with t.balance_lock:
                    balance = t.balance
                running = t.running
                opt = t.optimal_thresholds
                result_container['success'] = True
                result_container['drawdown'] = drawdown
                result_container['daily_pnl'] = daily_pnl
                result_container['total_pos'] = total_pos
                result_container['balance'] = balance
                result_container['running'] = running
                result_container['opt'] = opt
            except Exception as e:
                result_container['error'] = str(e)

        thread = threading.Thread(target=compute_status)
        thread.daemon = True
        thread.start()
        thread.join(timeout=5.0)

        if 'error' in result_container:
            logger.error(f"Status computation error: {result_container['error']}")
            await update.message.reply_text("⚠️ Error computing status. Try again.", reply_markup=get_main_keyboard())
            return

        if not result_container.get('success'):
            logger.warning("Status computation timed out or failed.")
            await update.message.reply_text("⚠️ Status temporarily unavailable. Please try again later.", reply_markup=get_main_keyboard())
            return

        drawdown = result_container['drawdown']
        daily_pnl = result_container['daily_pnl']
        total_pos = result_container['total_pos']
        balance = result_container['balance']
        running = result_container['running']
        opt = result_container['opt']
        opt_str = "\n".join([f"{k}: {v:.2f}" for k,v in opt.items()]) if opt else "Not optimized"
        msg = (
            f"📊 <b>Status</b>\nBalance: ${balance:.2f}\nDaily PnL: ${daily_pnl:.2f}\n"
            f"Open Positions: {total_pos}\n"
            f"Drawdown: {drawdown:.2f}%\nRunning: {'✅' if running else '⏸️'}\n"
            f"Optimized Thresholds:\n{opt_str}"
        )
        safe_msg = sanitize_html(msg)
        logger.info("Sending final status message...")
        await update.message.reply_text(safe_msg, parse_mode='HTML', reply_markup=get_main_keyboard())
        logger.info("Status sent successfully.")
    except Exception as e:
        logger.error(f"Error in status_cmd: {e}", exc_info=True)

async def performance(update: Update, context):
    try:
        await update.message.reply_text("⏳ Computing performance...", reply_markup=get_main_keyboard())
        logger.info(f"Received /performance from {update.effective_user.id}")
        if trader_global is None:
            await update.message.reply_text("Trader not ready.", reply_markup=get_main_keyboard())
            return
        summary = trader_global.performance_logger.get_summary()
        if not summary:
            await update.message.reply_text("No trades yet.", reply_markup=get_main_keyboard())
            return
        msg = (
            f"📈 <b>Performance</b>\n"
            f"Trades: {summary['total_trades']}\n"
            f"Win Rate: {summary['win_rate']:.1f}%\n"
            f"Total PnL: ${summary['total_pnl']:.2f}\n"
            f"Avg Win: ${summary['avg_win']:.2f}\n"
            f"Avg Loss: ${summary['avg_loss']:.2f}\n"
            f"Best: ${summary['best_trade']:.2f}\n"
            f"Worst: ${summary['worst_trade']:.2f}\n"
            f"Max Drawdown: {summary['max_drawdown']*100:.2f}%\n"
            f"Profit Factor: {summary['profit_factor']:.2f}\n"
            f"Balance: ${summary['current_balance']:.2f}"
        )
        safe_msg = sanitize_html(msg)
        await update.message.reply_text(safe_msg, parse_mode='HTML', reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in performance: {e}", exc_info=True)

async def backtest(update: Update, context):
    try:
        await update.message.reply_text("⏳ Running 12-month backtest...", reply_markup=get_main_keyboard())
        logger.info(f"Received /backtest from {update.effective_user.id}")
        if trader_global is None:
            await update.message.reply_text("Trader not ready.", reply_markup=get_main_keyboard())
            return
        args = context.args
        if not args:
            await update.message.reply_text("Usage: /backtest SYMBOL e.g., /backtest BTC-USDT", reply_markup=get_main_keyboard())
            return
        symbol = args[0].upper()
        if '-' not in symbol:
            symbol = symbol.replace('/', '-')
        if symbol not in trader_global.symbols:
            await update.message.reply_text(f"Symbol {symbol} not active. Active: {', '.join(trader_global.symbols)}", reply_markup=get_main_keyboard())
            return
        result = trader_global.backtest(symbol, lookback_days=Config.BACKTEST_MONTHS*30, timeframe='1hour')
        if isinstance(result, str):
            await update.message.reply_text(result, reply_markup=get_main_keyboard())
        else:
            msg = (
                f"📊 <b>12-Month Backtest Results</b>\n"
                f"Symbol: {result['symbol']}\n"
                f"Period: {result['period']}\n"
                f"Trades: {result['total_trades']}\n"
                f"Win Rate: {result['win_rate']:.1f}%\n"
                f"Total PnL: ${result['total_pnl']:.2f}\n"
                f"Final Balance: ${result['final_balance']:.2f}\n"
                f"Profit Factor: {result['profit_factor']:.2f}\n"
                f"Sharpe Ratio: {result.get('sharpe', 0):.2f}"
            )
            safe_msg = sanitize_html(msg)
            await update.message.reply_text(safe_msg, parse_mode='HTML', reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)

async def scan(update: Update, context):
    try:
        await update.message.reply_text("🔍 Scanning...", reply_markup=get_main_keyboard())
        logger.info(f"Received /scan from {update.effective_user.id}")
        if trader_global is None:
            return
        for sym in trader_global.symbols:
            price, atr, trend_ok, _ = trader_global.get_price_and_atr(sym, timeframe='1hour')
            if price is None:
                await update.message.reply_text(f"⚠️ No data for {sym}")
                continue
            direction, tf_details = trader_global.get_multi_tf_signal(sym, price, atr, trend_ok)
            msg = f"⚖️ {sym}: {'BUY' if direction==1 else 'SELL' if direction==-1 else 'NEUTRAL'}\n"
            msg += f"Timeframes: {' | '.join(tf_details) if tf_details else 'No consensus'}\n"
            msg += f"Price: ${price:.2f}\n"
            vol = (atr/price)*100 if atr else 0
            msg += f"Volatility: {vol:.2f}%"
            await update.message.reply_text(msg, reply_markup=get_main_keyboard())
        await update.message.reply_text("✅ Scan complete.", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in scan: {e}", exc_info=True)

async def pause(update: Update, context):
    try:
        await update.message.reply_text("⏳ Pausing...", reply_markup=get_main_keyboard())
        if trader_global:
            trader_global.running = False
        await update.message.reply_text("⏸️ Paused.", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in pause: {e}")

async def resume(update: Update, context):
    try:
        await update.message.reply_text("⏳ Resuming...", reply_markup=get_main_keyboard())
        if trader_global:
            trader_global.running = True
        await update.message.reply_text("▶️ Resumed.", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in resume: {e}")

async def restartloop_cmd(update: Update, context):
    try:
        await update.message.reply_text("⏳ Restarting trading loop...", reply_markup=get_main_keyboard())
        global trader_global
        if trader_global is None:
            await update.message.reply_text("Trader not initialized.")
            return
        trader_global.running = False
        time.sleep(1)
        trader_global.running = True
        threading.Thread(target=trader_global.run_loop, daemon=True).start()
        await update.message.reply_text("✅ Trading loop restarted.", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in restartloop: {e}")
        await update.message.reply_text(f"❌ Failed to restart loop: {e}", reply_markup=get_main_keyboard())

async def settings_cmd(update: Update, context):
    try:
        await update.message.reply_text("⏳ Loading settings...", reply_markup=get_main_keyboard())
        chat_id = update.effective_user.id
        if trader_global is None:
            await update.message.reply_text("Trader not ready.", reply_markup=get_main_keyboard())
            return
        settings = trader_global.settings_manager.get(chat_id)
        if not settings:
            settings = {
                'symbols': ','.join(Config.SYMBOLS),
                'consensus_threshold': Config.CONSENSUS_THRESHOLD,
                'min_sources': Config.MIN_SOURCES,
                'volatility_min': Config.VOLATILITY_MIN,
                'per_trade_risk_pct': Config.PER_TRADE_RISK_PCT,
                'max_daily_loss_pct': Config.MAX_DAILY_LOSS_PCT,
                'max_drawdown': Config.MAX_DRAWDOWN,
                'max_positions_global': Config.MAX_POSITIONS_GLOBAL,
                'trend_filter': Config.TREND_FILTER,
                'weight_ma': Config.SOURCE_WEIGHTS['ma'],
                'weight_volume_momentum': Config.SOURCE_WEIGHTS['volume_momentum'],
                'weight_breakout': Config.SOURCE_WEIGHTS['breakout'],
                'weight_whale': Config.SOURCE_WEIGHTS['whale'],
            }
        msg = (
            "⚙️ <b>Your Trading Settings</b>\n\n"
            f"📊 Symbols: <code>{settings['symbols']}</code>\n"
            f"🎯 Consensus Threshold: {settings['consensus_threshold']}\n"
            f"📊 Min Sources: {settings['min_sources']}\n"
            f"📉 Volatility Min: {settings['volatility_min']}\n"
            f"💵 Risk per Trade: {settings['per_trade_risk_pct']*100:.1f}%\n"
            f"📉 Max Daily Loss: {settings['max_daily_loss_pct']*100:.1f}%\n"
            f"📊 Max Drawdown: {settings['max_drawdown']*100:.1f}%\n"
            f"📌 Max Positions: {settings['max_positions_global']}\n"
            f"📈 Trend Filter: {'✅' if settings['trend_filter'] else '❌'}\n"
            f"⚖️ Weight MA: {settings['weight_ma']}\n"
            f"⚖️ Weight Volume Momentum: {settings['weight_volume_momentum']}\n"
            f"⚖️ Weight Breakout: {settings['weight_breakout']}\n"
            f"⚖️ Weight Whale: {settings['weight_whale']}\n\n"
            "Use /set &lt;key&gt; &lt;value&gt; to change, e.g.\n"
            "<code>/set consensus_threshold 0.45</code>\n"
            "<code>/set symbols BTC-USDT,ETH-USDT</code>\n"
            "Or /reset to restore defaults."
        )
        await update.message.reply_text(msg, parse_mode='HTML', reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in settings: {e}", exc_info=True)

async def set_cmd(update: Update, context):
    try:
        chat_id = update.effective_user.id
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("Usage: /set &lt;key&gt; &lt;value&gt;\nExample: /set consensus_threshold 0.45", reply_markup=get_main_keyboard())
            return
        key = args[0]
        value = ' '.join(args[1:])
        numeric_keys = ['consensus_threshold', 'volatility_min', 'per_trade_risk_pct', 'max_daily_loss_pct', 'max_drawdown', 'weight_ma', 'weight_volume_momentum', 'weight_breakout', 'weight_whale']
        int_keys = ['min_sources', 'max_positions_global']
        bool_keys = ['trend_filter']
        if key in numeric_keys:
            try:
                value = float(value)
            except ValueError:
                await update.message.reply_text(f"Invalid number for {key}.", reply_markup=get_main_keyboard())
                return
        elif key in int_keys:
            try:
                value = int(value)
            except ValueError:
                await update.message.reply_text(f"Invalid integer for {key}.", reply_markup=get_main_keyboard())
                return
        elif key in bool_keys:
            value = value.lower() in ['true', '1', 'yes', 'on']
        elif key == 'symbols':
            symbols = [s.strip() for s in value.split(',') if s.strip()]
            if not symbols:
                await update.message.reply_text("Symbols cannot be empty.", reply_markup=get_main_keyboard())
                return
        else:
            await update.message.reply_text(f"Unknown key: {key}. Available: symbols, consensus_threshold, min_sources, volatility_min, per_trade_risk_pct, max_daily_loss_pct, max_drawdown, max_positions_global, trend_filter, weight_ma, weight_volume_momentum, weight_breakout, weight_whale", reply_markup=get_main_keyboard())
            return
        if trader_global:
            trader_global.settings_manager.set(chat_id, key, value)
            if key == 'symbols':
                trader_global.reload_settings()
            else:
                trader_global.override_settings = trader_global.settings_manager.get(chat_id)
            await update.message.reply_text(f"✅ {key} set to {value}", reply_markup=get_main_keyboard())
        else:
            await update.message.reply_text("Trader not initialized.", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in set_cmd: {e}", exc_info=True)

async def reset_cmd(update: Update, context):
    try:
        await update.message.reply_text("⏳ Resetting...", reply_markup=get_main_keyboard())
        chat_id = update.effective_user.id
        if trader_global:
            trader_global.settings_manager.reset(chat_id)
            trader_global.override_settings = None
            trader_global.reload_settings()
            await update.message.reply_text("✅ All settings reset to defaults.", reply_markup=get_main_keyboard())
        else:
            await update.message.reply_text("Trader not ready.", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error in reset: {e}", exc_info=True)

# ---------- Button handler ----------
async def handle_button(update: Update, context):
    try:
        text = update.message.text
        logger.info(f"Button pressed: {text}")
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
        elif text == "⚙️ Settings":
            await settings_cmd(update, context)
        elif text == "❓ Help":
            await help_cmd(update, context)
    except Exception as e:
        logger.error(f"Error in handle_button: {e}", exc_info=True)

# ---------------------------- KEEP-ALIVE (fixed: wait for Flask to start) ----------------------------
def keep_alive():
    port = os.getenv('PORT', 5000)
    url = f"http://localhost:{port}/"
    # Wait for Flask to be ready
    time.sleep(5)
    while True:
        try:
            requests.get(url, timeout=5)
            logger.debug("Keep-alive ping sent")
        except Exception as e:
            logger.error(f"Keep-alive ping failed: {e}")
        time.sleep(240)

# ---------------------------- TRADING LOOP WRAPPER (Auto-Restart) ----------------------------
def start_trading_loop_with_restart(trader):
    while True:
        try:
            if trader is None:
                logger.error("Trader is None, cannot start loop.")
                time.sleep(30)
                continue
            trader.running = True
            trader.run_loop()
        except Exception as e:
            logger.error(f"Trading loop crashed: {e}. Restarting in 10 seconds...", exc_info=True)
            time.sleep(10)
            continue

# ---------------------------- TELEGRAM RUNNER (Fixed Event Loop & Conflict) ----------------------------
def run_telegram():
    if not Config.TELEGRAM_TOKEN:
        logger.warning("No Telegram token, skipping bot.")
        return
    app_tg = Application.builder().token(Config.TELEGRAM_TOKEN).build()
    app_tg.add_handler(CommandHandler("start", start))
    app_tg.add_handler(CommandHandler("help", help_cmd))
    app_tg.add_handler(CommandHandler("ping", ping_cmd))
    app_tg.add_handler(CommandHandler("status", status_cmd))
    app_tg.add_handler(CommandHandler("performance", performance))
    app_tg.add_handler(CommandHandler("backtest", backtest))
    app_tg.add_handler(CommandHandler("scan", scan))
    app_tg.add_handler(CommandHandler("pause", pause))
    app_tg.add_handler(CommandHandler("resume", resume))
    app_tg.add_handler(CommandHandler("restartloop", restartloop_cmd))
    app_tg.add_handler(CommandHandler("settings", settings_cmd))
    app_tg.add_handler(CommandHandler("set", set_cmd))
    app_tg.add_handler(CommandHandler("reset", reset_cmd))
    app_tg.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_button))

    while True:
        try:
            logger.info("Telegram bot started, polling...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            app_tg.run_polling()
        except Conflict:
            logger.warning("Conflict detected (another instance running). Stopping and retrying in 15 seconds...")
            try:
                app_tg.stop()
            except:
                pass
            time.sleep(15)
            continue
        except Exception as e:
            logger.error(f"Telegram polling crashed: {e}. Restarting in 10 seconds...")
            time.sleep(10)
            continue

# ---------------------------- MAIN ----------------------------
if __name__ == "__main__":
    db = TradeDB(Config.DB_FILE)
    risk_mgr = RiskManager(Config.INITIAL_BALANCE, {
        'MAX_DAILY_LOSS_PCT': Config.MAX_DAILY_LOSS_PCT,
        'MAX_POSITIONS_GLOBAL': Config.MAX_POSITIONS_GLOBAL,
        'MAX_POSITIONS_PER_SYMBOL': Config.MAX_POSITIONS_PER_SYMBOL,
        'CONSECUTIVE_LOSS_LIMIT': Config.CONSECUTIVE_LOSS_LIMIT,
        'PER_TRADE_RISK_PCT': Config.PER_TRADE_RISK_PCT,
        'VOLATILITY_MIN': Config.VOLATILITY_MIN,
        'MAX_DRAWDOWN': Config.MAX_DRAWDOWN,
    })
    live_broker = LiveBroker(Config.EXCHANGE_NAME, Config.EXCHANGE_API_KEY, Config.EXCHANGE_SECRET)

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

    # Send startup message
    if trader.telegram_token and trader.chat_id:
        try:
            trader.send_alert("🤖 Bot is starting up and online!")
        except Exception as e:
            logger.error(f"Startup message failed: {e}")

    # Start Flask server
    threading.Thread(
        target=app.run,
        kwargs={'host': '0.0.0.0', 'port': int(os.getenv('PORT', 5000))},
        daemon=True
    ).start()

    # Start keep-alive thread (with initial sleep)
    threading.Thread(target=keep_alive, daemon=True).start()
    logger.info("Keep-alive thread started.")

    # Start trading loop with auto-restart
    loop_thread = threading.Thread(target=start_trading_loop_with_restart, args=(trader,), daemon=True)
    loop_thread.start()
    logger.info("Trading loop thread started with auto-restart.")

    # Run Telegram bot on main thread
    logger.info("Starting Telegram bot on main thread...")
    run_telegram()