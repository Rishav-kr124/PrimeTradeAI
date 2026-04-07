import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import feedparser
import pytz
import concurrent.futures
import datetime as dt
import time
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime

# ─────────────────────────────────────────────
# LAYER 1: SECURITY & CONFIGURATION
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prime Trade AI | Ultra Terminal",
    page_icon="🦅",
    layout="wide"
)

def configure_ai():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error("⚠️ API Key not found in secrets.")
        return None

model = configure_ai()

# ─────────────────────────────────────────────
# STOCK UNIVERSE WITH SECTOR TAGS
# ─────────────────────────────────────────────
LIQUID_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", "BHARTIARTL.NS",
    "ITC.NS", "KOTAKBANK.NS", "AXISBANK.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS",
    "TITAN.NS", "MARUTI.NS", "TATASTEEL.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "M&M.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "APOLLOHOSP.NS", "BAJAJFINSV.NS", "BAJAJ-AUTO.NS", "BPCL.NS", "BRITANNIA.NS",
    "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "NESTLEIND.NS",
    "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TECHM.NS", "UPL.NS",
    "WIPRO.NS", "ZOMATO.NS", "DLF.NS", "HAL.NS", "VBL.NS", "JIOFIN.NS", "DMART.NS", "SIEMENS.NS",
    "TRENT.NS", "BEL.NS", "PFC.NS", "REC.NS", "IOC.NS", "GAIL.NS", "BANKBARODA.NS", "PNB.NS",
    "CHOLAFIN.NS", "SHRIRAMFIN.NS", "TVSMOTOR.NS", "GODREJCP.NS", "HAVELLS.NS", "ABB.NS", "INDIGO.NS",
    "LTIM.NS", "PIDILITIND.NS", "VEDL.NS", "AMBUJACEM.NS", "CANBK.NS", "NAUKRI.NS", "SBILIFE.NS"
]

SECTOR_MAP = {
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy", "IOC.NS": "Energy", "GAIL.NS": "Energy",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT", "TECHM.NS": "IT",
    "LTIM.NS": "IT", "NAUKRI.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking", "KOTAKBANK.NS": "Banking",
    "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking", "BANKBARODA.NS": "Banking", "PNB.NS": "Banking",
    "CANBK.NS": "Banking",
    "BAJFINANCE.NS": "NBFC", "BAJAJFINSV.NS": "NBFC", "CHOLAFIN.NS": "NBFC", "SHRIRAMFIN.NS": "NBFC",
    "HDFCLIFE.NS": "Insurance", "SBILIFE.NS": "Insurance",
    "SUNPHARMA.NS": "Pharma", "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "APOLLOHOSP.NS": "Healthcare",
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals", "VEDL.NS": "Metals",
    "COALINDIA.NS": "Metals",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "HEROMOTOCO.NS": "Auto", "EICHERMOT.NS": "Auto", "TVSMOTOR.NS": "Auto",
    "ASIANPAINT.NS": "Consumer", "HINDUNILVR.NS": "Consumer", "BRITANNIA.NS": "Consumer",
    "NESTLEIND.NS": "Consumer", "GODREJCP.NS": "Consumer", "TATACONSUM.NS": "Consumer",
    "TITAN.NS": "Consumer", "PIDILITIND.NS": "Consumer", "ITC.NS": "Consumer", "VBL.NS": "Consumer",
    "ULTRACEMCO.NS": "Cement", "GRASIM.NS": "Cement", "AMBUJACEM.NS": "Cement",
    "BHARTIARTL.NS": "Telecom", "ADANIENT.NS": "Conglomerate", "ADANIPORTS.NS": "Infra",
    "DLF.NS": "Realty", "NTPC.NS": "Power", "POWERGRID.NS": "Power", "PFC.NS": "Power", "REC.NS": "Power",
    "BEL.NS": "Defence", "HAL.NS": "Defence", "SIEMENS.NS": "Capital Goods", "ABB.NS": "Capital Goods",
    "HAVELLS.NS": "Capital Goods", "ZOMATO.NS": "Internet", "DMART.NS": "Retail",
    "TRENT.NS": "Retail", "JIOFIN.NS": "NBFC", "UPL.NS": "Agro", "INDIGO.NS": "Aviation",
}

# ─────────────────────────────────────────────
# LAYER 2: HELPER FUNCTIONS
# ─────────────────────────────────────────────

def get_live_news(ticker):
    """Fetches top 3 news headlines for a stock/index."""
    clean_ticker = ticker.replace(".NS", "").replace("^", "")
    rss_url = f"https://news.google.com/rss/search?q={clean_ticker}+stock+india&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(rss_url)
        headlines = [f"- {entry.title}" for entry in feed.entries[:3]]
        return "\n".join(headlines) if headlines else "No significant news found."
    except Exception:
        return "News unavailable."


def get_macro_data():
    """Fetches global indices and commodity prices with timeout guard."""
    symbols = {
        "Nifty 50": "^NSEI",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
        "Gold (Comex)": "GC=F",
        "Silver (Comex)": "SI=F",
    }
    macro_data = {}
    for name, ticker in symbols.items():
        try:
            df = yf.Ticker(ticker).history(period="5d", timeout=10)
            if len(df) >= 2:
                current = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2]
                change = ((current - prev) / prev) * 100
                macro_data[name] = {"price": round(float(current), 2), "change": round(float(change), 2)}
            else:
                macro_data[name] = {"price": 0.0, "change": 0.0}
        except Exception:
            macro_data[name] = {"price": 0.0, "change": 0.0}
    return macro_data


class MarketTimer:
    @staticmethod
    def get_status():
        IST = pytz.timezone("Asia/Kolkata")
        now = datetime.now(IST)
        market_open = dt.time(9, 15)
        market_close = dt.time(15, 30)
        if now.weekday() >= 5:
            return "CLOSED", "🔴 Closed (Weekend)", now
        if market_open <= now.time() <= market_close:
            return "OPEN", "🟢 Market Open", now
        return "CLOSED", "🔴 Market Closed", now


# ─────────────────────────────────────────────
# LAYER 3: PATTERN RECOGNITION ENGINE
# ─────────────────────────────────────────────
class PatternEngine:
    @staticmethod
    def detect_pattern(df: pd.DataFrame) -> str:
        try:
            highs = df["High"].values
            lows  = df["Low"].values
            peak_idx   = argrelextrema(highs, np.greater, order=5)[0]
            trough_idx = argrelextrema(lows,  np.less,    order=5)[0]

            if len(peak_idx) < 3 or len(trough_idx) < 3:
                return "Consolidation (Insufficient pivots)"

            p1, p2, p3 = highs[peak_idx][-3], highs[peak_idx][-2], highs[peak_idx][-1]
            t1, t2, t3 = lows[trough_idx][-3], lows[trough_idx][-2], lows[trough_idx][-1]

            tol = 0.012  # 1.2% tolerance

            if abs(p3 - p2) / p2 <= tol and p3 > p1:
                return "🛑 DOUBLE TOP (Bearish Reversal)"
            if abs(t3 - t2) / t2 <= tol and t3 < t1:
                return "🚀 DOUBLE BOTTOM (Bullish Reversal)"
            if p2 > p1 and p2 > p3 and abs(p1 - p3) / p1 < tol * 1.5:
                return "📉 HEAD & SHOULDERS (Strong Sell)"
            if t2 < t1 and t2 < t3 and abs(t1 - t3) / t1 < tol * 1.5:
                return "📈 INV. HEAD & SHOULDERS (Strong Buy)"
            if p3 > p2 > p1 and t3 > t2 > t1:
                return "✅ Higher Highs + Higher Lows (Strong Uptrend)"
            if p3 < p2 < p1 and t3 < t2 < t1:
                return "⚠️ Lower Highs + Lower Lows (Strong Downtrend)"
            if p3 > p2 > p1 and abs(t3 - t1) / t1 < tol:
                return "📐 ASCENDING TRIANGLE (Bullish Continuation)"
            if t3 < t2 < t1 and abs(p3 - p1) / p1 < tol:
                return "📐 DESCENDING TRIANGLE (Bearish Continuation)"
            return "↔️ Sideways / Choppy"
        except Exception:
            return "Insufficient Data for Pattern"

    @staticmethod
    def fibonacci_levels(df: pd.DataFrame) -> dict:
        """Returns key Fibonacci retracement levels from recent swing."""
        try:
            window = df.tail(60)
            high = float(window["High"].max())
            low  = float(window["Low"].min())
            diff = high - low
            return {
                "Swing High": round(high, 2),
                "0.236": round(high - 0.236 * diff, 2),
                "0.382": round(high - 0.382 * diff, 2),
                "0.500": round(high - 0.500 * diff, 2),
                "0.618": round(high - 0.618 * diff, 2),
                "0.786": round(high - 0.786 * diff, 2),
                "Swing Low": round(low, 2),
            }
        except Exception:
            return {}

    @staticmethod
    def pivot_levels(df: pd.DataFrame) -> dict:
        """Classic pivot point S/R using previous session."""
        try:
            prev = df.iloc[-2]
            H, L, C = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
            P  = (H + L + C) / 3
            R1 = round(2 * P - L, 2)
            R2 = round(P + (H - L), 2)
            S1 = round(2 * P - H, 2)
            S2 = round(P - (H - L), 2)
            return {"Pivot": round(P, 2), "R1": R1, "R2": R2, "S1": S1, "S2": S2}
        except Exception:
            return {}


# ─────────────────────────────────────────────
# LAYER 4: INDICATOR ENGINE (Confluence Scoring)
# ─────────────────────────────────────────────
class IndicatorEngine:
    """
    Computes a full suite of technical indicators and returns a
    confluence score (0–100) alongside individual indicator readings.
    """

    @staticmethod
    def _safe_scalar(series) -> float:
        try:
            if series is None or (hasattr(series, "empty") and series.empty):
                return 0.0
            val = series.iloc[-1]
            if pd.isna(val):
                return 0.0
            return float(val)
        except Exception:
            return 0.0

    @staticmethod
    def compute(df: pd.DataFrame) -> dict:
        ind = {}
        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # ── Core Trend ──────────────────────────────
        for length in (20, 50, 100, 200):
            key = f"EMA_{length}"
            ema = ta.ema(close, length=length)
            ind[key] = round(IndicatorEngine._safe_scalar(ema), 2) if ema is not None else 0.0

        # ── RSI ─────────────────────────────────────
        rsi = ta.rsi(close, length=14)
        ind["RSI"] = round(IndicatorEngine._safe_scalar(rsi), 2)

        # ── MACD ────────────────────────────────────
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            ind["MACD"]        = round(IndicatorEngine._safe_scalar(macd_df.iloc[:, 0]), 4)
            ind["MACD_Signal"] = round(IndicatorEngine._safe_scalar(macd_df.iloc[:, 2]), 4)
            ind["MACD_Hist"]   = round(IndicatorEngine._safe_scalar(macd_df.iloc[:, 1]), 4)
        else:
            ind["MACD"] = ind["MACD_Signal"] = ind["MACD_Hist"] = 0.0

        # ── Stochastic %K / %D ───────────────────────
        stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            ind["STOCH_K"] = round(IndicatorEngine._safe_scalar(stoch.iloc[:, 0]), 2)
            ind["STOCH_D"] = round(IndicatorEngine._safe_scalar(stoch.iloc[:, 1]), 2)
        else:
            ind["STOCH_K"] = ind["STOCH_D"] = 0.0

        # ── ADX (Trend Strength) ─────────────────────
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and not adx_df.empty:
            ind["ADX"]   = round(IndicatorEngine._safe_scalar(adx_df.iloc[:, 0]), 2)
            ind["DI_POS"] = round(IndicatorEngine._safe_scalar(adx_df.iloc[:, 1]), 2)
            ind["DI_NEG"] = round(IndicatorEngine._safe_scalar(adx_df.iloc[:, 2]), 2)
        else:
            ind["ADX"] = ind["DI_POS"] = ind["DI_NEG"] = 0.0

        # ── ATR (Volatility) ─────────────────────────
        atr = ta.atr(high, low, close, length=14)
        ind["ATR"] = round(IndicatorEngine._safe_scalar(atr), 2)

        # ── Bollinger Bands ──────────────────────────
        bb = ta.bbands(close, length=20)
        if bb is not None and not bb.empty:
            # Column names from pandas_ta: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
            bb_cols = bb.columns.tolist()
            lower_col  = next((c for c in bb_cols if c.startswith("BBL")), None)
            mid_col    = next((c for c in bb_cols if c.startswith("BBM")), None)
            upper_col  = next((c for c in bb_cols if c.startswith("BBU")), None)
            pct_col    = next((c for c in bb_cols if c.startswith("BBP")), None)
            ind["BB_LOWER"] = round(IndicatorEngine._safe_scalar(bb[lower_col])  if lower_col else 0.0, 2)
            ind["BB_MID"]   = round(IndicatorEngine._safe_scalar(bb[mid_col])    if mid_col  else 0.0, 2)
            ind["BB_UPPER"] = round(IndicatorEngine._safe_scalar(bb[upper_col])  if upper_col else 0.0, 2)
            ind["BB_PCT"]   = round(IndicatorEngine._safe_scalar(bb[pct_col])    if pct_col  else 0.5, 4)
        else:
            ind["BB_LOWER"] = ind["BB_MID"] = ind["BB_UPPER"] = ind["BB_PCT"] = 0.0

        # ── OBV (Volume Trend) ───────────────────────
        obv = ta.obv(close, vol)
        if obv is not None and len(obv) >= 10:
            obv_now  = IndicatorEngine._safe_scalar(obv)
            obv_prev = float(obv.iloc[-10])
            ind["OBV_TREND"] = "Rising" if obv_now > obv_prev else "Falling"
        else:
            ind["OBV_TREND"] = "Unknown"

        # ── Williams %R ──────────────────────────────
        willr = ta.willr(high, low, close, length=14)
        ind["WILLR"] = round(IndicatorEngine._safe_scalar(willr), 2)

        # ── SuperTrend ───────────────────────────────
        st_df = ta.supertrend(high, low, close, length=10, multiplier=3)
        if st_df is not None and not st_df.empty:
            direction_col = next((c for c in st_df.columns if "SUPERTd" in c), None)
            ind["SUPERTREND_DIR"] = int(IndicatorEngine._safe_scalar(st_df[direction_col])) if direction_col else 0
        else:
            ind["SUPERTREND_DIR"] = 0

        # ── Volume Ratio ─────────────────────────────
        avg_vol_10 = float(vol.tail(10).mean()) if len(vol) >= 10 else 1.0
        ind["VOL_RATIO"] = round(float(vol.iloc[-1]) / avg_vol_10, 2) if avg_vol_10 > 0 else 1.0

        # ── Current Price ────────────────────────────
        ind["PRICE"] = round(float(close.iloc[-1]), 2)

        # ── ATR-based SL / Target ────────────────────
        atr_val = ind["ATR"] if ind["ATR"] > 0 else ind["PRICE"] * 0.01
        ind["SL_LONG"]     = round(ind["PRICE"] - 1.5 * atr_val, 2)
        ind["TARGET_LONG"] = round(ind["PRICE"] + 3.0 * atr_val, 2)
        ind["SL_SHORT"]    = round(ind["PRICE"] + 1.5 * atr_val, 2)
        ind["TARGET_SHORT"]= round(ind["PRICE"] - 3.0 * atr_val, 2)

        # ── Confluence Score (0–100) ──────────────────
        ind["SCORE_BULL"], ind["SCORE_BEAR"] = IndicatorEngine.confluence_score(ind)

        return ind

    @staticmethod
    def confluence_score(ind: dict) -> tuple[int, int]:
        """
        Each condition casts 1 vote for BULL or BEAR.
        Final score = (votes / total) * 100, rounded to int.
        """
        bull, bear, total = 0, 0, 0
        price = ind.get("PRICE", 0)

        def vote(condition_bull, condition_bear=None):
            nonlocal bull, bear, total
            total += 1
            if condition_bull:
                bull += 1
            elif condition_bear:
                bear += 1

        # EMA alignment
        vote(price > ind["EMA_20"] > ind["EMA_50"],  price < ind["EMA_20"] < ind["EMA_50"])
        vote(price > ind["EMA_50"] > ind["EMA_200"], price < ind["EMA_50"] < ind["EMA_200"])
        vote(ind["EMA_20"] > ind["EMA_50"],          ind["EMA_20"] < ind["EMA_50"])
        vote(price > ind["EMA_200"],                 price < ind["EMA_200"])

        # RSI zones
        vote(50 < ind["RSI"] < 70,  ind["RSI"] < 40)
        vote(ind["RSI"] > 55,       ind["RSI"] < 45)

        # MACD
        vote(ind["MACD"] > ind["MACD_Signal"] and ind["MACD_Hist"] > 0,
             ind["MACD"] < ind["MACD_Signal"] and ind["MACD_Hist"] < 0)

        # Stochastic
        vote(ind["STOCH_K"] > ind["STOCH_D"] and ind["STOCH_K"] < 80,
             ind["STOCH_K"] < ind["STOCH_D"] and ind["STOCH_K"] > 20)
        vote(ind["STOCH_K"] > 50, ind["STOCH_K"] < 50)

        # ADX / DI
        vote(ind["ADX"] > 25 and ind["DI_POS"] > ind["DI_NEG"],
             ind["ADX"] > 25 and ind["DI_NEG"] > ind["DI_POS"])

        # Bollinger
        vote(ind["BB_PCT"] > 0.5 and price < ind["BB_UPPER"],
             ind["BB_PCT"] < 0.5 and price > ind["BB_LOWER"])

        # OBV
        vote(ind["OBV_TREND"] == "Rising",  ind["OBV_TREND"] == "Falling")

        # Williams %R
        vote(-50 < ind["WILLR"] < -20,  ind["WILLR"] > -50)

        # SuperTrend
        vote(ind["SUPERTREND_DIR"] == 1, ind["SUPERTREND_DIR"] == -1)

        # Volume confirmation
        vote(ind["VOL_RATIO"] > 1.2, ind["VOL_RATIO"] < 0.7)

        bull_pct = round((bull / total) * 100) if total > 0 else 0
        bear_pct = round((bear / total) * 100) if total > 0 else 0
        return bull_pct, bear_pct


# ─────────────────────────────────────────────
# LAYER 5: SCANNER ENGINE
# ─────────────────────────────────────────────
class ScannerEngine:

    @staticmethod
    def safe_ai_request(prompt: str):
        for attempt in range(3):
            try:
                if model is None:
                    return None
                response = model.generate_content(prompt)
                if response and response.text:
                    return response
            except Exception as e:
                if "429" in str(e):
                    time.sleep(10)
                else:
                    return None
        return None

    @staticmethod
    def _fetch_df(ticker: str, period: str = "1y") -> pd.DataFrame | None:
        """Safe yfinance download with MultiIndex flattening."""
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=15)
            if df is None or df.empty:
                return None
            # Flatten MultiIndex — handles tickers with special chars like L&T
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df.dropna(subset=["Close", "High", "Low", "Open", "Volume"], inplace=True)
            return df if len(df) >= 50 else None
        except Exception:
            return None

    @staticmethod
    def analyze_stock_logic(ticker: str, mode: str) -> dict | None:
        period = "1y" if mode in ["INTRADAY", "SHORT", "SWING", "BREAKOUT", "MOMENTUM"] else "3mo"
        df = ScannerEngine._fetch_df(ticker, period)
        if df is None:
            return None

        min_bars = {"INTRADAY": 200, "SHORT": 200, "SWING": 50,
                    "BTST": 20, "BREAKOUT": 100, "MOMENTUM": 50,
                    "VOL_SURGE": 20, "RSI_REV": 20}
        if len(df) < min_bars.get(mode, 50):
            return None

        ind = IndicatorEngine.compute(df)
        price   = ind["PRICE"]
        sector  = SECTOR_MAP.get(ticker, "Other")
        atr_val = ind["ATR"] if ind["ATR"] > 0 else price * 0.01
        vol_r   = ind["VOL_RATIO"]
        rsi     = ind["RSI"]

        result_base = {
            "Ticker": ticker,
            "Sector": sector,
            "Price": price,
            "Bull Score": f"{ind['SCORE_BULL']}%",
            "Bear Score": f"{ind['SCORE_BEAR']}%",
            "RSI": rsi,
            "ADX": ind["ADX"],
            "Vol Ratio": vol_r,
        }

        # ── INTRADAY BUY ─────────────────────────────
        # Condition: Price above EMA200, opens near prior low (gap-up or at support),
        # volume surge, MACD bullish, RSI in healthy zone
        if mode == "INTRADAY":
            open_p = float(df["Open"].iloc[-1])
            prev_low = float(df["Low"].iloc[-2])
            above_ema200 = price > ind["EMA_200"] > 0
            opens_near_support = open_p <= prev_low * 1.005          # opens at/near prev low (support)
            bullish_macd = ind["MACD"] > ind["MACD_Signal"]
            healthy_rsi  = 45 < rsi < 70
            vol_ok       = vol_r > 1.05
            adx_ok       = ind["ADX"] > 20

            if above_ema200 and opens_near_support and bullish_macd and healthy_rsi and vol_ok and adx_ok:
                sl     = round(price - 1.5 * atr_val, 2)
                target = round(price + 3.0 * atr_val, 2)
                rr     = round((target - price) / (price - sl), 2) if price > sl else 0
                return {**result_base, "Type": "🔥 Intraday Buy",
                        "Entry": price, "SL": sl, "Target": target, "R:R": rr}

        # ── SHORT SELL ───────────────────────────────
        elif mode == "SHORT":
            open_p = float(df["Open"].iloc[-1])
            prev_high = float(df["High"].iloc[-2])
            below_ema200 = price < ind["EMA_200"] if ind["EMA_200"] > 0 else False
            opens_near_resistance = open_p >= prev_high * 0.995
            bearish_macd = ind["MACD"] < ind["MACD_Signal"]
            weak_rsi     = rsi < 50
            vol_ok       = vol_r > 1.05
            adx_ok       = ind["ADX"] > 20

            if below_ema200 and opens_near_resistance and bearish_macd and weak_rsi and vol_ok and adx_ok:
                sl     = round(price + 1.5 * atr_val, 2)
                target = round(price - 3.0 * atr_val, 2)
                rr     = round((price - target) / (sl - price), 2) if sl > price else 0
                return {**result_base, "Type": "🩸 Short Sell",
                        "Entry": price, "SL": sl, "Target": target, "R:R": rr}

        # ── BTST ─────────────────────────────────────
        elif mode == "BTST":
            prev_close = float(df["Close"].iloc[-2])
            day_change = (price - prev_close) / prev_close * 100
            closes_near_high = price >= float(df["High"].iloc[-1]) * 0.98
            positive_day  = day_change >= 0.8
            macd_bullish  = ind["MACD"] > ind["MACD_Signal"]
            vol_ok        = vol_r >= 0.9

            if positive_day and closes_near_high and macd_bullish and vol_ok:
                sl     = round(float(df["Low"].iloc[-1]), 2)
                target = round(price + 2.0 * atr_val, 2)
                rr     = round((target - price) / (price - sl), 2) if price > sl else 0
                return {**result_base, "Type": "🌙 BTST",
                        "Entry": price, "SL": sl, "Target": target, "R:R": rr}

        # ── SWING TRADE ──────────────────────────────
        elif mode == "SWING":
            ema50_ok  = price > ind["EMA_50"] > 0
            ema200_ok = price > ind["EMA_200"] > 0
            rsi_ok    = 50 < rsi < 65
            macd_ok   = ind["MACD"] > ind["MACD_Signal"] and ind["MACD_Hist"] > 0
            adx_ok    = ind["ADX"] > 20
            vol_ok    = vol_r > 0.9
            supertrend_ok = ind["SUPERTREND_DIR"] == 1

            if ema50_ok and ema200_ok and rsi_ok and macd_ok and adx_ok and vol_ok and supertrend_ok:
                sl     = round(ind["EMA_50"] - atr_val, 2)
                target = round(price + 4.0 * atr_val, 2)
                rr     = round((target - price) / (price - sl), 2) if price > sl else 0
                return {**result_base, "Type": "📈 Swing Trade",
                        "Entry": price, "SL": sl, "Target": target, "R:R": rr}

        # ── MOMENTUM BREAKOUT ────────────────────────
        elif mode == "BREAKOUT":
            week_high = float(df["High"].tail(52).max())  # 52-week high
            near_high = price >= week_high * 0.98         # within 2% of 52W high
            strong_vol = vol_r > 1.5
            adx_strong = ind["ADX"] > 30
            rsi_ok     = rsi > 55
            di_ok      = ind["DI_POS"] > ind["DI_NEG"]
            macd_ok    = ind["MACD_Hist"] > 0

            if near_high and strong_vol and adx_strong and rsi_ok and di_ok and macd_ok:
                sl     = round(week_high * 0.97, 2)
                target = round(price + 5.0 * atr_val, 2)
                rr     = round((target - price) / (price - sl), 2) if price > sl else 0
                return {**result_base, "Type": "💥 Momentum Breakout",
                        "Entry": price, "SL": sl, "Target": target, "R:R": rr}

        # ── VOLUME SURGE ─────────────────────────────
        elif mode == "VOL_SURGE":
            massive_vol = vol_r > 2.5
            price_up    = price > float(df["Open"].iloc[-1])   # green candle
            rsi_ok      = 40 < rsi < 75
            macd_ok     = ind["MACD"] > ind["MACD_Signal"]

            if massive_vol and price_up and rsi_ok and macd_ok:
                sl     = round(float(df["Low"].iloc[-1]), 2)
                target = round(price + 3.5 * atr_val, 2)
                rr     = round((target - price) / (price - sl), 2) if price > sl else 0
                return {**result_base, "Type": "🚨 Volume Surge",
                        "Entry": price, "SL": sl, "Target": target, "R:R": rr}

        # ── RSI REVERSAL ─────────────────────────────
        elif mode == "RSI_REV":
            oversold     = rsi < 35
            willr_os     = ind["WILLR"] < -75
            stoch_cross  = ind["STOCH_K"] > ind["STOCH_D"] and ind["STOCH_K"] < 30
            price_above_support = price > ind["BB_LOWER"] if ind["BB_LOWER"] > 0 else True
            obv_rising   = ind["OBV_TREND"] == "Rising"

            if oversold and willr_os and stoch_cross and price_above_support and obv_rising:
                sl     = round(price - 2.0 * atr_val, 2)
                target = round(price + 4.0 * atr_val, 2)
                rr     = round((target - price) / (price - sl), 2) if price > sl else 0
                return {**result_base, "Type": "🔄 RSI Reversal",
                        "Entry": price, "SL": sl, "Target": target, "R:R": rr}

        return None

    @staticmethod
    def scan_market(stock_list: list, mode: str = "INTRADAY") -> pd.DataFrame:
        results = []
        progress_bar = st.progress(0, text=f"⚡ Scanning {len(stock_list)} stocks for [{mode}] setups…")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(ScannerEngine.analyze_stock_logic, s, mode): s for s in stock_list}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                data = future.result()
                if data:
                    results.append(data)
                progress_bar.progress((i + 1) / len(stock_list),
                                      text=f"⚡ Scanned {i + 1}/{len(stock_list)} | Found: {len(results)}")

        progress_bar.empty()
        if not results:
            return pd.DataFrame()

        df_out = pd.DataFrame(results)
        # Sort by Bull Score descending
        if "Bull Score" in df_out.columns:
            df_out["_sort"] = df_out["Bull Score"].str.replace("%", "").astype(float)
            df_out.sort_values("_sort", ascending=False, inplace=True)
            df_out.drop(columns=["_sort"], inplace=True)
        return df_out.reset_index(drop=True)


# ─────────────────────────────────────────────
# LAYER 6: MARKET SENTIMENT
# ─────────────────────────────────────────────
def get_market_sentiment(nifty_change: float, news_text: str) -> str:
    prompt = f"""
    Act as a Stock Market Anchor.
    Nifty 50 changed by {nifty_change:.2f}% today.
    Recent Headlines: {news_text}

    Give a punchy, 2-line summary of the current market condition and sentiment.
    Keep it strictly to 2 lines. Use emojis.
    """
    response = ScannerEngine.safe_ai_request(prompt)
    return response.text if response else "Market sentiment unavailable right now."


# ─────────────────────────────────────────────
# LAYER 7: NIFTY OPTIONS ANALYZER
# ─────────────────────────────────────────────
def run_nifty_analysis():
    st.info("🦅 Analyzing Nifty 50 for Call/Put Levels…")
    try:
        df = yf.download("^NSEI", period="1mo", interval="15m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        ind = IndicatorEngine.compute(df)
        current_price = ind["PRICE"]
        resistance    = round(float(df["High"].tail(50).max()), 2)
        support       = round(float(df["Low"].tail(50).min()), 2)
        pivots        = PatternEngine.pivot_levels(df)
        fib           = PatternEngine.fibonacci_levels(df)

        prompt = f"""
        Act as an expert F&O derivatives analyst.
        Nifty 50 Current Price : {current_price}
        Pivot Points           : {pivots}
        Fibonacci Levels       : {fib}
        Immediate Resistance   : {resistance}
        Immediate Support      : {support}
        15-Min RSI             : {ind['RSI']}
        MACD Histogram         : {ind['MACD_Hist']} (positive = bullish)
        ADX                    : {ind['ADX']} (>25 = trending)
        SuperTrend Direction   : {'Bullish' if ind['SUPERTREND_DIR'] == 1 else 'Bearish'}
        Bull Confluence Score  : {ind['SCORE_BULL']}%

        Provide a very quick, actionable view for today:
        1. When to buy a CALL (level, target, stop-loss, expiry suggestion)
        2. When to buy a PUT  (level, target, stop-loss, expiry suggestion)
        3. Key Fibonacci / Pivot levels to watch
        4. Overall bias (Bullish / Bearish / Neutral) with reason
        Format cleanly with emojis. Be precise with numbers.
        """
        response = ScannerEngine.safe_ai_request(prompt)
        if response:
            st.success("✅ Nifty Call/Put Analysis Ready")
            st.markdown(response.text)
        else:
            st.error("AI Busy. Try again.")
    except Exception as e:
        st.error(f"Could not load Nifty data: {e}")


# ─────────────────────────────────────────────
# LAYER 8: ADVANCED DEEP ANALYSIS ENGINE
# ─────────────────────────────────────────────
def run_advanced_analysis(ticker: str):
    st.info(f"🦅 Ultra-Analyst scanning {ticker}…")
    try:
        df = ScannerEngine._fetch_df(ticker, period="2y")
        if df is None:
            st.error(f"❌ No data found for '{ticker}'. Check the symbol.")
            return
        if len(df) < 200:
            st.error(f"⚠️ Only {len(df)} days of data — need 200+ for full analysis.")
            return

        ind      = IndicatorEngine.compute(df)
        pattern  = PatternEngine.detect_pattern(df)
        fib      = PatternEngine.fibonacci_levels(df)
        pivots   = PatternEngine.pivot_levels(df)
        news     = get_live_news(ticker)
        sector   = SECTOR_MAP.get(ticker, "Unknown")
        price    = ind["PRICE"]
        atr_val  = ind["ATR"]

        # Intraday / Swing / Positional levels from ATR
        sl_long     = ind["SL_LONG"]
        tgt_long    = ind["TARGET_LONG"]
        sl_short    = ind["SL_SHORT"]
        tgt_short   = ind["TARGET_SHORT"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"₹{price}")
        col2.metric("Bull Score 🟢", f"{ind['SCORE_BULL']}%")
        col3.metric("Bear Score 🔴", f"{ind['SCORE_BEAR']}%")

        with st.expander("📐 Full Indicator Dashboard", expanded=True):
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("RSI (14)", ind["RSI"])
            ic2.metric("MACD Histogram", ind["MACD_Hist"])
            ic3.metric("ADX", ind["ADX"])
            ic4.metric("ATR", ind["ATR"])

            ic5, ic6, ic7, ic8 = st.columns(4)
            ic5.metric("EMA 20", ind["EMA_20"])
            ic6.metric("EMA 50", ind["EMA_50"])
            ic7.metric("EMA 200", ind["EMA_200"])
            ic8.metric("SuperTrend", "🟢 Bullish" if ind["SUPERTREND_DIR"] == 1 else "🔴 Bearish")

            ic9, ic10, ic11, ic12 = st.columns(4)
            ic9.metric("Stoch %K", ind["STOCH_K"])
            ic10.metric("Stoch %D", ind["STOCH_D"])
            ic11.metric("Williams %R", ind["WILLR"])
            ic12.metric("OBV Trend", ind["OBV_TREND"])

            ic13, ic14, ic15, ic16 = st.columns(4)
            ic13.metric("BB Upper", ind["BB_UPPER"])
            ic14.metric("BB Mid",   ind["BB_MID"])
            ic15.metric("BB Lower", ind["BB_LOWER"])
            ic16.metric("Vol Ratio", f"{ind['VOL_RATIO']}x")

        with st.expander("📊 Chart Pattern & Key Levels"):
            st.markdown(f"**Pattern Detected:** {pattern}")
            st.markdown("**Fibonacci Retracement Levels (Last 60 days):**")
            st.json(fib)
            st.markdown("**Pivot Points (Previous Session):**")
            st.json(pivots)
            st.markdown(f"**ATR-Based Levels** | SL (Long): ₹{sl_long} | Target (Long): ₹{tgt_long} | "
                        f"SL (Short): ₹{sl_short} | Target (Short): ₹{tgt_short}")

        prompt = f"""
        Act as a Senior Technical Analyst at a top-tier brokerage.
        Stock: {ticker} | Sector: {sector}

        ── TECHNICALS ──
        Price        : ₹{price}
        Pattern      : {pattern}
        RSI          : {ind['RSI']}   (>70 overbought, <30 oversold)
        MACD         : {ind['MACD']} | Signal: {ind['MACD_Signal']} | Hist: {ind['MACD_Hist']}
        ADX          : {ind['ADX']} (>25 trending, >40 strong)
        DI+/DI-      : {ind['DI_POS']} / {ind['DI_NEG']}
        Stochastic   : K={ind['STOCH_K']} D={ind['STOCH_D']}
        Williams %R  : {ind['WILLR']}
        SuperTrend   : {'Bullish' if ind['SUPERTREND_DIR'] == 1 else 'Bearish'}
        OBV Trend    : {ind['OBV_TREND']}
        EMA 20/50/200: {ind['EMA_20']} / {ind['EMA_50']} / {ind['EMA_200']}
        BB Upper/Mid/Lower: {ind['BB_UPPER']} / {ind['BB_MID']} / {ind['BB_LOWER']}
        ATR          : {atr_val}
        Volume Ratio : {ind['VOL_RATIO']}x average

        ── S/R LEVELS ──
        Fibonacci    : {fib}
        Pivot Points : {pivots}

        ── CONFLUENCE ──
        Bull Score: {ind['SCORE_BULL']}% | Bear Score: {ind['SCORE_BEAR']}%

        ── NEWS ──
        {news}

        ── YOUR MISSION ──
        1. Overall Market Structure Analysis (2-3 lines, mention EMA alignment & trend)
        2. Pattern + Indicator Confluence Summary
        3. TRADE SETUPS — provide BOTH:
           a) LONG SETUP: Entry zone, Stop Loss, T1, T2 (with exact ₹ prices)
           b) SHORT SETUP: Entry zone, Stop Loss, T1, T2 (with exact ₹ prices)
        4. Recommended Timeframe (Intraday / BTST / Swing / Positional) with reason
        5. Key risk factors and what to watch
        6. Final Verdict: STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL

        Format with clear headers, emojis. Be quantitative and specific. No vague advice.
        """

        with st.spinner("🤖 AI Analyst generating report…"):
            response = ScannerEngine.safe_ai_request(prompt)

        if response:
            st.success(f"✅ Full Analysis Report — {ticker}")
            st.markdown(response.text)
        else:
            st.error("AI Busy. Try again in a moment.")

    except Exception as e:
        st.error(f"Processing Error: {e}")


# ─────────────────────────────────────────────
# LAYER 9: DASHBOARD
# ─────────────────────────────────────────────
def render_dashboard():
    status, status_label, current_time = MarketTimer.get_status()

    # ── SIDEBAR ──
    st.sidebar.title("🦅 Prime Trade AI")
    st.sidebar.markdown(f"**Status:** {status_label}")
    st.sidebar.markdown(f"**IST:** {current_time.strftime('%H:%M:%S')}")
    st.sidebar.divider()
    st.sidebar.subheader("🪙 Commodities")
    macro = get_macro_data()

    for key, symbol in [("Gold (Comex)", "$"), ("Silver (Comex)", "$")]:
        if key in macro:
            st.sidebar.metric(key.split(" ")[0], f"{symbol}{macro[key]['price']:,.2f}",
                              f"{macro[key]['change']:+.2f}%")
    st.sidebar.divider()

    # ── MAIN HEADER ──
    st.title("Prime Trade AI | Ultra Terminal 🦅")

    c1, c2, c3 = st.columns(3)
    for col, key, flag in [(c1, "Nifty 50", "🇮🇳"), (c2, "Dow Jones", "🇺🇸"), (c3, "Nasdaq", "🇺🇸")]:
        if key in macro:
            col.metric(f"{flag} {key}", f"{macro[key]['price']:,.2f}", f"{macro[key]['change']:+.2f}%")

    st.divider()

    # ── MARKET SENTIMENT ──
    st.subheader("📰 Live Market Condition")
    nifty_change = macro.get("Nifty 50", {}).get("change", 0.0)
    general_news = get_live_news("NIFTY 50")
    with st.spinner("Analyzing global sentiment…"):
        st.info(get_market_sentiment(nifty_change, general_news))
    with st.expander("View Latest Market Headlines"):
        st.markdown(general_news)

    st.divider()

    # ── TABS ──
    tab1, tab2, tab3 = st.tabs(["🔥 Stock Scanners", "🔎 Deep Analyzer", "📊 Nifty Options"])

    with tab1:
        st.markdown("### 🚦 Multi-Mode Market Scanner")
        st.caption("Each scan checks confluence of EMA, MACD, RSI, ADX, SuperTrend, OBV & more. Results sorted by Bull Score.")

        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7 = st.columns(3)

        with col1:
            if st.button("⚡ Intraday Buy", use_container_width=True):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, "INTRADAY")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"✅ {len(df)} setups found")
                else:
                    st.warning("No Intraday Buy setups found today.")

        with col2:
            if st.button("🩸 Short Sell", use_container_width=True):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, "SHORT")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No Short Sell setups found.")

        with col3:
            if st.button("🌙 BTST", use_container_width=True):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, "BTST")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No BTST setups found.")

        with col4:
            if st.button("📈 Swing Trade", use_container_width=True):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, "SWING")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No Swing setups found.")

        with col5:
            if st.button("💥 Momentum Breakout", use_container_width=True):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, "BREAKOUT")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No Breakout setups found.")

        with col6:
            if st.button("🚨 Volume Surge", use_container_width=True):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, "VOL_SURGE")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No Volume Surge setups found.")

        with col7:
            if st.button("🔄 RSI Reversal", use_container_width=True):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, "RSI_REV")
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No RSI Reversal setups found.")

    with tab2:
        st.markdown("### 🔎 Deep Pattern & Multi-Indicator Analysis")
        ticker_input = st.text_input("Enter NSE Symbol (e.g., ZOMATO.NS):").strip().upper()
        if ticker_input:
            if ticker_input.endswith(".NS"):
                run_advanced_analysis(ticker_input)
            else:
                st.warning("⚠️ Please append `.NS` to the symbol (e.g., RELIANCE.NS)")

    with tab3:
        st.markdown("### 🦅 Nifty 50 Options Levels (F&O)")
        if st.button("🔮 Analyze Nifty Levels", use_container_width=False):
            run_nifty_analysis()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    render_dashboard()
