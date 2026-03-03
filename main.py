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
from scipy.signal import argrelextrema # For finding peaks
from datetime import datetime

# --- LAYER 1: SECURITY & CONFIGURATION ---
st.set_page_config(
    page_title="Prime Trade AI | Ultra Terminal",
    page_icon="🦅",
    layout="wide"
)

# 1. SETUP AI (Using Stable Model)
def configure_ai():
    try:
        # 👇 PASTE YOUR KEY HERE. SECURITY WARNING: Keep this secret!
         
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        return None

model = configure_ai()

# 2. STOCK UNIVERSE
LIQUID_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS", "BHARTIARTL.NS", 
    "ITC.NS", "L&T.NS", "KOTAKBANK.NS", "AXISBANK.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", 
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

# --- LAYER 2: HELPER FUNCTIONS ---

def get_live_news(ticker):
   def get_macro_data():
       """Fetches global indices and commodity prices."""
       symbols = {
            "Nifty 50": "^NSEI", 
            "Dow Jones": "^DJI", 
            "Nasdaq": "^IXIC",
            "Gold (Comex)": "GC=F",
            "Silver (Comex)": "SI=F"
        }
        macro_data = {}
        for name, ticker in symbols.items():
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period="5d")
                if len(df) >= 2:
                    current = df['Close'].iloc[-1]
                    prev = df['Close'].iloc[-2]
                    change = ((current - prev) / prev) * 100
                    macro_data[name] = {"price": round(current, 2), "change": round(change, 2)}
            except:
                macro_data[name] = {"price": 0.0, "change": 0.0}
        return macro_data

def get_market_sentiment(nifty_change, news_text):
    """Uses AI to give a 2-line market condition summary."""
    prompt = f"""
    Act as a Stock Market Anchor. 
    Nifty 50 changed by {nifty_change}% today.
    Recent Headlines: {news_text}
    
    Give a punchy, 2-line summary of the current market condition and sentiment. 
    Keep it strictly to 2 lines. Use emojis.
    """
    response = ScannerEngine.safe_ai_request(prompt)
    return response.text if response else "Market sentiment unavailable right now."
class MarketTimer:
    @staticmethod
    def get_status():
        IST = pytz.timezone('Asia/Kolkata')
        now = datetime.now(IST)
        market_open = dt.time(9, 15)
        market_close = dt.time(15, 30)
        current_time_only = now.time()

        if market_open <= current_time_only <= market_close:
            if now.weekday() >= 5: return "CLOSED", "🔴 Closed (Weekend)", now
            return "OPEN", "🟢 Market Open", now
        else:
            return "CLOSED", "🔴 Market Closed", now

# --- LAYER 3: PATTERN RECOGNITION ENGINE ---
class PatternEngine:
    @staticmethod
    def detect_pattern(df):
        """Uses Math to find Head & Shoulders, Double Tops, etc."""
        try:
            highs = df['High'].values
            lows = df['Low'].values
            
            peak_idx = argrelextrema(highs, np.greater, order=5)[0]
            trough_idx = argrelextrema(lows, np.less, order=5)[0]
            
            if len(peak_idx) < 3 or len(trough_idx) < 3:
                return "Consolidation (No clear pattern)"

            last_peaks = highs[peak_idx][-3:]
            last_troughs = lows[trough_idx][-3:]
            
            p1, p2, p3 = last_peaks[-3], last_peaks[-2], last_peaks[-1]
            t1, t2, t3 = last_troughs[-3], last_troughs[-2], last_troughs[-1]

            if abs(p3 - p2) <= (p3 * 0.01) and p3 > p1:
                return "🛑 DOUBLE TOP DETECTED (Bearish Reversal)"
            if abs(t3 - t2) <= (t3 * 0.01) and t3 < t1:
                return "🚀 DOUBLE BOTTOM DETECTED (Bullish Reversal)"
            if p2 > p1 and p2 > p3 and abs(p1 - p3) < (p1 * 0.02):
                return "📉 HEAD & SHOULDERS (Strong Sell)"
            if t2 < t1 and t2 < t3 and abs(t1 - t3) < (t1 * 0.02):
                return "📈 INV. HEAD & SHOULDERS (Strong Buy)"
            if p3 > p2 > p1:
                return "✅ Higher Highs (Strong Uptrend)"
            if p3 < p2 < p1:
                return "⚠️ Lower Highs (Strong Downtrend)"

            return "Sideways / Choppy"
        except:
            return "Insufficient Data for Patterns"

class ScannerEngine:
    @staticmethod
    def safe_ai_request(prompt):
        """Restores the AI connection and handles rate limits."""
        for attempt in range(3):
            try:
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
    def analyze_stock_logic(ticker, mode):
        """The core logic that filters stocks based on the selected mode."""
        try:
            period = "1y" if mode in ["INTRADAY", "SHORT", "SWING"] else "1mo"
            
            # 🛠️ THE FIX: Use yf.Ticker().history() instead of yf.download()
            # This is 100% thread-safe and prevents data mixing!
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            if df is None or df.empty: 
                return None

            # Need at least 200 days to calculate 200 EMA
            if len(df) < 200 and mode in ["INTRADAY", "SHORT"]:
                return None

            # 🧹 No more messy MultiIndex column code needed! .history() is clean.
            curr_price = float(df['Close'].iloc[-1])
            open_p = float(df['Open'].iloc[-1])
            high_p = float(df['High'].iloc[-1])
            low_p = float(df['Low'].iloc[-1])
            vol = int(df['Volume'].iloc[-1])
            avg_vol = int(df['Volume'].tail(10).mean())

            if mode == "INTRADAY":
                ema_200 = ta.ema(df['Close'], length=200).iloc[-1]
                if (open_p <= low_p * 1.01) and (curr_price > ema_200) and (vol > avg_vol * 1.05):
                    return {"Ticker": ticker, "Price": round(curr_price, 2), "Type": "🔥 Intraday Buy"}
            
            elif mode == "BTST":
                prev_close = float(df['Close'].iloc[-2])
                if (curr_price > prev_close * 1.01) and (vol >= avg_vol * 0.9):
                    return {"Ticker": ticker, "Price": round(curr_price, 2), "Type": "🌙 BTST"}
            
            elif mode == "SHORT":
                ema_200 = ta.ema(df['Close'], length=200).iloc[-1]
                if (open_p >= high_p * 0.99) and (curr_price < ema_200) and (vol > avg_vol * 1.05):
                    return {"Ticker": ticker, "Price": round(curr_price, 2), "Type": "🩸 Short Sell"}

            elif mode == "SWING":
                ema_50 = ta.ema(df['Close'], length=50).iloc[-1]
                rsi = ta.rsi(df['Close'], length=14).iloc[-1]
                if (curr_price > ema_50) and (rsi > 50) and (vol > avg_vol):
                    return {"Ticker": ticker, "Price": round(curr_price, 2), "Type": "📈 Swing Trade"}

            return None
        except Exception as e:
            return None

    @staticmethod
    def scan_market(stock_list, mode="INTRADAY"):
        results = []
        progress_bar = st.progress(0, text=f"⚡ Scanning for {mode} setups...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_stock = {executor.submit(ScannerEngine.analyze_stock_logic, s, mode): s for s in stock_list}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_stock)):
                data = future.result()
                if data:
                    results.append(data)
                progress_bar.progress((i + 1) / len(stock_list))
        
        progress_bar.empty()
        return pd.DataFrame(results)
# --- LAYER 4: NIFTY OPTIONS ANALYZER (NEW) ---
def run_nifty_analysis():
    st.info("🦅 Analyzing Nifty 50 for Call/Put Levels...")
    try:
        df = yf.download("^NSEI", period="1mo", interval="15m", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        current_price = float(df['Close'].iloc[-1])
        highs = df['High'].tail(50).values
        lows = df['Low'].tail(50).values
        
        # Simple Support/Resistance based on recent extremes
        resistance = round(np.max(highs), 2)
        support = round(np.min(lows), 2)
        rsi = round(ta.rsi(df['Close'], length=14).iloc[-1], 2)

        prompt = f"""
        Act as an expert F&O derivatives analyst. 
        Nifty 50 Current Price: {current_price}
        Immediate Resistance: {resistance}
        Immediate Support: {support}
        15-Min RSI: {rsi}
        
        Provide a very quick, actionable view for today:
        1. When to buy a CALL option (Above what level? Target? Stop Loss?)
        2. When to buy a PUT option (Below what level? Target? Stop Loss?)
        3. Overall Market Sentiment right now.
        Format cleanly with emojis.
        """
        response = ScannerEngine.safe_ai_request(prompt)
        if response:
            st.success("✅ Nifty Call/Put Analysis Ready")
            st.markdown(response.text)
        else:
            st.error("AI Busy. Try again.")
    except Exception as e:
        st.error(f"Could not load Nifty data: {e}")

# --- LAYER 5: ADVANCED ANALYSIS ENGINE ---
def run_advanced_analysis(ticker):
    st.info(f"🦅 Ultra-Analyst is scanning {ticker}...")
    
    def safe_extract(series):
        try:
            if series is None or series.empty: return 0.0
            val = series.iloc[-1]
            if pd.isna(val): return 0.0
            return float(val.item()) if hasattr(val, 'item') else float(val)
        except:
            return 0.0

    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        
        if df is None or df.empty:
            st.error(f"❌ ERROR: No data found for '{ticker}'.")
            return

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, level=1, axis=1)
            except:
                pass 

        if len(df) < 200:
            st.error(f"⚠️ Not enough data ({len(df)} days). Need 200+ days for EMA.")
            return

        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['EMA_200'] = ta.ema(df['Close'], length=200)
        
        bb = ta.bbands(df['Close'], length=20)
        if bb is not None:
            df['BB_UPPER'] = bb[bb.columns[2]] # Fixed: Usually upper band is index 2
        else:
            df['BB_UPPER'] = df['Close']
        
        current_price = safe_extract(df['Close'])
        rsi = round(safe_extract(df['RSI']), 2)
        ema_200 = round(safe_extract(df['EMA_200']), 2)
        upper_band = round(safe_extract(df['BB_UPPER']), 2)
        
        chart_pattern = PatternEngine.detect_pattern(df)
        news = get_live_news(ticker)

    except Exception as e:
        st.error(f"Data Processing Error: {e}")
        return

    # Updated Prompt to force Entry, Stoploss, Target and timeframe calculations
    prompt = f"""
    Act as a Senior Technical Analyst. Analyze {ticker}.
    
    📊 DATA:
    - Pattern: {chart_pattern}
    - Price: {current_price}
    - RSI: {rsi}
    - 200 EMA: {ema_200}
    - Volatility Upper: {upper_band}
    
    📰 NEWS:
    {news}
    
    🔮 MISSION:
    1. Analyze the Technicals & News briefly.
    2. Suggest the best timeframe for this trade (e.g., Intraday, 1-week Swing, Long-term).
    3. MANDATORY: Give exact mathematical levels for:
       - 🟢 ENTRY PRICE
       - 🔴 STOP LOSS
       - 🎯 TARGET PRICE(S)
    4. Provide the exact risk-to-reward ratio based on your levels.
    """

    response = ScannerEngine.safe_ai_request(prompt)
    
    if response:
        st.success(f"✅ Deep Analysis for {ticker}")
        st.markdown(response.text)
    else:
        st.error("AI Busy. Try again.")

# --- LAYER 6: DASHBOARD ---
# --- LAYER 6: DASHBOARD ---
def render_dashboard():
    status, status_label, current_time = MarketTimer.get_status()
    
    # --- SIDEBAR: COMMODITIES ---
    st.sidebar.title("🦅 Prime Trade AI")
    st.sidebar.markdown(f"**Status:** {status_label}")
    st.sidebar.markdown(f"**Time:** {current_time.strftime('%H:%M:%S IST')}")
    st.sidebar.divider()
    
    st.sidebar.subheader("🪙 Commodities (Global/MCX Proxy)")
    macro = get_macro_data()
    
    if "Gold (Comex)" in macro:
        st.sidebar.metric("Gold", f"${macro['Gold (Comex)']['price']}", f"{macro['Gold (Comex)']['change']}%")
    if "Silver (Comex)" in macro:
        st.sidebar.metric("Silver", f"${macro['Silver (Comex)']['price']}", f"{macro['Silver (Comex)']['change']}%")
        
    st.sidebar.divider()
    st.sidebar.info("💡 Tip: Global commodity futures closely track Indian MCX pricing.")

    # --- MAIN SCREEN: GLOBAL INDICES & SENTIMENT ---
    st.title("Prime Trade AI | Ultra Terminal")
    
    # Top Ticker Tape
    c1, c2, c3 = st.columns(3)
    if "Nifty 50" in macro:
        c1.metric("🇮🇳 Nifty 50", f"{macro['Nifty 50']['price']}", f"{macro['Nifty 50']['change']}%")
    if "Dow Jones" in macro:
        c2.metric("🇺🇸 Dow Jones", f"{macro['Dow Jones']['price']}", f"{macro['Dow Jones']['change']}%")
    if "Nasdaq" in macro:
        c3.metric("🇺🇸 Nasdaq", f"{macro['Nasdaq']['price']}", f"{macro['Nasdaq']['change']}%")
        
    st.divider()
    
    # AI Market Condition & News
    st.subheader("📰 Live Market Condition")
    nifty_change = macro.get("Nifty 50", {}).get("change", 0.0)
    general_news = get_live_news("NIFTY 50") 
    
    with st.spinner("Analyzing global sentiment..."):
        market_condition = get_market_sentiment(nifty_change, general_news)
        st.info(market_condition)
        
    with st.expander("View Latest Market Headlines"):
        st.markdown(general_news)

    st.divider()

    # --- YOUR TABS (RESTORED!) ---
    tab1, tab2, tab3 = st.tabs(["🔥 Stock Scanners", "🔎 Deep Pattern Analyzer", "📊 Nifty Options (Call/Put)"])
    
    with tab1:
        st.markdown("### 🚦 Quick Market Scanners")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("⚡ Intraday Buy"):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, mode="INTRADAY")
                if not df.empty: st.dataframe(df)
                else: st.warning("No Intraday Buy setups.")
        with col2:
            if st.button("🩸 Short Sell"):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, mode="SHORT")
                if not df.empty: st.dataframe(df)
                else: st.warning("No Short Sell setups.")
        with col3:
            if st.button("🌙 BTST"):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, mode="BTST")
                if not df.empty: st.dataframe(df)
                else: st.warning("No BTST setups.")
        with col4:
            if st.button("📈 Swing Trades"):
                df = ScannerEngine.scan_market(LIQUID_STOCKS, mode="SWING")
                if not df.empty: st.dataframe(df)
                else: st.warning("No Swing setups.")
                
        st.info("💡 Tip: Take the tickers from these scanners and plug them into the **Deep Pattern Analyzer** tab to get your Entry, Target, and Stop Loss calculations!")

    with tab2:
        ticker = st.text_input("Enter Symbol for Pattern Analysis (e.g., ZOMATO.NS):").upper()
        if ticker: 
            if ticker.endswith(".NS"):
                run_advanced_analysis(ticker)
            else:
                st.warning("Please add .NS")
                
    with tab3:
        st.markdown("### 🦅 Nifty 50 Options Levels")
        st.write("Get strong support/resistance levels and Call/Put recommendations based on current market data.")
        if st.button("🔮 Analyze Nifty Levels"):
            run_nifty_analysis()

if __name__ == "__main__":
    render_dashboard()
