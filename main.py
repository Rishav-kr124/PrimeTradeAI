import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import google.generativeai as genai
import plotly.graph_objects as go
import feedparser  # <--- NEW: Added for News
import urllib.parse
from datetime import datetime, time
import pytz

# --- ARCHITECTURAL CONFIGURATION ---
st.set_page_config(
    page_title="Prime Trade AI | Pro Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "SBIN.NS",
    "BHARTIARTL.NS", "ITC.NS", "L&T.NS", "KOTAKBANK.NS", "AXISBANK.NS", "HCLTECH.NS",
    "ULTRACEMCO.NS", "ASIANPAINT.NS", "TITAN.NS", "MARUTI.NS", "TATASTEEL.NS", "BAJFINANCE.NS",
    "SUNPHARMA.NS", "M&M.NS" 
]

# --- LAYER 1: SECURITY & SETUP ---
def configure_ai():
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-flash-latest')
    except Exception as e:
        st.error("🚨 API Key Missing! Please set GEMINI_API_KEY in Streamlit Secrets.")
        st.stop()

model = configure_ai()

# --- LAYER 2: UTILITY SERVICES ---
class MarketTimer:
    """Determines if the Indian Market is currently Open or Closed."""
    @staticmethod
    def get_status():
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        market_open = time(9, 15)
        market_close = time(15, 30)
        
        is_weekday = now.weekday() < 5
        is_open_time = market_open <= now.time() <= market_close
        
        if is_weekday and is_open_time:
            return "OPEN", "🔴 LIVE MARKET", now
        else:
            return "CLOSED", "🌙 AFTER HOURS / PRE-MARKET", now

class DataService:
    """Handles Data Fetching (Prices + Global Cues)."""
    @staticmethod
    def get_global_cues():
        tickers = {"^GSPC": "S&P 500 (US)", "^IXIC": "NASDAQ (Tech)", "^NSEI": "NIFTY 50"}
        data = []
        for symbol, name in tickers.items():
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(period="2d")
                if len(hist) >= 2:
                    close = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2]
                    change = ((close - prev_close) / prev_close) * 100
                    data.append({"Index": name, "Price": close, "Change %": change})
            except:
                continue
        return pd.DataFrame(data)

    @staticmethod
    def get_stock_data(ticker, period="1y", interval="1d"):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)
            return df, stock.info
        except:
            return None, None

class NewsService:
    """NEW CLASS: Fetches Google News RSS for the stock."""
    @staticmethod
    def get_latest_news(ticker):
        # Remove .NS to search properly (e.g., "RELIANCE" instead of "RELIANCE.NS")
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
        encoded_ticker = urllib.parse.quote(clean_ticker)
        
        # Google News RSS URL for India (English)
        rss_url = f"https://news.google.com/rss/search?q={encoded_ticker}+stock+news&hl=en-IN&gl=IN&ceid=IN:en"
        
        try:
            feed = feedparser.parse(rss_url)
            headlines = []
            # Get top 5 headlines
            for entry in feed.entries[:5]:
                headlines.append(f"- {entry.title} (Source: {entry.source.title})")
            
            return "\n".join(headlines) if headlines else "No specific news found recently."
        except:
            return "Could not fetch news."

# --- LAYER 3: ANALYSIS ENGINE ---
class TechnicalAnalyst:
    @staticmethod
    def calculate_indicators(df):
        if df is None or len(df) < 50:
            return df
        
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'])
        df = pd.concat([df, macd], axis=1)
        
        return df.dropna().tail(1)

# --- LAYER 4: PRESENTATION & AI CORE ---
def render_dashboard():
    status, status_label, current_time = MarketTimer.get_status()
    
    st.title("🚀 Prime Trade AI")
    st.markdown(f"**Status:** {status_label} | **Time:** {current_time.strftime('%Y-%m-%d %H:%M:%S IST')}")
    
    # Global Cues
    st.subheader("🌍 Global Market Sentiment")
    global_df = DataService.get_global_cues()
    if not global_df.empty:
        cols = st.columns(len(global_df))
        for index, row in global_df.iterrows():
            cols[index].metric(label=row['Index'], value=f"{row['Price']:.2f}", delta=f"{row['Change %']:.2f}%")
    st.divider()

    # Inputs
    col1, col2 = st.columns([1, 2])
    with col1:
        st.info("🔎 **Search Any Stock**")
        search_input = st.text_input("Enter Symbol (e.g., ZOMATO.NS):").upper()
    with col2:
        st.success("📊 **Quick Select (Nifty 50)**")
        quick_select = st.selectbox("Or choose:", ["Select..."] + NIFTY_50_SYMBOLS)

    target_stock = search_input if search_input else (quick_select if quick_select != "Select..." else None)

    if target_stock:
        if not target_stock.endswith((".NS", ".BO")):
             st.warning("⚠️ Add .NS or .BO (e.g., TATASTEEL.NS)")
        else:
            run_analysis(target_stock, status)

def run_analysis(ticker, market_status):
    st.divider()
    st.header(f"📉 Deep Analysis: {ticker}")
    
    with st.spinner(f"🤖 AI is reading news & charts for {ticker}..."):
        # 1. Fetch Price Data
        df, info = DataService.get_stock_data(ticker)
        if df is None or df.empty:
            st.error("❌ Stock not found.")
            return

        # 2. Fetch News (NEW STEP)
        news_summary = NewsService.get_latest_news(ticker)
        
        # 3. Render Chart
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(height=500, title=f"{ticker} Price Action")
        st.plotly_chart(fig, use_container_width=True)

        # 4. Calculate Technicals
        tech_data = TechnicalAnalyst.calculate_indicators(df)
        last_price = df['Close'].iloc[-1]
        
        # 5. The "Super Prompt" (Math + News)
        prompt = f"""
        You are a Senior Financial Analyst.
        
        **MARKET CONTEXT:**
        - Status: {market_status}
        - Ticker: {ticker}
        - Current Price: {last_price}
        
        **LATEST NEWS HEADLINES (Sentiment Analysis):**
        {news_summary}
        
        **TECHNICAL INDICATORS:**
        {tech_data.to_string()}
        
        **TASK:**
        Combine the News Sentiment (Positive/Negative) with the Technical Indicators (RSI, SMA).
        If News is bad but Technicals are good, be cautious.
        
        **OUTPUT:**
        1. **Verdict:** (BUY / SELL / WAIT)
        2. **News Sentiment:** (Positive / Neutral / Negative) - Explain briefly based on headlines.
        3. **Trade Setup:** Entry, Stop Loss, Target.
        4. **Reasoning:** Combine news and math.
        """
        
        response = model.generate_content(prompt)
        st.markdown("### 🤖 Prime Trade Verdict")
        st.markdown(response.text)
        
        with st.expander("Show News Source"):
            st.write(news_summary)

if __name__ == "__main__":
    render_dashboard()