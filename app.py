import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import os
import json
import ta
from textblob import TextBlob
from newsapi import NewsApiClient
from fpdf import FPDF
import io
import base64



# Configure Streamlit page
st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache trending stocks data for 30 minutes
@st.cache_data(ttl=1800)
def get_trending_stocks():
    """Get trending stocks with their performance metrics"""
    # Comprehensive list of popular stocks to analyze for trending
    trending_candidates = [
        # Tech Giants
        'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL',
        # Electric Vehicles & Transport
        'TSLA', 'UBER', 'LYFT', 'NIO', 'RIVN', 'LCID',
        # Semiconductors & Hardware
        'AMD', 'INTC', 'QCOM', 'MU', 'AVGO', 'TSM', 'MRVL',
        # Finance & Fintech
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'PYPL', 'SQ', 'V', 'MA',
        # E-commerce & Retail
        'SHOP', 'BABA', 'JD', 'PDD', 'WMT', 'TGT', 'COST',
        # Entertainment & Media
        'DIS', 'ROKU', 'SPOT', 'TWTR', 'SNAP', 'PINS', 'ZM', 'DOCU',
        # Healthcare & Biotech
        'JNJ', 'PFE', 'MRNA', 'ABBV', 'UNH', 'TMO', 'DHR',
        # Energy & Utilities
        'XOM', 'CVX', 'NEE', 'DUK', 'SO',
        # Consumer Goods
        'KO', 'PEP', 'NKE', 'SBUX', 'MCD', 'PG',
        # Industrial & Aerospace
        'BA', 'CAT', 'GE', 'MMM', 'HON',
        # REITs & Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX',
        # Crypto-related
        'COIN', 'MSTR', 'SQ'
    ]
    
    trending_data = []
    total_stocks = min(len(trending_candidates), 50)  # Analyze up to 50 stocks
    
    with st.spinner("Analyzing trending stocks..."):
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(trending_candidates[:total_stocks]):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1mo")  # Get 1 month data for quick analysis
                info = stock.info
                
                if not hist.empty and len(hist) > 5:
                    # Calculate 1-month performance
                    first_price = hist['Close'].iloc[0]
                    last_price = hist['Close'].iloc[-1]
                    monthly_return = ((last_price - first_price) / first_price) * 100
                    
                    # Calculate 1-week performance
                    week_price = hist['Close'].iloc[-5] if len(hist) >= 5 else first_price
                    weekly_return = ((last_price - week_price) / week_price) * 100
                    
                    trending_data.append({
                        'ticker': ticker,
                        'name': info.get('longName', ticker),
                        'current_price': last_price,
                        'monthly_return': monthly_return,
                        'weekly_return': weekly_return,
                        'market_cap': info.get('marketCap', 0),
                        'sector': info.get('sector', 'N/A')
                    })
                
                progress_bar.progress((i + 1) / total_stocks)
                
            except Exception as e:
                continue  # Skip stocks that fail to load
        
        progress_bar.empty()
    
    # Sort by monthly return (descending)
    trending_data.sort(key=lambda x: x['monthly_return'], reverse=True)
    return trending_data  # Return all analyzed stocks

@st.cache_data(ttl=300)
def get_quick_stock_metrics(ticker):
    """Get quick metrics for a single stock"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info
        
        if hist.empty:
            return None
        
        # Calculate key metrics
        current_price = hist['Close'].iloc[-1]
        year_start_price = hist['Close'].iloc[0]
        ytd_return = ((current_price - year_start_price) / year_start_price) * 100
        
        return {
            'current_price': current_price,
            'ytd_return': ytd_return,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0)
        }
    except:
        return None

# Alpha Vantage API functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_stocks_alphavantage(query):
    """Search for stocks using Alpha Vantage API"""
    if len(query) < 2:
        return []
    
    try:
        api_key = "Z2FC2BEC2ILA7130"
        if not api_key:
            st.error("Alpha Vantage API key not found")
            return []
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': 'SYMBOL_SEARCH',
            'keywords': query,
            'apikey': api_key
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'bestMatches' in data:
            results = []
            for match in data['bestMatches'][:10]:  # Limit to top 10 results
                symbol = match.get('1. symbol', '')
                name = match.get('2. name', '')
                region = match.get('4. region', '')
                market_open = match.get('5. marketOpen', '')
                market_close = match.get('6. marketClose', '')
                
                # Filter out closed markets for better results
                if market_open and market_close:
                    results.append({
                        'symbol': symbol,
                        'name': name,
                        'region': region,
                        'display': f"{symbol} - {name}",
                        'market_open': market_open,
                        'market_close': market_close
                    })
            return results
        else:
            return []
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error while searching stocks: {str(e)}")
        return []
    except json.JSONDecodeError as e:
        st.error("Error parsing stock search results")
        return []
    except Exception as e:
        st.error(f"Error searching stocks: {str(e)}")
        return []

def convert_ticker_for_yfinance(ticker):
    """Convert Alpha Vantage ticker format to yfinance compatible format"""
    if not ticker:
        return ticker
    
    ticker = ticker.upper()
    
    # Handle Indian stock exchanges
    if ticker.endswith('.BSE'):
        # Convert BSE format to Bombay Stock Exchange format for yfinance
        base_symbol = ticker.replace('.BSE', '')
        return f"{base_symbol}.BO"
    elif ticker.endswith('.NS'):
        # NSE format is already compatible with yfinance
        return ticker
    elif ticker.endswith('.NSE'):
        # Convert .NSE to .NS for yfinance
        base_symbol = ticker.replace('.NSE', '')
        return f"{base_symbol}.NS"
    
    # Handle other common exchange formats
    elif ticker.endswith('.L'):
        # London Stock Exchange - keep as is
        return ticker
    elif ticker.endswith('.TO'):
        # Toronto Stock Exchange - keep as is  
        return ticker
    elif ticker.endswith('.HK'):
        # Hong Kong Stock Exchange - keep as is
        return ticker
    elif ticker.endswith('.SS') or ticker.endswith('.SZ'):
        # Shanghai and Shenzhen Stock Exchanges - keep as is
        return ticker
    
    # For US stocks and others, return as is
    return ticker

def validate_and_suggest_ticker(original_ticker):
    """Try different ticker formats if the original fails"""
    ticker = convert_ticker_for_yfinance(original_ticker)
    
    # Try the converted ticker first
    try:
        test_stock = yf.Ticker(ticker)
        test_data = test_stock.history(period="5d")  # Quick test
        if not test_data.empty:
            return ticker
    except:
        pass
    
    # If original ticker has exchange suffix, try alternatives
    if '.' in original_ticker:
        base_symbol = original_ticker.split('.')[0]
        
        # For Indian stocks, try both BSE and NSE formats
        if original_ticker.endswith('.BSE') or original_ticker.endswith('.NSE'):
            alternatives = [f"{base_symbol}.NS", f"{base_symbol}.BO"]
            for alt_ticker in alternatives:
                try:
                    test_stock = yf.Ticker(alt_ticker)
                    test_data = test_stock.history(period="5d")
                    if not test_data.empty:
                        return alt_ticker
                except:
                    continue
        
        # Try without exchange suffix for US stocks
        try:
            test_stock = yf.Ticker(base_symbol)
            test_data = test_stock.history(period="5d")
            if not test_data.empty:
                return base_symbol
        except:
            pass
    
    # Return original ticker if all alternatives fail
    return ticker

def create_stock_search_interface():
    """Create an interactive stock search interface with autocomplete"""
    st.sidebar.header("🔍 Stock Search")
    
    # Initialize session state for search
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_from_search' not in st.session_state:
        st.session_state.selected_from_search = None
    
    # Search input
    search_query = st.sidebar.text_input(
        "Search for stocks",
        value=st.session_state.search_query,
        placeholder="Type at least 2 characters...",
        help="Search for stocks by symbol or company name",
        key="stock_search_input"
    )
    
    # Update search results when query changes
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
        if len(search_query) >= 2:
            with st.sidebar:
                with st.spinner("Searching stocks..."):
                    st.session_state.search_results = search_stocks_alphavantage(search_query)
        else:
            st.session_state.search_results = []
    
    # Display search results
    if st.session_state.search_results:
        st.sidebar.subheader("Search Results")
        
        for result in st.session_state.search_results:
            col1, col2 = st.sidebar.columns([3, 1])
            
            with col1:
                original_symbol = result['symbol']
                converted_symbol = convert_ticker_for_yfinance(original_symbol)
                
                # Show both symbols if they're different
                if original_symbol != converted_symbol:
                    st.write(f"**{original_symbol}** → **{converted_symbol}**")
                else:
                    st.write(f"**{original_symbol}**")
                
                st.caption(result['name'][:40] + "..." if len(result['name']) > 40 else result['name'])
                
                # Show exchange info for clarity
                if result.get('region'):
                    st.caption(f"📍 {result['region']}")
            
            with col2:
                if st.button("Select", key=f"select_{result['symbol']}", use_container_width=True):
                    # Convert ticker format for yfinance compatibility
                    converted_ticker = convert_ticker_for_yfinance(result['symbol'])
                    st.session_state.selected_from_search = converted_ticker
                    st.session_state.search_query = ""  # Clear search
                    st.session_state.search_results = []  # Clear results
                    st.rerun()
    
    elif len(search_query) >= 2:
        st.sidebar.info("No results found. Try a different search term.")
    elif len(search_query) > 0:
        st.sidebar.info("Type at least 2 characters to search")
    
    return st.session_state.selected_from_search

# Cache data for 5 minutes to improve performance
@st.cache_data(ttl=300)
def fetch_stock_data(ticker, period="5y"):
    """Fetch stock data from Yahoo Finance with improved ticker handling"""
    if not ticker:
        return None, None, None, None, None
    
    # First, validate and convert the ticker format
    validated_ticker = validate_and_suggest_ticker(ticker)
    
    try:
        stock = yf.Ticker(validated_ticker)
        
        # Get historical data
        hist = stock.history(period=period)
        
        # Check if we got valid data
        if hist.empty:
            # If empty, show a more helpful error message
            if ticker != validated_ticker:
                st.error(f"No data found for {ticker}. Tried alternative format {validated_ticker} but no data available.")
            else:
                st.error(f"No data found for {ticker}. Please check the ticker symbol or try a different time period.")
            return None, None, None, None, None
        
        # Get stock info
        info = stock.info
        
        # Get financial data
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        
        # Show ticker conversion info if it was changed
        if ticker != validated_ticker:
            st.info(f"📝 Converted ticker from '{ticker}' to '{validated_ticker}' for compatibility")
        
        return hist, info, financials, balance_sheet, cashflow
        
    except Exception as e:
        # Try one more fallback for Indian stocks
        if ticker != validated_ticker:
            st.error(f"Error fetching data for {ticker} (tried {validated_ticker}): {str(e)}")
        else:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
        
        # If it's an Indian stock, provide helpful suggestions
        if '.BSE' in ticker or '.NSE' in ticker or '.BO' in ticker or '.NS' in ticker:
            st.info("💡 Tip: For Indian stocks, try searching with just the company name (e.g., 'Reliance' instead of 'RELIANCE.BSE')")
        
        return None, None, None, None, None

def calculate_technical_indicators(hist_data):
    """Calculate technical indicators for the stock data"""
    if hist_data is None or hist_data.empty:
        return None
    
    try:
        data = hist_data.copy()
        
        # Calculate RSI (14-day)
        data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()
        
        # Calculate MACD
        macd = ta.trend.MACD(close=data['Close'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_histogram'] = macd.macd_diff()
        
        # Calculate Moving Averages
        data['MA_50'] = ta.trend.SMAIndicator(close=data['Close'], window=50).sma_indicator()
        data['MA_200'] = ta.trend.SMAIndicator(close=data['Close'], window=200).sma_indicator()
        
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=data['Close'])
        data['BB_upper'] = bollinger.bollinger_hband()
        data['BB_lower'] = bollinger.bollinger_lband()
        data['BB_middle'] = bollinger.bollinger_mavg()
        
        return data
        
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}")
        return None

def find_support_resistance_levels(hist_data, window=20):
    """Find support and resistance levels"""
    if hist_data is None or hist_data.empty:
        return [], []
    
    try:
        highs = hist_data['High'].rolling(window=window, center=True).max()
        lows = hist_data['Low'].rolling(window=window, center=True).min()
        
        # Find resistance levels (local maxima)
        resistance_levels = []
        for i in range(window, len(hist_data) - window):
            if hist_data['High'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(hist_data['High'].iloc[i])
        
        # Find support levels (local minima)  
        support_levels = []
        for i in range(window, len(hist_data) - window):
            if hist_data['Low'].iloc[i] == lows.iloc[i]:
                support_levels.append(hist_data['Low'].iloc[i])
        
        # Get most significant levels (remove duplicates and sort)
        resistance_levels = sorted(list(set([round(level, 2) for level in resistance_levels[-5:]])))
        support_levels = sorted(list(set([round(level, 2) for level in support_levels[-5:]])))
        
        return support_levels, resistance_levels
        
    except Exception as e:
        st.error(f"Error finding support/resistance levels: {str(e)}")
        return [], []

def create_advanced_price_chart(data_with_indicators, ticker, support_levels, resistance_levels):
    """Create advanced price chart with technical indicators and support/resistance"""
    if data_with_indicators is None or data_with_indicators.empty:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            f'{ticker} - Price with Technical Indicators',
            'RSI (14)',
            'MACD',
            'Volume'
        ),
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.2, 0.15]
    )
    
    # Main price chart with candlesticks
    fig.add_trace(
        go.Candlestick(
            x=data_with_indicators.index,
            open=data_with_indicators['Open'],
            high=data_with_indicators['High'],
            low=data_with_indicators['Low'],
            close=data_with_indicators['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['MA_50'],
            mode='lines',
            name='MA 50',
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['MA_200'],
            mode='lines',
            name='MA 200',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['BB_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', dash='dash'),
            opacity=0.5
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['BB_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', dash='dash'),
            fill='tonexty',
            opacity=0.3
        ),
        row=1, col=1
    )
    
    # Add support and resistance levels
    for level in resistance_levels:
        fig.add_hline(y=level, line_dash="dot", line_color="red", 
                     annotation_text=f"R: ${level:.2f}", row=1, col=1)
    
    for level in support_levels:
        fig.add_hline(y=level, line_dash="dot", line_color="green", 
                     annotation_text=f"S: ${level:.2f}", row=1, col=1)
    
    # RSI chart
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add RSI overbought/oversold levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD chart
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data_with_indicators.index,
            y=data_with_indicators['MACD_signal'],
            mode='lines',
            name='Signal',
            line=dict(color='red'),
            showlegend=False
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=data_with_indicators.index,
            y=data_with_indicators['MACD_histogram'],
            name='Histogram',
            marker_color='gray',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=data_with_indicators.index,
            y=data_with_indicators['Volume'],
            name='Volume',
            marker_color='lightblue',
            showlegend=False
        ),
        row=4, col=1
    )
    
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(x=0, y=1, traceorder='normal')
    )
    
    return fig

def get_stock_news_and_sentiment(ticker):
    """Fetch news articles and analyze sentiment"""
    try:
        newsapi_key = os.getenv('NEWSAPI_KEY')
        if not newsapi_key:
            return [], "Neutral"
        
        newsapi = NewsApiClient(api_key="5d26082d96ea4d8d8cb545fe6bbdf6e1")

        
        # Get company name from ticker for better search
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # Search for news articles
        articles = newsapi.get_everything(
            q=f"{ticker} OR {company_name}",
            language='en',
            sort_by='publishedAt',
            page_size=5
        )
        
        news_articles = []
        sentiment_scores = []
        
        if articles['articles']:
            for article in articles['articles']:
                # Clean and prepare article data
                title = article.get('title', 'No Title')
                description = article.get('description', 'No Description')
                url = article.get('url', '')
                published_at = article.get('publishedAt', '')
                source = article.get('source', {}).get('name', 'Unknown')
                
                news_articles.append({
                    'title': title,
                    'description': description,
                    'url': url,
                    'published_at': published_at,
                    'source': source
                })
                
                # Analyze sentiment of title and description
                text_to_analyze = f"{title} {description}"
                blob = TextBlob(text_to_analyze)
                sentiment_scores.append(blob.sentiment.polarity)
        
        # Calculate overall sentiment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment > 0.1:
                overall_sentiment = "Positive"
            elif avg_sentiment < -0.1:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"
        else:
            overall_sentiment = "Neutral"
        
        return news_articles, overall_sentiment
        
    except Exception as e:
        st.error(f"Error fetching news: {str(e)}")
        return [], "Neutral"

def calculate_yearly_returns(ticker, years=10):
    """Calculate year-wise returns for the stock"""
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch historical data for the specified number of years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        hist_data = stock.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            return None
        
        # Group data by year and calculate annual returns
        yearly_data = []
        
        # Extract year from index and add as column
        hist_data = hist_data.copy()
        hist_data['Year'] = pd.DatetimeIndex(hist_data.index).year
        
        years_available = sorted(hist_data['Year'].unique())
        
        for year in years_available:
            year_data = hist_data[hist_data['Year'] == year]
            
            if len(year_data) == 0:
                continue
                
            # Get first and last trading day of the year
            start_price = year_data['Close'].iloc[0]
            end_price = year_data['Close'].iloc[-1]
            
            # Calculate annual return
            annual_return = ((end_price - start_price) / start_price) * 100
            
            yearly_data.append({
                'year': year,
                'return': annual_return,
                'start_price': start_price,
                'end_price': end_price
            })
        
        return yearly_data
        
    except Exception as e:
        st.error(f"Error calculating yearly returns: {str(e)}")
        return None

def create_yearly_returns_chart(yearly_data, ticker):
    """Create a bar chart showing year-wise returns"""
    if not yearly_data:
        return None
    
    years = [data['year'] for data in yearly_data]
    returns = [data['return'] for data in yearly_data]
    
    # Create colors based on positive/negative returns
    colors = ['green' if ret >= 0 else 'red' for ret in returns]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=years,
        y=returns,
        marker_color=colors,
        name='Annual Returns',
        text=[f'{ret:+.1f}%' for ret in returns],
        textposition='outside',
        textfont=dict(size=12),
        hovertemplate='<b>%{x}</b><br>' +
                      'Annual Return: %{y:.1f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add horizontal line at 0%
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} - Annual Returns by Year',
        xaxis_title='Year',
        yaxis_title='Annual Return (%)',
        height=500,
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            gridcolor='lightgray',
            gridwidth=0.5
        ),
        yaxis=dict(
            gridcolor='lightgray',
            gridwidth=0.5,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=2
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def calculate_growth_metrics(hist_data):
    """Calculate growth metrics from historical data"""
    if hist_data is None or hist_data.empty:
        return {}
    
    try:
        # Get first and last prices
        first_price = hist_data['Close'].iloc[0]
        last_price = hist_data['Close'].iloc[-1]
        
        # Calculate total return
        total_return = ((last_price - first_price) / first_price) * 100
        
        # Calculate annualized return (assuming 5 years)
        years = len(hist_data) / 252  # Approximate trading days per year
        annualized_return = ((last_price / first_price) ** (1/years) - 1) * 100
        
        # Calculate volatility
        daily_returns = hist_data['Close'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Calculate max drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'current_price': last_price,
            'start_price': first_price
        }
    except Exception as e:
        st.error(f"Error calculating growth metrics: {str(e)}")
        return {}

def create_price_chart(hist_data, ticker):
    """Create interactive price chart"""
    if hist_data is None or hist_data.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{ticker} Stock Price', 'Volume'),
        vertical_spacing=0.05,
        row_width=[0.7, 0.3]
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name='Volume',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{ticker} - 5 Year Price History',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_returns_chart(hist_data, ticker):
    """Create returns analysis chart"""
    if hist_data is None or hist_data.empty:
        return None
    
    daily_returns = hist_data['Close'].pct_change().dropna() * 100
    cumulative_returns = ((1 + hist_data['Close'].pct_change()).cumprod() - 1) * 100
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Cumulative Returns (%)',
            'Daily Returns Distribution',
            'Rolling 30-Day Volatility (%)',
            'Monthly Returns Heatmap'
        ),
        specs=[[{"colspan": 2}, None],
               [{"type": "xy"}, {"type": "xy"}]]
    )
    
    # Cumulative returns
    fig.add_trace(
        go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    
    # Returns distribution
    fig.add_trace(
        go.Histogram(
            x=daily_returns,
            name='Daily Returns',
            nbinsx=50,
            marker_color='blue',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Rolling volatility
    rolling_vol = daily_returns.rolling(30).std() * np.sqrt(252)
    fig.add_trace(
        go.Scatter(
            x=rolling_vol.index,
            y=rolling_vol,
            mode='lines',
            name='30-Day Volatility',
            line=dict(color='red')
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True)
    return fig

def display_key_metrics(info, growth_metrics):
    """Display key financial metrics"""
    if not info:
        st.warning("Financial information not available")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Market Cap",
            f"${info.get('marketCap', 0):,.0f}" if info.get('marketCap') else "N/A"
        )
        st.metric(
            "P/E Ratio",
            f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else "N/A"
        )
    
    with col2:
        st.metric(
            "Dividend Yield",
            f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A"
        )
        st.metric(
            "Beta",
            f"{info.get('beta', 0):.2f}" if info.get('beta') else "N/A"
        )
    
    with col3:
        if growth_metrics:
            st.metric(
                "5-Year Return",
                f"{growth_metrics.get('total_return', 0):.2f}%",
                f"{growth_metrics.get('total_return', 0):.2f}%"
            )
            st.metric(
                "Annualized Return",
                f"{growth_metrics.get('annualized_return', 0):.2f}%"
            )
    
    with col4:
        if growth_metrics:
            st.metric(
                "Volatility",
                f"{growth_metrics.get('volatility', 0):.2f}%"
            )
            st.metric(
                "Max Drawdown",
                f"{growth_metrics.get('max_drawdown', 0):.2f}%"
            )

def display_trending_stocks(trending_data):
    """Display trending stocks in an attractive layout"""
    if not trending_data:
        st.warning("Unable to load trending stocks data")
        return
    
    st.subheader("📈 Stock Market Overview - Real-time Performance")
    
    # Add filter options
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        view_option = st.selectbox(
            "View",
            ["Top Gainers", "Top Losers", "All Stocks"],
            key="view_filter"
        )
    
    with col2:
        num_stocks = st.selectbox(
            "Show",
            [20, 30, 40, "All"],
            key="num_stocks_filter"
        )
    
    with col3:
        sector_filter = st.selectbox(
            "Sector Filter",
            ["All Sectors"] + sorted(list(set([stock['sector'] for stock in trending_data if stock['sector'] != 'N/A']))),
            key="sector_filter"
        )
    
    # Filter data based on selections
    filtered_data = trending_data.copy()
    
    # Apply sector filter
    if sector_filter != "All Sectors":
        filtered_data = [stock for stock in filtered_data if stock['sector'] == sector_filter]
    
    # Apply view filter
    if view_option == "Top Gainers":
        filtered_data = [stock for stock in filtered_data if stock['monthly_return'] > 0]
    elif view_option == "Top Losers":
        filtered_data = [stock for stock in filtered_data if stock['monthly_return'] < 0]
        filtered_data.sort(key=lambda x: x['monthly_return'])  # Sort ascending for losers
    
    # Apply quantity filter
    if num_stocks != "All":
        filtered_data = filtered_data[:num_stocks]
    
    # Display summary stats
    if filtered_data:
        total_gainers = sum(1 for stock in filtered_data if stock['monthly_return'] > 0)
        total_losers = len(filtered_data) - total_gainers
        avg_return = sum(stock['monthly_return'] for stock in filtered_data) / len(filtered_data)
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric("Total Stocks", len(filtered_data))
        with metrics_col2:
            st.metric("Gainers", total_gainers, delta=f"{(total_gainers/len(filtered_data)*100):.0f}%")
        with metrics_col3:
            st.metric("Losers", total_losers, delta=f"-{(total_losers/len(filtered_data)*100):.0f}%")
        with metrics_col4:
            st.metric("Avg Return", f"{avg_return:.1f}%", delta=f"{avg_return:.1f}%")
    
    # Create columns for trending stocks (5 columns for more compact display)
    if filtered_data:
        cols = st.columns(5)
        
        for i, stock_data in enumerate(filtered_data):
            col_idx = i % 5
            
            with cols[col_idx]:
                # Create a card-like container
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background-color: #f8f9fa; 
                        padding: 12px; 
                        border-radius: 8px; 
                        border-left: 4px solid {'#28a745' if stock_data['monthly_return'] > 0 else '#dc3545'};
                        margin-bottom: 10px;
                        height: 140px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    ">
                        <div>
                            <h5 style="margin: 0 0 3px 0; font-size: 16px;">{stock_data['ticker']}</h5>
                            <p style="margin: 0 0 3px 0; font-size: 10px; color: #666;">
                                {stock_data['name'][:20]}{'...' if len(stock_data['name']) > 20 else ''}
                            </p>
                            <p style="margin: 0 0 3px 0; font-size: 12px;">
                                <strong>${stock_data['current_price']:.2f}</strong>
                            </p>
                        </div>
                        <div>
                            <p style="margin: 0 0 2px 0; font-size: 11px; color: {'green' if stock_data['monthly_return'] > 0 else 'red'};">
                                Month: {stock_data['monthly_return']:+.1f}%
                            </p>
                            <p style="margin: 0 0 5px 0; font-size: 11px; color: {'green' if stock_data['weekly_return'] > 0 else 'red'};">
                                Week: {stock_data['weekly_return']:+.1f}%
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add clickable button to analyze this stock
                    if st.button(f"Analyze", key=f"btn_{stock_data['ticker']}", use_container_width=True):
                        st.session_state.selected_ticker = stock_data['ticker']
                        st.rerun()
    else:
        st.info("No stocks match your current filters. Try adjusting the filters above.")

def get_enhanced_financials(ticker):
    """Get enhanced financial data with 5-year history"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial statements
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        info = stock.info
        
        # Prepare 5-year financial metrics
        financial_metrics = {}
        
        if not financials.empty:
            # Revenue (Total Revenue or Total Operating Revenues)
            revenue_keys = ['Total Revenue', 'Operating Revenue', 'Total Operating Revenues']
            for key in revenue_keys:
                if key in financials.index:
                    financial_metrics['Revenue'] = financials.loc[key].head(5)
                    break
            
            # Net Income
            if 'Net Income' in financials.index:
                financial_metrics['Net Income'] = financials.loc['Net Income'].head(5)
            
            # Operating Income  
            if 'Operating Income' in financials.index:
                financial_metrics['Operating Income'] = financials.loc['Operating Income'].head(5)
        
        if not balance_sheet.empty:
            # Total Debt
            debt_keys = ['Total Debt', 'Long Term Debt', 'Short Long Term Debt']
            for key in debt_keys:
                if key in balance_sheet.index:
                    financial_metrics['Total Debt'] = balance_sheet.loc[key].head(5)
                    break
            
            # Total Equity
            equity_keys = ['Stockholder Equity', 'Total Stockholder Equity', 'Total Equity Gross Minority Interest']
            for key in equity_keys:
                if key in balance_sheet.index:
                    financial_metrics['Total Equity'] = balance_sheet.loc[key].head(5)
                    break
        
        # Calculate ratios and per-share metrics
        shares_outstanding = info.get('sharesOutstanding', 1)
        
        enhanced_metrics = {}
        for metric, data in financial_metrics.items():
            if not data.empty:
                enhanced_metrics[metric] = data
                
                # Calculate per-share metrics for relevant items
                if metric in ['Revenue', 'Net Income'] and shares_outstanding:
                    per_share_data = data / shares_outstanding
                    enhanced_metrics[f'{metric} per Share'] = per_share_data
        
        return enhanced_metrics, info
        
    except Exception as e:
        st.error(f"Error getting enhanced financials: {str(e)}")
        return {}, {}

def create_financial_charts(financial_metrics):
    """Create mini bar charts for financial metrics"""
    charts = {}
    
    for metric, data in financial_metrics.items():
        if len(data) > 1:  # Need at least 2 data points
            years = [str(year.year) for year in data.index]
            values = data.values / 1e9  # Convert to billions
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=years,
                y=values,
                name=metric,
                marker_color='lightblue' if all(v >= 0 for v in values) else ['red' if v < 0 else 'lightblue' for v in values]
            ))
            
            fig.update_layout(
                title=f"{metric} (Billions)",
                height=300,
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            charts[metric] = fig
    
    return charts

def get_peer_comparison(ticker, info):
    """Get peer comparison data"""
    try:
        sector = info.get('sector', '')
        if not sector:
            return []
        
        # Common competitors by sector (simplified mapping)
        sector_peers = {
            'Technology': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
            'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'T'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON']
        }
        
        peers = sector_peers.get(sector, [])
        if ticker in peers:
            peers.remove(ticker)  # Remove the current stock
        
        peer_data = []
        for peer_ticker in peers[:3]:  # Get top 3 peers
            try:
                peer_stock = yf.Ticker(peer_ticker)
                peer_info = peer_stock.info
                peer_hist = peer_stock.history(period="1y")
                
                if not peer_hist.empty:
                    # Calculate YTD return
                    ytd_return = ((peer_hist['Close'].iloc[-1] - peer_hist['Close'].iloc[0]) / peer_hist['Close'].iloc[0]) * 100
                    
                    peer_data.append({
                        'ticker': peer_ticker,
                        'name': peer_info.get('longName', peer_ticker),
                        'current_price': peer_hist['Close'].iloc[-1],
                        'ytd_return': ytd_return,
                        'market_cap': peer_info.get('marketCap', 0),
                        'pe_ratio': peer_info.get('trailingPE', 0)
                    })
            except:
                continue
        
        return peer_data
        
    except Exception as e:
        st.error(f"Error getting peer comparison: {str(e)}")
        return []

def initialize_watchlist():
    """Initialize watchlist in session state"""
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []

def add_to_watchlist(ticker):
    """Add stock to watchlist"""
    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)
        st.success(f"Added {ticker} to watchlist!")
    else:
        st.info(f"{ticker} is already in your watchlist.")

def remove_from_watchlist(ticker):
    """Remove stock from watchlist"""
    if ticker in st.session_state.watchlist:
        st.session_state.watchlist.remove(ticker)
        st.success(f"Removed {ticker} from watchlist!")

def display_watchlist():
    """Display watchlist with quick metrics"""
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add some stocks to get started!")
        return
    
    st.subheader("📋 Your Watchlist")
    
    cols = st.columns(min(len(st.session_state.watchlist), 4))
    
    for i, ticker in enumerate(st.session_state.watchlist):
        col_idx = i % 4
        
        with cols[col_idx]:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d", interval="1d")
                
                if not hist.empty:
                    current_price = hist['Close'][-1]
                    
                    st.metric(
                        ticker,
                        f"${current_price:.2f}",
                        # delta=f"{daily_change:+.1f}%"
                    )
                    
                    if st.button(f"Remove {ticker}", key=f"remove_{ticker}"):
                        remove_from_watchlist(ticker)
                        st.rerun()
            except:
                st.error(f"Error loading {ticker}")

def initialize_paper_trading():
    """Initialize paper trading in session state"""
    if 'paper_trades' not in st.session_state:
        st.session_state.paper_trades = []
    if 'paper_balance' not in st.session_state:
        st.session_state.paper_balance = 100000  # Start with $100,000

def execute_paper_trade(ticker, action, quantity, price):
    """Execute a paper trade"""
    trade_value = quantity * price
    
    if action == "BUY":
        if trade_value <= st.session_state.paper_balance:
            st.session_state.paper_balance -= trade_value
            st.session_state.paper_trades.append({
                'ticker': ticker,
                'action': action,
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'timestamp': datetime.now()
            })
            st.success(f"✅ Bought {quantity} shares of {ticker} at ${price:.2f}")
        else:
            st.error("❌ Insufficient balance for this trade")
    
    elif action == "SELL":
        # Check if we own enough shares
        owned_shares = sum(trade['quantity'] for trade in st.session_state.paper_trades 
                          if trade['ticker'] == ticker and trade['action'] == 'BUY') - \
                      sum(trade['quantity'] for trade in st.session_state.paper_trades 
                          if trade['ticker'] == ticker and trade['action'] == 'SELL')
        
        if quantity <= owned_shares:
            st.session_state.paper_balance += trade_value
            st.session_state.paper_trades.append({
                'ticker': ticker,
                'action': action,
                'quantity': quantity,
                'price': price,
                'value': trade_value,
                'timestamp': datetime.now()
            })
            st.success(f"✅ Sold {quantity} shares of {ticker} at ${price:.2f}")
        else:
            st.error("❌ You don't own enough shares to sell")

def display_paper_trading_portfolio():
    """Display paper trading portfolio"""
    if not st.session_state.paper_trades:
        st.info("No trades executed yet. Start by making your first paper trade!")
        return
    
    st.subheader("💼 Paper Trading Portfolio")
    
    # Calculate current positions
    positions = {}
    for trade in st.session_state.paper_trades:
        ticker = trade['ticker']
        if ticker not in positions:
            positions[ticker] = {'shares': 0, 'avg_price': 0, 'total_cost': 0}
        
        if trade['action'] == 'BUY':
            positions[ticker]['shares'] += trade['quantity']
            positions[ticker]['total_cost'] += trade['value']
        else:  # SELL
            positions[ticker]['shares'] -= trade['quantity']
            positions[ticker]['total_cost'] -= trade['value']
        
        if positions[ticker]['shares'] > 0:
            positions[ticker]['avg_price'] = positions[ticker]['total_cost'] / positions[ticker]['shares']
    
    # Display portfolio summary
    col1, col2, col3 = st.columns(3)
    
    total_portfolio_value = st.session_state.paper_balance
    total_pnl = 0
    
    for ticker, position in positions.items():
        if position['shares'] > 0:
            try:
                stock = yf.Ticker(ticker)
                current_price = stock.history(period="1d")['Close'][-1]
                current_value = position['shares'] * current_price
                total_portfolio_value += current_value
                
                pnl = current_value - position['total_cost']
                total_pnl += pnl
            except:
                pass
    
    with col1:
        st.metric("Cash Balance", f"${st.session_state.paper_balance:,.2f}")
    with col2:
        st.metric("Total Portfolio Value", f"${total_portfolio_value:,.2f}")
    with col3:
        st.metric("Total P&L", f"${total_pnl:,.2f}", f"{total_pnl:+,.2f}")
    
    # Display individual positions
    if positions:
        st.subheader("Current Positions")
        for ticker, position in positions.items():
            if position['shares'] > 0:
                try:
                    stock = yf.Ticker(ticker)
                    current_price = stock.history(period="1d")['Close'][-1]
                    current_value = position['shares'] * current_price
                    pnl = current_value - position['total_cost']
                    pnl_pct = (pnl / position['total_cost']) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**{ticker}**")
                        st.write(f"{position['shares']} shares")
                    with col2:
                        st.write(f"Avg: ${position['avg_price']:.2f}")
                        st.write(f"Current: ${current_price:.2f}")
                    with col3:
                        st.write(f"Value: ${current_value:,.2f}")
                    with col4:
                        color = "green" if pnl >= 0 else "red"
                        st.markdown(f"<span style='color: {color}'>P&L: ${pnl:+,.2f}</span>", 
                                  unsafe_allow_html=True)
                        st.markdown(f"<span style='color: {color}'>{pnl_pct:+.1f}%</span>", 
                                  unsafe_allow_html=True)
                except:
                    st.error(f"Error loading data for {ticker}")

def create_pdf_export(ticker, charts_data, financial_data):
    """Create PDF export of stock analysis"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        
        # Title
        pdf.cell(200, 10, txt=f"Stock Analysis Report - {ticker}", ln=True, align="C")
        pdf.ln(10)
        
        # Add timestamp
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ln=True, align="C")
        pdf.ln(10)
        
        # Add financial summary
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Financial Summary", ln=True)
        pdf.ln(5)
        
        # Note: This is a basic PDF. In a full implementation, you would add charts and more data
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="This is a sample PDF export. Charts and detailed data would be included in a full implementation.", 
                ln=True)
        
        # Return PDF as bytes
        pdf_output = io.BytesIO()
        pdf_string = pdf.output(dest='S').encode('latin1')
        pdf_output.write(pdf_string)
        pdf_output.seek(0)
        
        return pdf_output.getvalue()
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def display_financial_statements(financials, balance_sheet, cashflow):
    """Display financial statements data"""
    tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    
    with tab1:
        if financials is not None and not financials.empty:
            st.subheader("Income Statement (Annual)")
            # Display recent years data
            recent_financials = financials.iloc[:, :4] if financials.shape[1] >= 4 else financials
            st.dataframe(recent_financials.T, use_container_width=True)
        else:
            st.warning("Income statement data not available")
    
    with tab2:
        if balance_sheet is not None and not balance_sheet.empty:
            st.subheader("Balance Sheet (Annual)")
            recent_balance = balance_sheet.iloc[:, :4] if balance_sheet.shape[1] >= 4 else balance_sheet
            st.dataframe(recent_balance.T, use_container_width=True)
        else:
            st.warning("Balance sheet data not available")
    
    with tab3:
        if cashflow is not None and not cashflow.empty:
            st.subheader("Cash Flow Statement (Annual)")
            recent_cashflow = cashflow.iloc[:, :4] if cashflow.shape[1] >= 4 else cashflow
            st.dataframe(recent_cashflow.T, use_container_width=True)
        else:
            st.warning("Cash flow data not available")

def main():
    st.title("📈 Advanced Stock Dashboard")
    st.markdown("Complete stock analysis with technical indicators, news, and paper trading")
    
    # Initialize session state
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = None
    
    initialize_watchlist()
    initialize_paper_trading()
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Watchlist section in sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("📋 Watchlist", expanded=False):
        display_watchlist()
    
    # Paper trading section in sidebar
    st.sidebar.markdown("---")
    with st.sidebar.expander("💼 Paper Trading", expanded=False):
        display_paper_trading_portfolio()
    
    # Show trending stocks section first
    st.markdown("---")
    
    # Load and display trending stocks
    try:
        trending_data = get_trending_stocks()
        display_trending_stocks(trending_data)
    except Exception as e:
        st.error("Unable to load trending stocks. Please try refreshing the page.")
    
    st.markdown("---")
    
    # Use the new search interface
    selected_from_search = create_stock_search_interface()
    
    # Fallback: Popular stocks dropdown for quick access
    st.sidebar.markdown("---")
    st.sidebar.header("Quick Select")
    
    popular_stocks = [
        "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "BABA"
    ]
    
    selected_stock = st.sidebar.selectbox(
        "Popular stocks:",
        [""] + popular_stocks,
        key="popular_stocks_select"
    )
    
    # Manual ticker input as backup
    ticker_input = st.sidebar.text_input(
        "Or enter ticker manually:",
        placeholder="e.g., AAPL, GOOGL, TSLA",
        help="Enter a valid stock ticker symbol",
        key="manual_ticker_input"
    ).upper()
    
    # Determine the ticker to use (priority: search > trending > manual > dropdown)
    ticker = (selected_from_search if selected_from_search
              else st.session_state.selected_ticker if st.session_state.selected_ticker
              else ticker_input if ticker_input 
              else selected_stock)
    
    # Clear session state after using it
    if st.session_state.selected_ticker:
        st.session_state.selected_ticker = None
    if selected_from_search:
        st.session_state.selected_from_search = None
    
   if ticker:
    st.sidebar.markdown(f"**Analyzing: {ticker}**")
    
    # your analysis logic here
    st.write("Analysis results go here...")
    st.line_chart(...)

    # Scroll to the analysis results automatically
    st.write(
        "<script>window.scrollTo(0, document.body.scrollHeight);</script>",
        unsafe_allow_html=True
    )

        
        # Add loading spinner
        with st.spinner(f'Fetching data for {ticker}...'):
            hist_data, info, financials, balance_sheet, cashflow = fetch_stock_data(ticker)
        
        if hist_data is not None and not hist_data.empty:
            # Calculate growth metrics
            growth_metrics = calculate_growth_metrics(hist_data)
            
            # Company info section with watchlist and trading buttons
            if info:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.header(f"{info.get('longName', ticker)} ({ticker})")
                    st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
                    st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
                
                with col2:
                    if ticker not in st.session_state.watchlist:
                        if st.button(f"➕ Add to Watchlist", key=f"add_watchlist_{ticker}"):
                            add_to_watchlist(ticker)
                            st.rerun()
                    else:
                        if st.button(f"➖ Remove from Watchlist", key=f"remove_watchlist_{ticker}"):
                            remove_from_watchlist(ticker)
                            st.rerun()
                
                with col3:
                    if st.button("🔄 Refresh Data", key=f"refresh_{ticker}"):
                        st.cache_data.clear()
                        st.rerun()
                
                if info.get('longBusinessSummary'):
                    with st.expander("Company Description"):
                        st.write(info['longBusinessSummary'])
            
            # News and Sentiment Section
            st.header("📰 Latest News & Sentiment")
            with st.spinner("Fetching latest news..."):
                news_articles, sentiment = get_stock_news_and_sentiment(ticker)
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                sentiment_color = {"Positive": "green", "Negative": "red", "Neutral": "gray"}[sentiment]
                st.markdown(f"**Market Sentiment:** <span style='color: {sentiment_color}'>{sentiment}</span>", 
                           unsafe_allow_html=True)
                
                if news_articles:
                    st.metric("News Articles", len(news_articles))
            
            with col2:
                if news_articles:
                    for i, article in enumerate(news_articles[:3]):  # Show top 3 news
                        with st.expander(f"📄 {article['title'][:60]}..."):
                            st.write(f"**Source:** {article['source']}")
                            st.write(f"**Published:** {article['published_at'][:10]}")
                            st.write(article['description'])
                            st.markdown(f"[Read Full Article]({article['url']})")
                else:
                    st.info("No recent news articles found")
            
            # Key metrics
            st.header("📊 Key Metrics")
            display_key_metrics(info, growth_metrics)
            
            # Year-wise returns chart
            st.header("📊 Annual Returns Analysis")
            with st.spinner("Calculating yearly returns..."):
                yearly_returns_data = calculate_yearly_returns(ticker, years=10)
            
            if yearly_returns_data:
                yearly_returns_chart = create_yearly_returns_chart(yearly_returns_data, ticker)
                if yearly_returns_chart:
                    st.plotly_chart(yearly_returns_chart, use_container_width=True)
                
                # Display yearly returns summary
                if len(yearly_returns_data) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    positive_years = sum(1 for data in yearly_returns_data if data['return'] > 0)
                    negative_years = len(yearly_returns_data) - positive_years
                    avg_return = sum(data['return'] for data in yearly_returns_data) / len(yearly_returns_data)
                    best_year = max(yearly_returns_data, key=lambda x: x['return'])
                    worst_year = min(yearly_returns_data, key=lambda x: x['return'])
                    
                    with col1:
                        st.metric("Positive Years", f"{positive_years}/{len(yearly_returns_data)}")
                    
                    with col2:
                        st.metric("Average Return", f"{avg_return:.1f}%")
                    
                    with col3:
                        st.metric("Best Year", f"{best_year['year']}", f"{best_year['return']:+.1f}%")
                    
                    with col4:
                        st.metric("Worst Year", f"{worst_year['year']}", f"{worst_year['return']:+.1f}%")
            else:
                st.warning("Unable to calculate yearly returns for this stock. Data might be limited.")
            
            # Advanced Technical Analysis
            st.header("📈 Technical Analysis")
            
            # Calculate technical indicators
            with st.spinner("Calculating technical indicators..."):
                data_with_indicators = calculate_technical_indicators(hist_data)
                support_levels, resistance_levels = find_support_resistance_levels(hist_data)
            
            if data_with_indicators is not None:
                # Create advanced chart
                advanced_chart = create_advanced_price_chart(data_with_indicators, ticker, support_levels, resistance_levels)
                if advanced_chart:
                    st.plotly_chart(advanced_chart, use_container_width=True)
                
                # Display current technical indicators
                col1, col2, col3, col4 = st.columns(4)
                
                current_rsi = data_with_indicators['RSI'].iloc[-1]
                current_macd = data_with_indicators['MACD'].iloc[-1]
                current_ma50 = data_with_indicators['MA_50'].iloc[-1]
                current_price = data_with_indicators['Close'].iloc[-1]
                
                with col1:
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    st.metric("RSI (14)", f"{current_rsi:.1f}", rsi_signal)
                
                with col2:
                    macd_signal = "Bullish" if current_macd > 0 else "Bearish"
                    st.metric("MACD", f"{current_macd:.4f}", macd_signal)
                
                with col3:
                    ma50_signal = "Above MA50" if current_price > current_ma50 else "Below MA50"
                    st.metric("MA50 Position", f"${current_ma50:.2f}", ma50_signal)
                
                with col4:
                    if support_levels:
                        nearest_support = max([level for level in support_levels if level < current_price], default=0)
                        st.metric("Nearest Support", f"${nearest_support:.2f}")
                    else:
                        st.metric("Support Levels", "Not found")
            
            # Paper Trading Section
            st.header("📊 Paper Trading")
            current_price = hist_data['Close'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Current Price")
                st.metric("Live Price", f"${current_price:.2f}")
            
            with col2:
                st.subheader("Buy Order")
                buy_quantity = st.number_input("Shares to Buy", min_value=1, value=10, key=f"buy_qty_{ticker}")
                if st.button("🟢 Buy", key=f"buy_btn_{ticker}"):
                    execute_paper_trade(ticker, "BUY", buy_quantity, current_price)
                    st.rerun()
            
            with col3:
                st.subheader("Sell Order")
                sell_quantity = st.number_input("Shares to Sell", min_value=1, value=10, key=f"sell_qty_{ticker}")
                if st.button("🔴 Sell", key=f"sell_btn_{ticker}"):
                    execute_paper_trade(ticker, "SELL", sell_quantity, current_price)
                    st.rerun()
            
            # Enhanced Financials Section
            st.header("💼 Enhanced Financials")
            
            with st.spinner("Loading financial data..."):
                enhanced_financials, info_data = get_enhanced_financials(ticker)
                financial_charts = create_financial_charts(enhanced_financials)
            
            if enhanced_financials:
                # Display financial charts in tabs
                tabs = st.tabs(list(enhanced_financials.keys())[:4])  # Limit to 4 tabs
                
                for i, (metric, data) in enumerate(list(enhanced_financials.items())[:4]):
                    with tabs[i]:
                        if metric in financial_charts:
                            st.plotly_chart(financial_charts[metric], use_container_width=True)
                        
                        # Show data table
                        df = pd.DataFrame({
                            'Year': [year.year for year in data.index],
                            metric: data.values
                        })
                        st.dataframe(df, use_container_width=True)
            
            # Peer Comparison
            st.header("🏆 Peer Comparison")
            
            with st.spinner("Loading peer data..."):
                peer_data = get_peer_comparison(ticker, info)
            
            if peer_data:
                st.subheader(f"Top 3 Competitors in {info.get('sector', 'Same Sector')}")
                
                cols = st.columns(len(peer_data) + 1)
                
                # Current stock
                with cols[0]:
                    st.metric(
                        f"{ticker} (Current)",
                        f"${current_price:.2f}",
                        f"{growth_metrics.get('total_return', 0):.1f}% (5Y)"
                    )
                
                # Peer stocks
                for i, peer in enumerate(peer_data):
                    with cols[i + 1]:
                        st.metric(
                            peer['ticker'],
                            f"${peer['current_price']:.2f}",
                            f"{peer['ytd_return']:.1f}% (YTD)"
                        )
                        st.caption(peer['name'][:20] + "...")
            else:
                st.info("No peer comparison data available")
            
            # Returns analysis
            st.header("💹 Returns Analysis")
            returns_chart = create_returns_chart(hist_data, ticker)
            if returns_chart:
                st.plotly_chart(returns_chart, use_container_width=True)
            
            # Financial statements
            st.header("📋 Financial Statements")
            display_financial_statements(financials, balance_sheet, cashflow)
            
            # Enhanced Export Options
            st.header("📥 Export & Download")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # CSV Export
                csv_data = hist_data.to_csv()
                st.download_button(
                    label="📊 Download Historical Data (CSV)",
                    data=csv_data,
                    file_name=f"{ticker}_historical_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel Export with multiple sheets
                if growth_metrics:
                    # Create Excel file in memory
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        # Historical data sheet - remove timezone for Excel compatibility
                        hist_data_export = hist_data.copy()
                        if hist_data_export.index.tz is not None:
                            hist_data_export.index = hist_data_export.index.tz_localize(None)
                        hist_data_export.to_excel(writer, sheet_name='Historical_Data')
                        
                        # Metrics sheet
                        metrics_df = pd.DataFrame([growth_metrics]).T
                        metrics_df.to_excel(writer, sheet_name='Growth_Metrics')
                        
                        # Technical indicators sheet
                        if data_with_indicators is not None:
                            tech_data = data_with_indicators[['Close', 'RSI', 'MACD', 'MA_50', 'MA_200']].tail(100).copy()
                            if tech_data.index.tz is not None:
                                tech_data.index = tech_data.index.tz_localize(None)
                            tech_data.to_excel(writer, sheet_name='Technical_Indicators')
                    
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📈 Download Analysis Report (Excel)",
                        data=excel_data,
                        file_name=f"{ticker}_complete_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                # PDF Export (basic version)
                pdf_data = create_pdf_export(ticker, {}, growth_metrics)
                if pdf_data:
                    st.download_button(
                        label="📄 Download Summary (PDF)",
                        data=pdf_data,
                        file_name=f"{ticker}_summary_report.pdf",
                        mime="application/pdf"
                    )
            
            # Last updated info
            st.sidebar.markdown("---")
            st.sidebar.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        else:
            st.error(f"❌ Unable to fetch data for ticker '{ticker}'. Please check if the ticker symbol is correct.")
    
    else:
        st.info("👆 Please enter a stock ticker symbol or select from the dropdown to get started.")
        
        # Show example tickers
        st.subheader("Popular Stock Tickers")
        cols = st.columns(5)
        example_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        for i, example_ticker in enumerate(example_tickers):
            if cols[i % 5].button(example_ticker):
                st.rerun()

if __name__ == "__main__":
    main()
