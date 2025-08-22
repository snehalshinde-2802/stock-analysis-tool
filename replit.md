# Stock Dashboard

## Overview

A web-based stock market dashboard built with Streamlit that provides comprehensive financial analysis and visualization capabilities. The application fetches real-time stock data from Yahoo Finance and presents interactive charts, financial metrics, and historical performance data in an intuitive interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **Layout**: Wide layout configuration with expandable sidebar for navigation
- **Caching Strategy**: 5-minute TTL (time-to-live) data caching to balance performance and data freshness
- **Page Configuration**: Custom page title, icon, and responsive design settings

### Data Processing Layer
- **Data Source**: Yahoo Finance API via yfinance library for stock market data
- **Data Types**: Historical price data, company information, financial statements (income, balance sheet, cash flow)
- **Performance Optimization**: Streamlit's caching decorator to minimize API calls and improve response times
- **Error Handling**: Comprehensive exception handling for API failures and data inconsistencies

### Visualization Engine
- **Primary Library**: Plotly for interactive charts and graphs
- **Chart Types**: Multiple visualization options including line charts, candlestick charts, and subplots
- **Interactivity**: Plotly Express and Graph Objects for enhanced user interaction
- **Data Analysis**: NumPy and Pandas integration for mathematical calculations and data manipulation

### Core Features
- **Real-time Data Fetching**: Automated retrieval of stock prices and financial metrics
- **Growth Metrics Calculation**: Custom algorithms for calculating performance indicators
- **Multi-timeframe Analysis**: Support for various time periods (configurable via period parameter)
- **Financial Statement Analysis**: Processing of income statements, balance sheets, and cash flow data

## External Dependencies

### Data Sources
- **Yahoo Finance API**: Primary data source for stock prices, company information, and financial statements
- **yfinance Library**: Python wrapper for Yahoo Finance API access

### Visualization and Analysis
- **Plotly**: Interactive charting and data visualization library
- **Pandas**: Data manipulation and analysis framework
- **NumPy**: Numerical computing library for mathematical operations

### Web Framework
- **Streamlit**: Core web application framework for UI and deployment
- **Python Standard Libraries**: datetime and time modules for temporal data handling

### Development Tools
- **Error Handling**: Built-in exception management for robust data fetching
- **Performance Monitoring**: Caching mechanisms to optimize application responsiveness