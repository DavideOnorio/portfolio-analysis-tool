import yfinance as yf
import numpy as np

class StockData:
    """
    Downloads stock data and prepares it for portfolio analysis 
        Attributes:
        ticker (list): List of stock tickers to analyze
        start (str): Start date for data retrieval
        end (str): End date for data retrieval (defaults to today)
    """
    def __init__(self, ticker: list, start: str, end:str):
        self.ticker= ticker
        self.start = start
        self.end = end
        self.stock_data = yf.download(self.ticker, start=self.start, end=self.end)['Close']

    def get_geom_mean_return(self):
        log_returns = np.log(self.stock_data / self.stock_data.shift(1)).dropna()
        mean_log_returns = log_returns.mean()
        mean_log_returns = np.exp(mean_log_returns) - 1 
        return mean_log_returns

    def get_cov_matrix(self):
        log_returns = np.log(self.stock_data / self.stock_data.shift(1)).dropna()
        cov_matrix = log_returns.cov()
        return cov_matrix

