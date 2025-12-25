import pandas as pd
import numpy as np

class Montecarlo:
    """
    This generate a Montecarlo simulation for the portfolio
    """
    def __init__(self, portfolio: dict[str, float], stock_data: pd.DataFrame, mean_return: pd.DataFrame, cov_matrix: pd.DataFrame):
        self.portfolio = portfolio
        self.stock_data = stock_data
        self.mean_return = mean_return
        self.cov_matrix = cov_matrix
        self.initail_portfolio = sum([stock * self.stock_data[ticker].iloc[-1] for ticker, stock in self.portfolio.items()])

    def get_weights(self):
        weights = {}
        for ticker, stock in self.portfolio.items():
            single_stock_weight = stock * self.stock_data[ticker].iloc[-1]
            weights[ticker] = single_stock_weight / self.initail_portfolio 
        return weights
    
    def montecarlo(self, mc_simulation, T):

        weights = self.get_weights()
        mean_matrix = np.full(shape=(T, len(weights.values())), fill_value=self.mean_return)
        mean_matrix = mean_matrix.T

        #This creates an empty container to store the results of the final portfolio value for each day of the simulation
        portfolio_sims = np.full(shape=(T, mc_simulation), fill_value=0.0)

        for n in range(0, mc_simulation):
            #Montecarlo loops
            Z = np.random.normal(size=(T, len(weights.values())))
            L = np.linalg.cholesky(self.cov_matrix)
            daily_returns = mean_matrix + np.inner(L, Z)
            portfolio_sims[:,n] = np.cumprod(np.inner(list(weights.values()), daily_returns.T) + 1)* self.initail_portfolio
        
        return  portfolio_sims




