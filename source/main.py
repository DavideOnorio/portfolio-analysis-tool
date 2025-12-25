from montecarlo import Montecarlo
from data import StockData
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

portfolio_to_analyse = {'AMZN': 10, 'ASML': 3, 'ELF': 30, 'GOOGL': 17, 'HOOD': 30, 'URTH':85, 'TSLA': 8, 'UBER': 20, 'UNH': 12}

stock_list = list(portfolio_to_analyse.keys())
start_date = '2010-01-01'

end_date = dt.datetime.now()
data = StockData(ticker=stock_list, start=start_date, end=end_date)
montec = Montecarlo(portfolio=portfolio_to_analyse, mean_return=data.get_geom_mean_return(), stock_data=data.stock_data, cov_matrix=data.get_cov_matrix())

portfolio_sims = montec.montecarlo(T=365, mc_simulation=1000)
initial_portfolio = montec.initail_portfolio

#It should be discounted (?)
expected_value = np.mean(portfolio_sims[-1, :])
percentile_5th = np.percentile(portfolio_sims[-1, :], 5)
VaR_95 = initial_portfolio - percentile_5th

print(f"95% Value at Risk: ${VaR_95:,.2f}")
print(f"Expected Portfolio Value: ${expected_value:,.2f}")
plt.plot(portfolio_sims)
plt.ylabel('Portfolio value ($)')
plt.xlabel('Days')
plt.show()