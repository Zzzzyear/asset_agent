import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import empyrical
from scipy.stats import norm

app = Flask(__name__)


class FinancialMetrics:
    def beta(self):
        """处理无基准数据的情况"""
        if self.benchmark_returns is None:
            print("警告：缺少基准数据，贝塔系数无法计算")
            return np.nan
        return empyrical.beta(...)

    def information_ratio(self):
        """添加基准检查"""
        if self.benchmark_returns is None:
            print("警告：缺少基准数据，信息比率无法计算")
            return np.nan

    def __init__(self, returns, benchmark_returns=None, risk_free_rate=0.0):
        """
        初始化FinancialMetrics类。

        :param returns: pandas.Series 或 np.array，策略的收益率序列。
        :param benchmark_returns: pandas.Series 或 np.array，基准的收益率序列。
        :param risk_free_rate: float，无风险利率。
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate

    def volatility(self):
        """
        计算年化波动性。
        """
        return empyrical.annual_volatility(self.returns)

    def beta(self):
        """
        计算贝塔系数。
        """
        return empyrical.beta(self.returns, self.benchmark_returns)

    def sharpe_ratio(self):
        """
        计算夏普比率。
        """

        return empyrical.sharpe_ratio(self.returns, risk_free=self.risk_free_rate)

    def treynor_ratio(self):
        """
        计算特雷诺比率。
        """
        beta = self.beta()
        treynor_ratio = (np.mean(self.returns) - 0.0) / beta
        return treynor_ratio

    def sortino_ratio(self):
        """
        计算索提诺比率。
        """
        return empyrical.sortino_ratio(self.returns)

    def max_drawdown(self):
        """
        计算最大回撤。
        """
        return empyrical.max_drawdown(self.returns)

    def information_ratio(self):
        """
        计算信息比率。
        """
        # 计算超额收益
        excess_returns = self.returns - self.benchmark_returns

        # 计算超额收益的平均值
        mean_excess_return = np.mean(excess_returns)

        # 计算跟踪误差（超额收益的标准差）
        tracking_error = np.std(excess_returns)

        # 计算信息比率
        if tracking_error == 0:
            return float('inf')  # 避免除以零
        information_ratio = mean_excess_return / tracking_error
        return information_ratio

    def var(self, confidence_level=0.95):
        """
        计算VaR值。

        :param confidence_level: float，置信水平。
        """
        VaR = -norm.ppf(1 - confidence_level) * np.std(self.returns) * np.sqrt(len(self.returns))
        return VaR

    def cvar(self, confidence_level=0.95):
        """
        计算CVaR。

        :param confidence_level: float，置信水平。
        """
        sorted_returns = np.sort(self.returns)
        index_at_var = int((1 - confidence_level) * len(self.returns))
        cvar = -np.mean(sorted_returns[:index_at_var])
        return cvar

    def calmar_ratio(self, period='daily', annualization=None):
        """
        计算卡尔玛比率。

        :param period: str, 回报数据的周期，默认为天。
        :param annualization: int, 交易日总数（用于计算年化），默认为252个交易日。
        """
        return empyrical.calmar_ratio(self.returns, period=period, annualization=annualization)

def financial_metrics(returns, benchmark_returns, risk_free_rate):

    metrics = FinancialMetrics(returns, benchmark_returns, risk_free_rate)

    results = {
        "波动性": metrics.volatility(),
        "贝塔": metrics.beta(),
        "夏普比率": metrics.sharpe_ratio(),
        "特雷诺比率": metrics.treynor_ratio(),
        "索提诺比率": metrics.sortino_ratio(),
        "最大回撤": metrics.max_drawdown(),
        "卡尔玛比率": metrics.calmar_ratio(),
        "信息比率": metrics.information_ratio(),
        "var": metrics.var(),
        "cvar": metrics.cvar()
    }

    return results

# if __name__ == "__main__":
#     app.run(debug=True)

# 使用示例
if __name__ == "__main__":
    import pandas as  pd
    df = pd.read_excel('择时策略.xlsx',sheet_name='收益曲线')
    strategy_returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02,
                                 0.01, -0.02, 0.03, -0.01, 0.02,
                                 0.01, -0.04, 0.03, -0.01, 0.02,
                                 0.01, -0.02, 0.05, -0.02, 0.02,])
    benchmark_returns = np.array([0.01, 0.02, -0.03, 0.01, 0.01,
                                  0.02, 0.02, -0.03, 0.01, -0.01,
                                  0.05, -0.02, -0.03, 0.01, 0.01,
                                  0.01, 0.02, -0.04, -0.01, 0.01,])
    risk_free_rate = 0.02 / 252  # 假设无风险利率为2%
    metrics = FinancialMetrics(strategy_returns, benchmark_returns, risk_free_rate)
    print(f"波动性: {metrics.volatility():.4f}")
    print(f"贝塔系数: {metrics.beta():.4f}")
    print(f"夏普比率: {metrics.sharpe_ratio():.4f}")
    print(f"特雷诺比率: {metrics.treynor_ratio():.4f}")
    print(f"索提诺比率: {metrics.sortino_ratio():.4f}")
    print(f"最大回撤: {metrics.max_drawdown():.4f}")
    print(f"卡尔玛比率: {metrics.calmar_ratio():.4f}")
    print(f"信息比率: {metrics.information_ratio():.4f}")
    print(f"VaR值: {metrics.var():.4f}")
    print(f"CVaR值: {metrics.cvar():.4f}")
    results = financial_metrics(strategy_returns, benchmark_returns, risk_free_rate)


