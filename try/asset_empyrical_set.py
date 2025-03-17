import pandas as pd
import numpy as np
import empyrical
from scipy.stats import norm


class FinancialMetrics:
    def __init__(self, returns, benchmark_returns=None, risk_free_rate=0.0, window=63):
        """
        完整版风险指标计算类
        - returns: 策略收益率序列
        - benchmark_returns: 基准收益率序列
        - risk_free_rate: 日化无风险利率
        - window: 滚动窗口周期（默认63个交易日）
        """
        # 数据预处理
        returns = returns.dropna()
        self.returns = returns.copy()
        self.risk_free_rate = risk_free_rate
        self.window = window

        # 基准数据处理
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.dropna()
            # 日期对齐
            common_idx = returns.index.intersection(benchmark_returns.index)
            self.returns_aligned = returns.loc[common_idx]
            self.benchmark_aligned = benchmark_returns.loc[common_idx]
        else:
            self.benchmark_aligned = None

    # --------------------------
    # 核心时间序列计算方法
    # --------------------------
    def rolling_volatility(self):
        """滚动年化波动率"""
        return self.returns.rolling(self.window).std() * np.sqrt(252)

    def rolling_beta(self):
        """动态贝塔系数"""
        if self.benchmark_aligned is None:
            return pd.Series(np.nan, index=self.returns.index)

        covariance = self.returns_aligned.rolling(self.window).cov(self.benchmark_aligned)
        benchmark_var = self.benchmark_aligned.rolling(self.window).var()
        beta = covariance / benchmark_var
        return beta.reindex(self.returns.index, method='ffill')

    def rolling_sharpe(self):
        """滚动夏普比率"""
        rolling_mean = self.returns.rolling(self.window).mean()
        rolling_std = self.returns.rolling(self.window).std()
        return (rolling_mean - self.risk_free_rate) / rolling_std * np.sqrt(252)

    def rolling_sortino(self):
        """滚动索提诺比率"""
        downside_returns = self.returns.where(self.returns < 0, 0)
        downside_std = downside_returns.rolling(self.window).std() * np.sqrt(252)
        rolling_mean = self.returns.rolling(self.window).mean()
        return rolling_mean / downside_std.replace(0, np.nan)

    def dynamic_max_drawdown(self):
        """动态最大回撤"""
        return self.returns.expanding().apply(
            lambda x: empyrical.max_drawdown(pd.Series(x))
        )

    def rolling_var(self, confidence_level=0.95):
        """滚动VaR（历史模拟法）"""
        return self.returns.rolling(self.window).apply(
            lambda x: -x.quantile(1 - confidence_level) * np.sqrt(len(x))
        )

    def rolling_information_ratio(self):
        """滚动信息比率"""
        if self.benchmark_aligned is None:
            return pd.Series(np.nan, index=self.returns.index)

        excess_returns = self.returns_aligned - self.benchmark_aligned
        tracking_error = excess_returns.rolling(self.window).std()
        return (excess_returns.rolling(self.window).mean() / tracking_error).reindex(self.returns.index, method='ffill')

    def rolling_treynor(self):
        """滚动特雷诺比率"""
        beta = self.rolling_beta()
        excess_return = self.returns.rolling(self.window).mean() - self.risk_free_rate
        return excess_return / beta.replace(0, np.nan)

    def rolling_calmar(self):
        """滚动卡尔玛比率（3年窗口）"""
        calmar_window = 252 * 3  # 3年窗口
        rolling_return = self.returns.rolling(calmar_window).mean()
        max_dd = self.returns.rolling(calmar_window).apply(empyrical.max_drawdown)
        return (rolling_return - self.risk_free_rate) / abs(max_dd.replace(0, np.nan))

    def rolling_cvar(self, confidence_level=0.95):
        """滚动CVaR"""

        def _cvar(x):
            var = -x.quantile(1 - confidence_level)
            return -x[x <= -var].mean()

        return self.returns.rolling(self.window).apply(_cvar)


def financial_metrics(returns, benchmark_returns, risk_free_rate, mode='series'):
    """
    完整版指标计算入口
    - mode:
      'series' - 返回时间序列指标
      'single' - 返回单值指标
    """
    metrics = FinancialMetrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate
    )

    if mode == 'single':
        return {
            "波动性": empyrical.annual_volatility(returns),
            "贝塔": empyrical.beta(returns, benchmark_returns),
            "夏普比率": empyrical.sharpe_ratio(returns, risk_free=risk_free_rate),
            "索提诺比率": empyrical.sortino_ratio(returns),
            "最大回撤": empyrical.max_drawdown(returns),
            "信息比率": empyrical.excess_sharpe(returns, benchmark_returns),
            "特雷诺比率": (returns.mean() - risk_free_rate) / empyrical.beta(returns, benchmark_returns),
            "卡尔玛比率": empyrical.calmar_ratio(returns),
            "VaR": -returns.quantile(0.05) * np.sqrt(252),
            "CVaR": -returns[returns <= returns.quantile(0.05)].mean()
        }
    else:
        return {
            "波动性": metrics.rolling_volatility(),
            "贝塔": metrics.rolling_beta(),
            "夏普比率": metrics.rolling_sharpe(),
            "索提诺比率": metrics.rolling_sortino(),
            "最大回撤": metrics.dynamic_max_drawdown(),
            "VaR": metrics.rolling_var(),
            "信息比率": metrics.rolling_information_ratio(),
            "特雷诺比率": metrics.rolling_treynor(),
            "卡尔玛比率": metrics.rolling_calmar(),
            "CVaR": metrics.rolling_cvar()
        }