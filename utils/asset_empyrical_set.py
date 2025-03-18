import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) # 当交易数据样本较少或资产价格波动极小时，信息比率计算中可能会出现除零情况，从而产生警告，此时结果可能为 NaN，可忽略该警告。

def calculate_risk_metrics(strategy_returns: pd.Series,
                           benchmark_returns: pd.Series,
                           risk_free_rate: float = 0.015) -> dict:
    """风险指标计算"""

    # 夏普比率（年化）
    excess_returns = strategy_returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    # 最大回撤
    cum_returns = (1 + strategy_returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns - peak) / peak
    max_drawdown = drawdown.min()

    # 信息比率
    active_returns = strategy_returns - benchmark_returns
    info_ratio = np.sqrt(252) * active_returns.mean() / active_returns.std()

    return {
        "夏普比率": round(sharpe, 4),
        "最大回撤": round(max_drawdown, 4),
        "信息比率": round(info_ratio, 4)
    }


def rolling_metrics(strategy_returns: pd.Series,
                    benchmark_returns: pd.Series,
                    window: int = 90) -> pd.DataFrame:
    """滚动风险指标"""

    rolling_data = []
    for i in range(len(strategy_returns) - window + 1):
        sub_returns = strategy_returns.iloc[i:i + window]
        sub_bench = benchmark_returns.iloc[i:i + window]

        metrics = calculate_risk_metrics(sub_returns, sub_bench)
        rolling_data.append(metrics)

    return pd.DataFrame(rolling_data, index=strategy_returns.index[window - 1:])
