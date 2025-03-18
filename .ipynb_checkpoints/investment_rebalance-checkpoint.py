import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
from datetime import datetime
from typing import Dict

from .utils.data_jq_api import fetch_stock_close_price
from .utils.asset_empyrical_set import calculate_risk_metrics, rolling_metrics

# 设置中文显示（确保系统已安装 SimHei 字体）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 策略配置
STRATEGY_CONFIG = {
    "assets": {
        "gold": {"weight": 0.25},
        "bond": {"weight": 0.25},
        "strategy": {"code": "000905.XSHG", "weight": 0.5}
    },
    "benchmark": {"code": "000300.XSHG"},
    "risk_free": 0.015,
    "start_date": "2022-01-01"
}


def load_data(gold_code: str, bond_code: str) -> pd.DataFrame:
    """加载各标的数据，并合并成一个DataFrame（日期对齐）"""
    code_map = {
        "gold": gold_code,
        "bond": bond_code,
        "strategy": STRATEGY_CONFIG["assets"]["strategy"]["code"],
        "benchmark": STRATEGY_CONFIG["benchmark"]["code"]
    }

    # 针对 gold 和 bond，若 CSV 文件不存在或为空则下载数据
    for name in ['gold', 'bond', 'strategy']:
        code = code_map[name]
        file_path = os.path.join(os.path.dirname(__file__), os.path.join("data", f"{code.replace('.', '_')}_close.csv"))
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"正在下载 {code} 数据...")
            try:
                fetch_stock_close_price(code, STRATEGY_CONFIG["start_date"], datetime.today().strftime('%Y-%m-%d'))
            except Exception as e:
                raise RuntimeError(f"无法获取 {code} 数据: {str(e)}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件生成失败: {file_path}")

    # 读取各标的数据并处理数据格式
    data_frames = {}
    for name, code in code_map.items():
        file_path = os.path.join(os.path.dirname(__file__), os.path.join("data", f"{code.replace('.', '_')}_close.csv"))
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"读取 {file_path} 文件失败: {str(e)}")

        if df.empty:
            raise ValueError(f"{file_path} 文件为空，请检查数据源或重新下载。")

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        df.rename(columns={'close': name}, inplace=True)
        data_frames[name] = df

    data = pd.concat(data_frames.values(), axis=1, join='inner')
    data.dropna(inplace=True)

    if data.empty:
        raise ValueError("合并后的数据为空，请检查各 CSV 文件的数据是否完整。")

    return data


def rebalance_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    模拟月末再平衡策略，并每日更新组合净值。
    每月最后一个交易日重新调仓，使得bond，gold和strategy保持1:1:2的市值比例，其余日按持仓市值更新净值。
    """
    capital = 1e6  # 初始本金，总资金为 100 万
    weights = [
        STRATEGY_CONFIG["assets"]["gold"]["weight"],
        STRATEGY_CONFIG["assets"]["bond"]["weight"],
        STRATEGY_CONFIG["assets"]["strategy"]["weight"]
    ]
    positions = {}
    strategy_nav = pd.Series(index=data.index, dtype=float)

    # 每月末（ME：Month End）作为调仓日
    rebalancing_dates = data.resample('ME').last().index
    for date in data.index:
        if date in rebalancing_dates:
            # 如果已有持仓，则先按当日价格更新市值
            if positions:
                capital = sum(positions[asset] * data.loc[date, asset] for asset in positions)
            prices = data.loc[date, ['gold', 'bond', 'strategy']]
            positions = {
                'gold': (capital * weights[0]) / prices.iloc[0],
                'bond': (capital * weights[1]) / prices.iloc[1],
                'strategy': (capital * weights[2]) / prices.iloc[2]
            }
        else:
            if positions:
                capital = sum(positions[asset] * data.loc[date, asset] for asset in positions)
        strategy_nav.loc[date] = capital  # 记录实际资金

    return strategy_nav


def generate_report(data: pd.DataFrame, strategy_nav: pd.Series) -> None:
    """生成报告，包括两张图和 JSON 文件"""
    os.makedirs("res", exist_ok=True)

    # 计算各标归一化净值（归一化价格）
    gold_norm = 1e6 * STRATEGY_CONFIG["assets"]["gold"]["weight"] * (data['gold'] / data['gold'].iloc[0])
    bond_norm = 1e6 * STRATEGY_CONFIG["assets"]["bond"]["weight"] * (data['bond'] / data['bond'].iloc[0])
    strategy_single_norm = 1e6 * STRATEGY_CONFIG["assets"]["strategy"]["weight"] * (
                data['strategy'] / data['strategy'].iloc[0])

    # 计算组合净值（bond + gold + strategy）
    combined_nav = strategy_nav

    # benchmark按100万初始资金进行归一化
    benchmark_nav = 1e6 * (data['benchmark'] / data['benchmark'].iloc[0])

    # 第一张图：绘制黄金、债券、单一策略资产净值、组合净值和基准净值的曲线
    plt.figure(figsize=(12, 6))
    plt.plot(gold_norm.index, gold_norm, label="Gold")  # 黄金
    plt.plot(bond_norm.index, bond_norm, label="Bond")  # 债券
    plt.plot(strategy_single_norm.index, strategy_single_norm, label="Strategy Asset Net Value")  # 单一策略资产净值
    plt.plot(combined_nav.index, combined_nav, label="Combined Portfolio Net Value")  # 组合净值
    plt.plot(benchmark_nav.index, benchmark_nav, label="Benchmark Net Value")  # 基准净值
    plt.title("Net Value Curve Changes")  # 净值曲线变化
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(__file__), "res/net_value_curves.png"))
    plt.close()

    # 计算策略和基准的每日收益率（用于风险指标计算）
    strategy_returns = combined_nav.pct_change().dropna()
    bench_returns = benchmark_nav.pct_change().dropna()

    # 计算滚动风险指标（窗口期为90个交易日）
    roll_metrics = rolling_metrics(strategy_returns, bench_returns, window=90)

    # 第二张图：三个子图显示滚动风险指标曲线
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    axes[0].plot(roll_metrics.index, roll_metrics["夏普比率"], label="Sharpe Ratio", color='b')  # 夏普比率
    axes[0].set_title("Rolling Sharpe Ratio")  # 滚动夏普比率
    axes[0].legend()

    axes[1].plot(roll_metrics.index, roll_metrics["最大回撤"], label="Maximum Drawdown", color='r')  # 最大回撤
    axes[1].set_title("Rolling Maximum Drawdown")  # 滚动最大回撤
    axes[1].legend()

    axes[2].plot(roll_metrics.index, roll_metrics["信息比率"], label="Information Ratio", color='g')  # 信息比率
    axes[2].set_title("Rolling Information Ratio")  # 滚动信息比率
    axes[2].legend()

    plt.xlabel("Date")  # 日期
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "res/risk_metrics.png"))
    plt.close()

    # 计算整体风险指标（用全期收益率）
    overall_returns = combined_nav.pct_change().dropna()
    overall_bench_returns = benchmark_nav.pct_change().dropna()
    overall_metrics = calculate_risk_metrics(overall_returns, overall_bench_returns, STRATEGY_CONFIG["risk_free"])

    # 构造 JSON 报告文本（格式按照要求）
    report_text = (
        f"您选择的投资组合在过去有效期内的回测净值曲线如下：净值曲线图见 res/net_value_curves.png；"
        f"过去一个季度的三个风险指标分别是：夏普比率 {overall_metrics['夏普比率']}，"
        f"最大回撤 {overall_metrics['最大回撤']}，信息比率 {overall_metrics['信息比率']}；"
        f"过去有效期内指标的曲线图见 res/risk_metrics.png。"
    )

    report = {
        "report": report_text
    }

    with open("res/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report

def run_re_average(gold,bond):
    data = load_data(gold, bond)
    # 执行策略（每日更新净值）
    strategy_nav = rebalance_strategy(data)
    # 生成报告和图形
    report = generate_report(data, strategy_nav)
    return report
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True, help="黄金代码，如：159562.XSHE")
    parser.add_argument("--bond", type=str, required=True, help="债券代码，如：511270.XSHG")
    args = parser.parse_args()

    try:
        # 数据加载
        data = load_data(args.gold, args.bond)
        # 执行策略（每日更新净值）
        strategy_nav = rebalance_strategy(data)
        # 生成报告和图形
        generate_report(data, strategy_nav)
        print("回测完成！结果保存至 res/ 目录")
    except Exception as e:
        print(f"运行失败: {str(e)}")
        print("问题排查建议：")
        print("1. 检查所有标的代码是否正确")
        print("2. 确认data/目录下的csv文件包含完整日期数据")
        print("3. 尝试删除data/目录下的csv文件重新下载")
