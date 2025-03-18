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

# 策略配置（初始配置，同时新增 mode 键，默认采用传统策略）
STRATEGY_CONFIG = {
    "assets": {
        "gold": {"weight": 0.25},
        "bond": {"weight": 0.25},
        "strategy": {"code": "000905.XSHG", "weight": 0.5}  # 推荐策略用到
    },
    "benchmark": {"code": "000300.XSHG"},
    "risk_free": 0.015,
    "start_date": "2022-01-01",
    # "mode": "traditional"  # 默认模式为传统策略，即经典哈利布朗配比：黄金:债券:股票:现金=1:1:1:1
}


# ------------------------ 修改1：benchmark参数化 ------------------------
# （在 run_re_average 中将更新STRATEGY_CONFIG["benchmark"]["code"]）

def load_data(gold_code: str, bond_code: str) -> pd.DataFrame:
    """加载各标的数据，并合并成一个DataFrame（日期对齐）"""
    # 根据不同策略模式构造 code_map 和需要下载数据的标的列表
    if STRATEGY_CONFIG["mode"] == "recommended":
        code_map = {
            "gold": gold_code,
            "bond": bond_code,
            "strategy": STRATEGY_CONFIG["assets"]["strategy"]["code"],
            "benchmark": STRATEGY_CONFIG["benchmark"]["code"]
        }
        asset_keys = ['gold', 'bond', 'strategy']
    else:  # traditional 模式
        code_map = {
            "gold": gold_code,
            "bond": bond_code,
            "stock": STRATEGY_CONFIG["assets"]["stock"]["code"],
            "benchmark": STRATEGY_CONFIG["benchmark"]["code"]
        }
        asset_keys = ['gold', 'bond', 'stock']

    # 针对 gold, bond 及策略（或股票），若 CSV 文件不存在或为空则下载数据
    for name in asset_keys:
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
    原始再平衡策略（推荐策略）的模拟函数，不再直接使用，改为新增带组成部分记录的策略函数。
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


# ------------------------ 修改3：新增策略模拟函数 ------------------------
def recommended_strategy(data: pd.DataFrame) -> (pd.Series, pd.DataFrame):
    """
    模拟推荐策略（黄金:债券:策略 = 1:1:2），记录每日组合净值及各资产组成部分净值。
    修改原因：为满足需求3中升级版本对比输出，同时记录组合各组成部分，故新增该函数。
    """
    capital = 1e6
    weights = [
        STRATEGY_CONFIG["assets"]["gold"]["weight"],
        STRATEGY_CONFIG["assets"]["bond"]["weight"],
        STRATEGY_CONFIG["assets"]["strategy"]["weight"]
    ]
    assets = ['gold', 'bond', 'strategy']
    positions = {}
    strategy_nav = pd.Series(index=data.index, dtype=float)
    composition_nav = pd.DataFrame(index=data.index, columns=assets, dtype=float)

    rebalancing_dates = data.resample('ME').last().index
    for date in data.index:
        if date in rebalancing_dates:
            if positions:
                capital = sum(positions[a] * data.loc[date, a] for a in assets)
            prices = data.loc[date, assets]
            positions = {a: (capital * weights[i]) / prices[a] for i, a in enumerate(assets)}
        else:
            if positions:
                capital = sum(positions[a] * data.loc[date, a] for a in assets)
        strategy_nav.loc[date] = capital
        # 记录各资产市值组成部分
        if positions:
            for a in assets:
                composition_nav.loc[date, a] = positions[a] * data.loc[date, a]
        else:
            composition_nav.loc[date] = None
    return strategy_nav, composition_nav


def traditional_strategy(data: pd.DataFrame) -> (pd.Series, pd.DataFrame):
    """
    模拟传统策略（黄金:债券:股票:现金 = 1:1:1:1），记录每日组合净值及各资产组成部分净值。
    现金部分假设价格恒定为1。
    修改原因：满足需求2中传统哈利布朗配比，同时增加股票代码输入。
    """
    capital = 1e6
    weights = [0.25, 0.25, 0.25, 0.25]
    assets = ['gold', 'bond', 'stock', 'cash']
    positions = {}
    strategy_nav = pd.Series(index=data.index, dtype=float)
    composition_nav = pd.DataFrame(index=data.index, columns=assets, dtype=float)

    rebalancing_dates = data.resample('ME').last().index
    for date in data.index:
        if date in rebalancing_dates:
            if positions:
                capital = (positions['gold'] * data.loc[date, 'gold'] +
                           positions['bond'] * data.loc[date, 'bond'] +
                           positions['stock'] * data.loc[date, 'stock'] +
                           positions['cash'] * 1)
            positions = {
                'gold': (capital * weights[0]) / data.loc[date, 'gold'],
                'bond': (capital * weights[1]) / data.loc[date, 'bond'],
                'stock': (capital * weights[2]) / data.loc[date, 'stock'],
                'cash': capital * weights[3]  # 现金部分，单位即资金量
            }
        else:
            if positions:
                capital = (positions['gold'] * data.loc[date, 'gold'] +
                           positions['bond'] * data.loc[date, 'bond'] +
                           positions['stock'] * data.loc[date, 'stock'] +
                           positions['cash'] * 1)
        strategy_nav.loc[date] = capital
        if positions:
            composition_nav.loc[date, 'gold'] = positions['gold'] * data.loc[date, 'gold']
            composition_nav.loc[date, 'bond'] = positions['bond'] * data.loc[date, 'bond']
            composition_nav.loc[date, 'stock'] = positions['stock'] * data.loc[date, 'stock']
            composition_nav.loc[date, 'cash'] = positions['cash'] * 1
        else:
            composition_nav.loc[date] = None
    return strategy_nav, composition_nav


# ------------------------ 修改4：生成最终报告图（四子图） ------------------------
def generate_report(data: pd.DataFrame, benchmark_nav: pd.Series, trad_nav: pd.Series, trad_comp: pd.DataFrame, rec_nav: pd.Series = None, rec_comp: pd.DataFrame = None) -> dict:
    """生成最终报告，包含一张由四个子图（纵向排列）组成的图和 JSON 文件
       子图：
         a. 净值曲线对比（传统/升级策略 vs Benchmark）
         b. 资产组成部分净值占比（堆叠面积图）
         c. 动态最大回撤
         d. 动态夏普比率（升级版本时显示 推荐策略 & 传统策略 vs Benchmark）
    """
    os.makedirs("res", exist_ok=True)
    
    # benchmark按100万初始资金进行归一化
    benchmark_norm = 1e6 * (data['benchmark'] / data['benchmark'].iloc[0])
    
    # 设置四个子图纵向排列：4行1列
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # 子图 (a)：净值曲线对比
    ax = axes[0]
    if rec_nav is not None:
         ax.plot(rec_nav.index, rec_nav, label="Recommended Strategy", linewidth=1.5)
         ax.plot(trad_nav.index, trad_nav, label="Traditional Strategy", linewidth=1.5)
    else:
         ax.plot(trad_nav.index, trad_nav, label="Traditional Strategy", linewidth=1.5)
    ax.plot(benchmark_norm.index, benchmark_norm, label="Benchmark", linewidth=1.5)
    ax.set_title("Net Value Curves")
    ax.legend()
    
    # 子图 (b)：资产组成部分净值占比（堆叠面积图）
    ax = axes[1]
    if rec_comp is not None:
         comp = rec_comp.copy()
    else:
         comp = trad_comp.copy()
    # 计算各组成部分比例
    comp_prop = comp.div(comp.sum(axis=1), axis=0)
    ax.stackplot(comp_prop.index, comp_prop.T, labels=comp_prop.columns)
    ax.set_title("Composition Proportion")
    ax.legend(loc='upper left')
    
    # 计算滚动风险指标时动态调整窗口：若数据不足90天，则使用可用的最长天数
    if rec_nav is not None:
         base_returns = rec_nav.pct_change().dropna()
    else:
         base_returns = trad_nav.pct_change().dropna()
    window = 90 if len(base_returns) >= 90 else len(base_returns)
    
    bench_returns = benchmark_norm.pct_change().dropna()
    
    # 子图 (c)：动态最大回撤
    ax = axes[2]
    if rec_nav is not None:
         rec_returns = rec_nav.pct_change().dropna()
         roll_metrics_rec = rolling_metrics(rec_returns, bench_returns, window=window)
         roll_dd_rec = roll_metrics_rec["最大回撤"]
         # 采用推荐策略的最大回撤
         ax.plot(roll_dd_rec.index, roll_dd_rec, label="Recommended Max Drawdown", color='r', linewidth=1.5)
         
         trad_returns = trad_nav.pct_change().dropna()
         roll_metrics_trad = rolling_metrics(trad_returns, bench_returns, window=window)
         roll_dd_trad = roll_metrics_trad["最大回撤"]
         # 同时展示传统策略的最大回撤
         ax.plot(roll_dd_trad.index, roll_dd_trad, label="Traditional Max Drawdown", color='m', linewidth=1.5)
    else:
         trad_returns = trad_nav.pct_change().dropna()
         roll_metrics_trad = rolling_metrics(trad_returns, bench_returns, window=window)
         roll_dd_trad = roll_metrics_trad["最大回撤"]
         ax.plot(roll_dd_trad.index, roll_dd_trad, label="Traditional Max Drawdown", color='r', linewidth=1.5)
    
    # Benchmark的滚动最大回撤
    roll_metrics_bench = rolling_metrics(bench_returns, bench_returns, window=window)
    roll_dd_bench = roll_metrics_bench["最大回撤"]
    ax.plot(roll_dd_bench.index, roll_dd_bench, label="Benchmark Max Drawdown", color='k', linewidth=1.5)
    ax.set_title("Rolling Maximum Drawdown")
    ax.legend()
    
    # 子图 (d)：动态夏普比率
    ax = axes[3]
    if rec_nav is not None:
         rec_returns = rec_nav.pct_change().dropna()
         roll_metrics_rec = rolling_metrics(rec_returns, bench_returns, window=window)
         roll_sharpe_rec = roll_metrics_rec["夏普比率"]
         ax.plot(roll_sharpe_rec.index, roll_sharpe_rec, label="Recommended Sharpe", color='b', linewidth=1.5)
         
         trad_returns = trad_nav.pct_change().dropna()
         roll_metrics_trad = rolling_metrics(trad_returns, bench_returns, window=window)
         roll_sharpe_trad = roll_metrics_trad["夏普比率"]
         ax.plot(roll_sharpe_trad.index, roll_sharpe_trad, label="Traditional Sharpe", color='g', linewidth=1.5)
    else:
         trad_returns = trad_nav.pct_change().dropna()
         roll_metrics_trad = rolling_metrics(trad_returns, bench_returns, window=window)
         roll_sharpe_trad = roll_metrics_trad["夏普比率"]
         ax.plot(roll_sharpe_trad.index, roll_sharpe_trad, label="Traditional Sharpe", color='b', linewidth=1.5)
    
    roll_metrics_bench = rolling_metrics(bench_returns, bench_returns, window=window)
    roll_sharpe_bench = roll_metrics_bench["夏普比率"]
    ax.plot(roll_sharpe_bench.index, roll_sharpe_bench, label="Benchmark Sharpe", color='k', linewidth=1.5)
    ax.set_title("Rolling Sharpe Ratio")
    ax.legend()
    
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "res/final_report.png"))
    plt.close()
    
    # 计算整体风险指标（全期收益率）
    if rec_nav is not None:
         overall_returns = rec_nav.pct_change().dropna()
    else:
         overall_returns = trad_nav.pct_change().dropna()
    overall_bench_returns = benchmark_norm.pct_change().dropna()
    overall_metrics = calculate_risk_metrics(overall_returns, overall_bench_returns, STRATEGY_CONFIG["risk_free"])
    
    # 修改 JSON 内容，反映当前四子图展示的功能
    report_text = (
         "投资组合回测结果报告：\n"
         "1. 净值曲线对比：展示了传统策略（或推荐策略）与基准的净值变化；\n"
         "2. 资产组成占比：展示各组成部分在组合中的占比；\n"
         "3. 动态最大回撤：基于滚动窗口计算的最大回撤变化；\n"
         "4. 动态夏普比率：展示滚动窗口内夏普比率的动态变化；\n"
         "请查看 res/final_report.png 获取详细图示。\n"
         "全期风险指标（基于全期收益率）：夏普比率 {0}，最大回撤 {1}，信息比率 {2}。"
    ).format(overall_metrics['夏普比率'], overall_metrics['最大回撤'], overall_metrics['信息比率'])
    
    report = {"report": report_text}
    with open(os.path.join(os.path.dirname(__file__), "res", "report.json"), "w", encoding="utf-8") as f:
         json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report



def run_re_average(gold: str, bond: str, start_date: str = "2022-01-01", rebalance_ratio: str = "25:25:50",
                   benchmark: str = "000300.XSHG", recommended: bool = False, stock: str = None) -> Dict:
    # ------------------------ 修改2：更新全局配置 ------------------------
    STRATEGY_CONFIG["start_date"] = start_date
    STRATEGY_CONFIG["benchmark"]["code"] = benchmark  # 更新benchmark为可变参数
    if recommended:
        STRATEGY_CONFIG["mode"] = "recommended"
        try:
            ratio_parts = rebalance_ratio.split(":")
            if len(ratio_parts) != 3:
                raise ValueError("rebalance_ratio 必须包含3个部分，格式为 黄金:债券:策略，例如 25:25:50")
            ratio = [float(x) / 100 for x in ratio_parts]
            STRATEGY_CONFIG["assets"]["gold"]["weight"] = ratio[0]
            STRATEGY_CONFIG["assets"]["bond"]["weight"] = ratio[1]
            STRATEGY_CONFIG["assets"]["strategy"]["weight"] = ratio[2]
        except Exception as e:
            raise ValueError(f"rebalance_ratio 格式错误: {str(e)}")
    else:
        STRATEGY_CONFIG["mode"] = "traditional"
        # 传统策略固定采用1:1:1:1配比
        STRATEGY_CONFIG["assets"]["gold"]["weight"] = 0.25
        STRATEGY_CONFIG["assets"]["bond"]["weight"] = 0.25
        if not stock:
            raise ValueError("传统策略模式下必须提供股票代码参数 --stock")
        STRATEGY_CONFIG["assets"]["stock"] = {"code": stock, "weight": 0.25}
        STRATEGY_CONFIG["assets"]["cash"] = {"weight": 0.25}

    # 数据加载
    data = load_data(gold, bond)
    # benchmark归一化
    benchmark_nav = 1e6 * (data['benchmark'] / data['benchmark'].iloc[0])

    # 根据策略模式执行不同策略模拟
    if STRATEGY_CONFIG["mode"] == "recommended":
        rec_nav, rec_comp = recommended_strategy(data)
        # 同时模拟传统策略用于对比
        trad_nav, trad_comp = traditional_strategy(data)
    else:
        trad_nav, trad_comp = traditional_strategy(data)
        rec_nav, rec_comp = None, None

    # 生成最终报告（包含一张四子图图形和 JSON 报告）
    report = generate_report(data, benchmark_nav, trad_nav, trad_comp, rec_nav, rec_comp)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True, help="黄金代码，如：159562.XSHE")
    parser.add_argument("--bond", type=str, required=True, help="债券代码，如：511270.XSHG")
    # 原有参数
    parser.add_argument("--start_date", type=str, required=False, default="2022-01-01",
                        help="回测起始时间，如：2022-01-01")
    parser.add_argument("--rebalance_ratio", type=str, required=False, default="25:25:50",
                        help="再平衡比例（黄金:债券:策略），例如 25:25:50")
    # ------------------------ 修改1、3：新增参数 ------------------------
    parser.add_argument("--benchmark", type=str, required=False, default="000300.XSHG",
                        help="基准代码, 默认 000300.XSHG")
    parser.add_argument("--recommended", action="store_true",
                        help="使用推荐策略 (黄金:债券:策略 = 1:1:2)，默认使用传统策略")
    parser.add_argument("--stock", type=str, required=False, help="股票代码，用于传统策略，如：000001.XSHE")
    args = parser.parse_args()

    try:
        # 通过调用 run_re_average 函数传入新的入参，注意这个函数内部会更新全局配置
        report = run_re_average(args.gold, args.bond, start_date=args.start_date, rebalance_ratio=args.rebalance_ratio,
                                benchmark=args.benchmark, recommended=args.recommended, stock=args.stock)
        print("回测完成！结果保存至 res/ 目录")
    except Exception as e:
        print(f"运行失败: {str(e)}")
        print("问题排查建议：")
        print("1. 检查所有标的代码是否正确")
        print("2. 确认data/目录下的csv文件包含完整日期数据")
        print("3. 尝试删除data/目录下的csv文件重新下载")
