import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
from datetime import datetime
from typing import Dict
import warnings
# 如需屏蔽 RuntimeWarning 可启用以下两行
# warnings.filterwarnings("ignore", category=RuntimeWarning)

from .utils.data_jq_api import fetch_stock_close_price
from .utils.asset_empyrical_set import calculate_risk_metrics, rolling_metrics

# 设置中文显示（确保系统已安装 SimHei 字体）
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 策略配置（初始配置，不再使用字符串 mode，而用 recommended 布尔值，默认采用传统策略即 False）
STRATEGY_CONFIG = {
    "assets": {
        "gold": {"weight": 0.25},
        "bond": {"weight": 0.25},
        "strategy": {"code": "000905.XSHG", "weight": 0.5}  # 推荐策略用到
    },
    "benchmark": {"code": "000300.XSHG"},
    "risk_free": 0.015,
    "start_date": "2022-01-01"
    # 当采用传统策略时，会动态增加 "stock" 和 "cash" 字段；
    # 推荐策略由布尔字段 "recommended" 表示，默认 False
}


# ------------------------ 修改1：benchmark参数化 ------------------------
def load_data(gold_code: str, bond_code: str) -> pd.DataFrame:
    """加载各标的数据，并合并成一个 DataFrame（日期对齐）"""
    # 根据 STRATEGY_CONFIG["recommended"] 和是否存在 "stock" 键决定加载哪些数据
    if STRATEGY_CONFIG.get("recommended", False):
        if "stock" in STRATEGY_CONFIG["assets"]:
            # 推荐模式下同时加载传统策略所需数据，用于对比
            code_map = {
                "gold": gold_code,
                "bond": bond_code,
                "strategy": STRATEGY_CONFIG["assets"]["strategy"]["code"],
                "stock": STRATEGY_CONFIG["assets"]["stock"]["code"],
                "benchmark": STRATEGY_CONFIG["benchmark"]["code"]
            }
            asset_keys = ['gold', 'bond', 'strategy', 'stock']
        else:
            # 仅加载推荐策略所需数据
            code_map = {
                "gold": gold_code,
                "bond": bond_code,
                "strategy": STRATEGY_CONFIG["assets"]["strategy"]["code"],
                "benchmark": STRATEGY_CONFIG["benchmark"]["code"]
            }
            asset_keys = ['gold', 'bond', 'strategy']
    else:
        # 传统策略模式：必须提供 stock
        code_map = {
            "gold": gold_code,
            "bond": bond_code,
            "stock": STRATEGY_CONFIG["assets"]["stock"]["code"],
            "benchmark": STRATEGY_CONFIG["benchmark"]["code"]
        }
        asset_keys = ['gold', 'bond', 'stock']

    # 针对各标的，若 CSV 文件不存在或为空则下载数据
    for name in asset_keys:
        code = code_map[name]
        file_path = os.path.join(os.path.dirname(__file__), "data", f"{code.replace('.', '_')}_close.csv")
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
        file_path = os.path.join(os.path.dirname(__file__), "data", f"{code.replace('.', '_')}_close.csv")
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


def rebalance_strategy(data: pd.DataFrame) -> pd.Series:
    """
    原始再平衡策略（推荐策略）的模拟函数，不再直接使用，
    改为新增带组成部分记录的策略函数。
    模拟月末再平衡策略，并每日更新组合净值。
    每月最后一个交易日重新调仓，使得 bond、gold 和 strategy 保持 1:1:2 的市值比例，
    其余日按持仓市值更新净值。
    """
    capital = 1e6  # 初始本金，总资金为 100 万
    weights = [
        STRATEGY_CONFIG["assets"]["gold"]["weight"],
        STRATEGY_CONFIG["assets"]["bond"]["weight"],
        STRATEGY_CONFIG["assets"]["strategy"]["weight"]
    ]
    positions = {}
    strategy_nav = pd.Series(index=data.index, dtype=float)

    # 每月末（ME）作为调仓日
    rebalancing_dates = data.resample('ME').last().index
    for date in data.index:
        if date in rebalancing_dates:
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
        strategy_nav.loc[date] = capital

    return strategy_nav


# ------------------------ 修改3：新增策略模拟函数 ------------------------
def recommended_strategy(data: pd.DataFrame) -> (pd.Series, pd.DataFrame):
    """
    模拟推荐策略（黄金:债券:策略 = 1:1:2），记录每日组合净值及各资产组成部分净值。
    修改原因：满足升级版本对比输出，同时记录组合各组成部分，故新增该函数。
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
            positions = { a: (capital * weights[i]) / prices[a] for i, a in enumerate(assets) }
        else:
            if positions:
                capital = sum(positions[a] * data.loc[date, a] for a in assets)
        strategy_nav.loc[date] = capital
        if positions:
            for a in assets:
                composition_nav.loc[date, a] = positions[a] * data.loc[date, a]
        else:
            composition_nav.loc[date] = None
    return strategy_nav, composition_nav


def traditional_strategy(data: pd.DataFrame) -> (pd.Series, pd.DataFrame):
    """
    模拟传统策略（黄金:债券:股票:现金 = 1:1:1:1），记录每日组合净值及各资产组成部分净值。
    现金部分假设价格恒定为 1。
    修改原因：满足传统哈利布朗配比，同时增加股票代码输入。
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
                'cash': capital * weights[3]
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
         b. 各资产净值曲线图
         c. 动态最大回撤
         d. 动态夏普比率（升级版本时显示 推荐策略 & 传统策略 vs Benchmark）
    """
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "res")
    os.makedirs(output_dir, exist_ok=True)

    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # benchmark 按 100 万初始资金归一化
    benchmark_norm = 1e6 * (data['benchmark'] / data['benchmark'].iloc[0])
    
    # 设置四个子图纵向排列：4 行 1 列
    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    
    # 子图 (a)：净值曲线对比
    ax = axes[0]
    if rec_nav is not None:
        ax.plot(rec_nav.index, rec_nav / 1e4, label="Recommended Strategy", linewidth=1.5)  # 转换为一万单位
        if trad_nav is not None:
            ax.plot(trad_nav.index, trad_nav / 1e4, label="Traditional Strategy", linewidth=1.5)  # 转换为一万单位
    else:
        ax.plot(trad_nav.index, trad_nav / 1e4, label="Traditional Strategy", linewidth=1.5)  # 转换为一万单位
    ax.plot(benchmark_norm.index, benchmark_norm / 1e4, label="Benchmark", linewidth=1.5)  # 转换为一万单位
    ax.set_title("Net Value Curves (Ten Thousand)")
    ax.set_ylabel("Net Value (×10,000)")  # 设置纵轴标签，标明单位为一万
    ax.legend()
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))  # 保留两位小数
    
    # 子图 (b)：各资产净值曲线图（根据 composition 数据绘制）
    ax = axes[1]
    if rec_comp is not None:
        comp = rec_comp.copy() / 1e4  # 将单位转换为一万
        ax.set_title("Recommended Strategy Asset Net Value Curves (Ten Thousand)")
    else:
        comp = trad_comp.copy() / 1e4  # 将单位转换为一万
        ax.set_title("Traditional Strategy Asset Net Value Curves (Ten Thousand)")

    for col in comp.columns:
        ax.plot(comp.index, comp[col], label=col, linewidth=1.5)

    ax.set_ylabel("Net Value (×10,000)")  # 设置纵轴标签，标明单位为一万
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))  # 保留两位小数
    
    # 根据可用数据动态调整滚动窗口
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
        ax.plot(roll_dd_rec.index, roll_dd_rec, label="Recommended Max Drawdown", color='r', linewidth=1.5)
        if trad_nav is not None:
            trad_returns = trad_nav.pct_change().dropna()
            roll_metrics_trad = rolling_metrics(trad_returns, bench_returns, window=window)
            roll_dd_trad = roll_metrics_trad["最大回撤"]
            ax.plot(roll_dd_trad.index, roll_dd_trad, label="Traditional Max Drawdown", color='m', linewidth=1.5)
    else:
        trad_returns = trad_nav.pct_change().dropna()
        roll_metrics_trad = rolling_metrics(trad_returns, bench_returns, window=window)
        roll_dd_trad = roll_metrics_trad["最大回撤"]
        ax.plot(roll_dd_trad.index, roll_dd_trad, label="Traditional Max Drawdown", color='r', linewidth=1.5)
    
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
        if trad_nav is not None:
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
    # 保存图片文件，添加时间戳
    plt.savefig(os.path.join(output_dir, f"final_report_{timestamp}.png"))
    plt.close()
    
    # 计算整体风险指标（全期收益率），使用推荐策略数据（如果存在）
    if rec_nav is not None:
        overall_returns = rec_nav.pct_change().dropna()
    else:
        overall_returns = trad_nav.pct_change().dropna()
    overall_bench_returns = benchmark_norm.pct_change().dropna()
    overall_metrics = calculate_risk_metrics(overall_returns, overall_bench_returns, STRATEGY_CONFIG["risk_free"])
    
    report_text = (
         "投资组合回测结果报告：\n"
         "1. 净值曲线对比：展示了传统策略（或推荐策略）与基准的净值变化；\n"
         "2. 资产组成净值曲线：展示各组成部分在组合中的净值变化；\n"
         "3. 动态最大回撤：基于滚动窗口计算的最大回撤变化；\n"
         "4. 动态夏普比率：展示滚动窗口内夏普比率的动态变化；\n"
         f"请查看 res/final_report_{timestamp}.png 获取详细图示。\n"
         "全期风险指标（基于全期收益率）：夏普比率 {0}，最大回撤 {1}，信息比率 {2}。"
    ).format(overall_metrics['夏普比率'], overall_metrics['最大回撤'], overall_metrics['信息比率'])
    
    report = {"report": report_text}
    # 保存 JSON 文件，添加时间戳
    with open(os.path.join(output_dir, f"report_{timestamp}.json"), "w", encoding="utf-8") as f:
         json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report


def run_re_average(gold: str, bond: str, start_date: str = "2022-01-01", rebalance_ratio: str = "25:25:50", benchmark: str = "000300.XSHG", recommended: bool = False, stock: str = None) -> Dict:
    STRATEGY_CONFIG["start_date"] = start_date
    STRATEGY_CONFIG["benchmark"]["code"] = benchmark  # 更新 benchmark 为可变参数
    STRATEGY_CONFIG["recommended"] = recommended  # 用布尔值表示策略类型

    if recommended:
        # 推荐策略：使用 黄金:债券:策略 = 1:1:2 配比，不需要提供股票代码
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
        # 如果用户提供了 stock 参数，则同时加载传统策略对比，否则仅加载推荐策略
        if stock:
            STRATEGY_CONFIG["assets"]["stock"] = {"code": stock, "weight": 0.25}
            STRATEGY_CONFIG["assets"]["cash"] = {"weight": 0.25}
        else:
            STRATEGY_CONFIG["assets"].pop("stock", None)
            STRATEGY_CONFIG["assets"].pop("cash", None)
    else:
        # 传统策略：固定采用 1:1:1:1 配比，需要提供股票代码参数 --stock
        if not stock:
            raise ValueError("传统策略模式下必须提供股票代码参数 --stock")
        STRATEGY_CONFIG["assets"]["gold"]["weight"] = 0.25
        STRATEGY_CONFIG["assets"]["bond"]["weight"] = 0.25
        STRATEGY_CONFIG["assets"]["stock"] = {"code": stock, "weight": 0.25}
        STRATEGY_CONFIG["assets"]["cash"] = {"weight": 0.25}

    # 数据加载
    data = load_data(gold, bond)
    # benchmark 归一化：按 100 万初始资金计算
    benchmark_nav = 1e6 * (data['benchmark'] / data['benchmark'].iloc[0])
    
    # 根据策略类型执行不同策略模拟
    if STRATEGY_CONFIG.get("recommended", False):
        rec_nav, rec_comp = recommended_strategy(data)
        # 若 "stock" 键存在，则同时模拟传统策略用于对比；否则置为 None
        if "stock" in STRATEGY_CONFIG["assets"]:
            trad_nav, trad_comp = traditional_strategy(data)
        else:
            trad_nav, trad_comp = None, None
    else:
        trad_nav, trad_comp = traditional_strategy(data)
        rec_nav, rec_comp = None, None

    # 生成最终报告
    report = generate_report(data, benchmark_nav, trad_nav, trad_comp, rec_nav, rec_comp)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=str, required=True, help="黄金代码，如：159562.XSHE")
    parser.add_argument("--bond", type=str, required=True, help="债券代码，如：511270.XSHG")
    # 原有参数
    parser.add_argument("--start_date", type=str, required=False, default="2022-01-01", help="回测起始时间，如：2022-01-01")
    parser.add_argument("--rebalance_ratio", type=str, required=False, default="25:25:50", help="再平衡比例（黄金:债券:策略），例如 25:25:50")
    parser.add_argument("--benchmark", type=str, required=False, default="000300.XSHG", help="基准代码, 默认 000300.XSHG")
    # 修改：将传统和推荐的参数合并为布尔参数 --recommended，
    # 推荐策略时无需提供股票代码；传统策略则必须提供 --stock 参数（可选用于对比）
    parser.add_argument("--recommended", action="store_true", help="使用推荐策略 (黄金:债券:策略 = 1:1:2)，不需要提供股票代码；默认使用传统策略")
    parser.add_argument("--stock", type=str, required=False, help="股票代码，用于传统策略对比，如：000001.XSHE")
    args = parser.parse_args()

    try:
        report = run_re_average(args.gold, args.bond, start_date=args.start_date, rebalance_ratio=args.rebalance_ratio, benchmark=args.benchmark, recommended=args.recommended, stock=args.stock)
        print("回测完成！结果保存至 res/ 目录")
    except Exception as e:
        print(f"运行失败: {str(e)}")
        print("问题排查建议：")
        print("1. 检查所有标的代码是否正确")
        print("2. 确认 data/ 目录下的 csv 文件包含完整日期数据")
        print("3. 尝试删除 data/ 目录下的 csv 文件重新下载")
