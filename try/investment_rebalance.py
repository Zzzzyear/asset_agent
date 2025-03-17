"""
哈利布朗组合策略回测系统
功能：多资产数据加载 -> 月末再平衡 -> 净值计算 -> 结果保存
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from asset_empyrical_set import financial_metrics

# ------------------------------
# 核心配置参数（可修改区域）
# ------------------------------
CONFIG = {
    # 目标资产比例配置
    "target_ratios": {
        "gold": 0.25,    # 黄金25%
        "bond": 0.25,    # 债券25%
        "strategy": 0.5  # 策略50%
    },
    "initial_capital": 1000000,  # 初始本金（100万元）
    "data_files": {
        "gold": "sample_gold.csv",    # 黄金数据文件
        "bond": "sample_bond.csv",    # 债券数据文件
        "strategy": "hs300_close_prices.csv"       # 策略数据文件
    },
    "output_image": "portfolio_performance.png",  # 输出图像路径
    # ------------------------------
    # 新增风险指标配置（保持原配置顺序不变）
    # ------------------------------
    "output_risk_image": "risk_metrics.png",     # 风险指标图输出路径
    "benchmark_file": "hs300_close_prices.csv",                      # 基准数据文件（需要时修改路径）
    "risk_free_rate": 0.015/252                   # 无风险利率（年化1.5%按日计算）
}

# ------------------------------
# 数据加载模块
# ------------------------------
def load_asset_data(file_path: str) -> pd.Series:
    """加载CSV数据（适配YYYY-MM-DD格式）"""
    try:
        # 读取CSV并指定列类型
        df = pd.read_csv(
            file_path,
            header=None,
            skiprows=1,
            names=["date", "close"],
            dtype={"date": str, "close": float}  # 强制日期列为字符串
        )

        # 转换日期格式（处理带时间的日期字符串）
        df["date"] = pd.to_datetime(
            df["date"].str.split().str[0],  # 分割日期和时间部分
            format="%Y-%m-%d"  # 明确指定日期格式
        )

        return df.set_index("date")["close"].sort_index()

    except Exception as e:
        print("出错的5行数据样例：")
        print(df.head() if 'df' in locals() else "无法读取数据")
        raise RuntimeError(f"文件加载失败: {file_path}\n错误信息: {str(e)}")


def load_all_data() -> pd.DataFrame:
    """加载并合并所有资产数据"""
    merged_df = pd.DataFrame()

    for asset in ["gold", "bond", "strategy"]:
        file_path = CONFIG["data_files"][asset]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 加载并归一化
        prices = load_asset_data(file_path)
        merged_df[asset] = prices / prices.iloc[0]

    # 策略数据占位
    # merged_df["strategy"] = 1.0

    # 新增基准数据加载（保持原有顺序，仅在最后添加）
    benchmark_file = CONFIG["benchmark_file"]
    if benchmark_file and os.path.exists(benchmark_file):
        prices = load_asset_data(benchmark_file)
        merged_df["benchmark"] = prices / prices.iloc[0]
    else:
        merged_df["benchmark"] = 1.0  # 基准数据占位

    # 填充缺失日期
    return merged_df.resample("D").last().ffill()

# ------------------------------
# 再平衡逻辑模块
# ------------------------------
def rebalance_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """执行月度再平衡（修复变量名错误）"""
    df_result = df.copy()
    initial = CONFIG["initial_capital"]

    # 初始化持仓份数
    holdings = {
        asset: (initial * ratio) / df[asset].iloc[0]
        for asset, ratio in CONFIG["target_ratios"].items()
    }

    # --- 关键修改：修正变量名 ---
    # 生成自然月末日期列表
    all_month_ends = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="M"
    )

    # 筛选实际存在的月末日期
    monthly_dates = [date for date in all_month_ends if date in df.index]
    monthly_dates = pd.DatetimeIndex(monthly_dates)  # 正确变量名

    # 记录持仓历史
    holdings_history = pd.DataFrame(
        index=df.index,
        columns=CONFIG["target_ratios"].keys()
    )
    holdings_history.iloc[0] = holdings

    # 按月调整持仓
    for i, date in enumerate(monthly_dates):
        if date not in df.index:
            print(f" 警告：{date.date()} 不存在于数据中，跳过")
            continue

        # 计算当前价值和目标持仓
        prices = df.loc[date]
        current_value = {asset: holdings[asset] * prices[asset] for asset in holdings}
        total = sum(current_value.values())
        target = {asset: total * ratio for asset, ratio in CONFIG["target_ratios"].items()}
        new_holdings = {asset: target[asset] / prices[asset] for asset in holdings}

        # 确定填充范围
        next_date = monthly_dates[i + 1] if i < len(monthly_dates) - 1 else df.index[-1]
        valid_dates = df.index[(df.index >= date) & (df.index <= next_date)]

        if not valid_dates.empty:
            # 生成正确形状的填充数据
            fill_data = pd.DataFrame(
                [new_holdings] * len(valid_dates),
                index=valid_dates,
                columns=holdings_history.columns
            )
            holdings_history.loc[valid_dates] = fill_data

    # 填充缺失值
    holdings_history.ffill(inplace=True)

    # 计算每日价值
    for asset in holdings:
        df_result[f"{asset}_value"] = holdings_history[asset] * df[asset]
    df_result["total_value"] = df_result[[f"{a}_value" for a in holdings]].sum(axis=1)

    return df_result

# ------------------------------
# 可视化与保存模块
# ------------------------------
def save_performance_plot(df: pd.DataFrame):
    """保存净值曲线图"""
    plt.figure(figsize=(12, 6))

    # 绘制总净值曲线
    df["total_value"].plot(
        linewidth=2,
        color='darkblue',
        label=f'总净值 ({df["total_value"].iloc[-1]/1e6:.2f}百万)'
    )

    # 绘制各资产净值
    colors = {"gold": "#FFD700", "bond": "#2E8B57", "strategy": "#6A5ACD"}
    for asset in ["gold", "bond", "strategy"]:
        df[f"{asset}_value"].plot(
            linestyle="--",
            alpha=0.6,
            color=colors[asset],
            label=f"{asset}净值"
        )

    # 图表装饰
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title("哈利布朗组合历史表现", fontsize=14)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("资产价值（人民币元）", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    # 保存图像
    plt.savefig(CONFIG["output_image"], dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------
# 新增风险指标可视化模块
# ------------------------------
def save_risk_metrics_plot(metrics_dict: dict):
    """绘制全量风险指标趋势图（动态布局）"""
    plt.figure(figsize=(16, 24))
    plt.suptitle("风险指标动态趋势分析", y=0.95, fontsize=14, fontweight='bold')

    # 指标配置参数
    metric_config = [
        {'key': '波动性', 'name': '年化波动率', 'color': '#1f77b4', 'ylim': None},
        {'key': '贝塔', 'name': '贝塔系数', 'color': '#ff7f0e', 'ylim': (-1, 3)},
        {'key': '夏普比率', 'name': '夏普比率', 'color': '#2ca02c', 'ylim': (-2, 5)},
        {'key': '索提诺比率', 'name': '索提诺比率', 'color': '#d62728', 'ylim': (-2, 5)},
        {'key': '最大回撤', 'name': '动态最大回撤', 'color': '#9467bd', 'ylim': (-0.5, 0)},
        {'key': 'VaR', 'name': '风险价值（95%）', 'color': '#8c564b', 'ylim': None},
        {'key': '信息比率', 'name': '信息比率', 'color': '#e377c2', 'ylim': (-1, 3)},
        {'key': '特雷诺比率', 'name': '特雷诺比率', 'color': '#7f7f7f', 'ylim': (-5, 10)},
        {'key': '卡尔玛比率', 'name': '卡尔玛比率', 'color': '#bcbd22', 'ylim': (-2, 5)},
        {'key': 'CVaR', 'name': '条件风险价值', 'color': '#17becf', 'ylim': None},
    ]

    # 动态创建子图
    rows = (len(metric_config) + 1) // 2
    for idx, config in enumerate(metric_config, 1):
        ax = plt.subplot(rows, 2, idx)
        series = metrics_dict.get(config['key'])

        if series is not None and not series.dropna().empty:
            # 绘制主曲线
            ax.plot(series.index, series,
                    color=config['color'],
                    linewidth=1.5,
                    label=config['name'])

            # 设置Y轴范围
            if config['ylim']:
                ax.set_ylim(config['ylim'])
            else:
                valid = series.dropna()
                if len(valid) > 0:
                    y_padding = (valid.max() - valid.min()) * 0.2
                    ax.set_ylim(valid.min() - y_padding, valid.max() + y_padding)

            # 零线标记
            if config['key'] in ['贝塔', '夏普比率', '信息比率']:
                ax.axhline(0, color='gray', linestyle=':', alpha=0.7)

            # 数据完整性标注
            completeness = 1 - series.isna().mean()
            ax.text(0.02, 0.95,
                    f"数据完整度: {completeness:.1%}",
                    transform=ax.transAxes,
                    fontsize=8,
                    color='#666666')

            # 图表装饰
            ax.set_title(config['name'], fontsize=10, pad=12)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='upper right', fontsize=8)
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(CONFIG["output_risk_image"], dpi=300, bbox_inches="tight")
    plt.close()

# ------------------------------
# 主程序
# ------------------------------
if __name__ == "__main__":
    # 数据加载
    print("加载数据中...")
    try:
        data = load_all_data()
        print(f"数据时间范围: {data.index[0].date()} 至 {data.index[-1].date()}")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit(1)

    # 执行再平衡
    print("\n 执行再平衡...")
    result = rebalance_portfolio(data)

    # 保存结果
    print("\n 生成可视化图表...")
    save_performance_plot(result)

    # 计算风险指标
    print("\n 计算风险指标...")
    returns = result["total_value"].pct_change().dropna()

    # 处理基准收益率
    benchmark_returns = None
    if "benchmark" in result:
        benchmark_returns = result["benchmark"].pct_change().dropna()
        # 精确日期对齐
        common_dates = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]

    # 获取全量时间序列指标
    metrics_series = financial_metrics(
        returns=returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=CONFIG["risk_free_rate"],
        mode='series'
    )

    # 异常值处理
    for key in metrics_series:
        series = metrics_series[key]
        metrics_series[key] = series.replace([np.inf, -np.inf], np.nan)

    save_risk_metrics_plot(metrics_series)
    print(f" 风险指标图已保存至: {os.path.abspath(CONFIG['output_risk_image'])}")