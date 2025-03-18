import sys
import os
from datetime import datetime

# 将父目录加入 sys.path，确保可以导入 investment_rebalance.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from invest_agent.tools.re_average.investment_reaverage import run_re_average

def main():
    # 测试参数
    gold_code = "159562.XSHE"  # 黄金代码
    bond_code = "511270.XSHG"  # 债券代码
    stock_code = "000001.XSHE"  # 股票代码（传统策略需要）
    start_date = "2022-01-01"  # 回测起始时间
    rebalance_ratio = "25:25:50"  # 再平衡比例
    benchmark = "000300.XSHG"  # 基准代码

    # 获取当前时间戳，用于区分文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 测试推荐策略
    print("测试推荐策略...")
    try:
        # 设置输出目录和文件名
        output_dir = os.path.join(os.path.dirname(__file__), "../res")
        os.makedirs(output_dir, exist_ok=True)
        report = run_re_average(
            gold=gold_code,
            bond=bond_code,
            start_date=start_date,
            rebalance_ratio=rebalance_ratio,
            benchmark=benchmark,
            recommended=True,  # 推荐策略
            stock=None  # 推荐策略不需要股票代码
        )
        # 重命名生成的图片文件
        os.rename(
            os.path.join(output_dir, "final_report.png"),
            os.path.join(output_dir, f"final_report_recommended_{timestamp}.png")
        )
        print("推荐策略测试完成，报告内容：")
        print(report)
    except Exception as e:
        print(f"推荐策略测试失败: {e}")

    # 测试传统策略
    print("\n测试传统策略...")
    try:
        # 设置输出目录和文件名
        output_dir = os.path.join(os.path.dirname(__file__), "../res")
        os.makedirs(output_dir, exist_ok=True)
        report = run_re_average(
            gold=gold_code,
            bond=bond_code,
            start_date=start_date,
            rebalance_ratio=rebalance_ratio,
            benchmark=benchmark,
            recommended=False,  # 传统策略
            stock=stock_code  # 传统策略需要股票代码
        )
        # 重命名生成的图片文件
        os.rename(
            os.path.join(output_dir, "final_report.png"),
            os.path.join(output_dir, f"final_report_traditional_{timestamp}.png")
        )
        print("传统策略测试完成，报告内容：")
        print(report)
    except Exception as e:
        print(f"传统策略测试失败: {e}")


if __name__ == "__main__":
    main()