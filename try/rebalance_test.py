"""
增强测试脚本 - 验证基准数据集成与风险指标生成
生成图像文件并保留结果
"""

import investment_rebalance as ir
import pandas as pd
import os

def test_benchmark_integration():
    """验证基准数据完整性与风险指标计算"""
    # 备份原始配置
    original_config = ir.CONFIG.copy()
    test_outputs = []

    try:
        # 配置测试环境
        print("\n🛠️ 初始化测试配置...")
        ir.CONFIG.update({
            "data_files": {
                "gold": "sample_gold.csv",
                "bond": "sample_bond.csv",
                "strategy": "hs300_close_prices.csv"

            },
            "benchmark_file": "hs300_close_prices.csv",
            "output_image": "enhanced_performance.png",
            "output_risk_image": "risk_metrics_report.png",
            "risk_free_rate": 0.015/252
        })
        test_outputs.extend([ir.CONFIG["output_image"], ir.CONFIG["output_risk_image"]])
        print("📁 输出文件将保留在:", [os.path.abspath(f) for f in test_outputs])

        # 文件存在性验证
        print("\n🔍 验证关键文件存在:")
        required_files = [
            ir.CONFIG["data_files"]["gold"],
            ir.CONFIG["data_files"]["bond"],
            ir.CONFIG["data_files"]["strategy"],
            ir.CONFIG["benchmark_file"]
        ]
        for f in required_files:
            assert os.path.exists(f), f"关键文件缺失: {f}"
            print(f"  ✓ {f} 存在")

        # 执行完整流程
        print("\n🚀 执行增强测试流程...")
        data = ir.load_all_data()

        # 验证基准数据加载
        assert "benchmark" in data.columns, "基准数据未成功加载"
        print(f"  ✓ 基准数据加载成功，时间范围: {data['benchmark'].first_valid_index()} 至 {data['benchmark'].last_valid_index()}")

        # 执行再平衡计算
        result = ir.rebalance_portfolio(data)

        # 验证风险指标计算
        returns = result["total_value"].pct_change().dropna()
        benchmark_returns = result["benchmark"].pct_change().dropna()

        # 日期对齐检查
        common_dates = returns.index.intersection(benchmark_returns.index)
        assert len(common_dates) > 0, "基准与组合日期无交集"
        print(f"  ✓ 日期对齐验证通过，共有{len(common_dates)}个交易日数据")

        # 生成可视化结果
        ir.save_performance_plot(result)
        ir.save_risk_metrics_plot(ir.financial_metrics(returns, benchmark_returns, ir.CONFIG["risk_free_rate"]))

        # 结果文件验证
        for output in test_outputs:
            assert os.path.exists(output), f"输出文件生成失败: {output}"
            print(f"  ✓ {output} 成功生成 → 文件路径: {os.path.abspath(output)}")

        # 风险指标合理性检查
        risk_metrics = ir.financial_metrics(returns, benchmark_returns, ir.CONFIG["risk_free_rate"],
                                            mode='single')  # 添加mode参数
        assert -1 < risk_metrics["贝塔"] < 2, "异常贝塔值范围"
        assert risk_metrics["夏普比率"] > -1, "不合理夏普比率"
        print("  ✓ 风险指标数值合理性验证通过")

        print("\n✅ 增强测试全部通过！结果文件已保留")

    finally:
        # 仅恢复配置，不删除文件
        print("\n🔧 恢复原始配置...")
        ir.CONFIG = original_config

if __name__ == "__main__":
    test_benchmark_integration()