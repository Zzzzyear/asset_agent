## 注意 ##
项目是一个Qwen-Agent的tool，若要运行需注意目录：

~/Qwen-Agent/invest_agent/tools/re_average



---
## 目录结构
```bash
~/Qwen-Agent/re_average/
├── data/               # 市场数据存储
│   ├── gold_close.csv
│   ├── bond_close.csv
│   └── ...其他资产数据
├── utils/              # 工具模块
│   ├── data_jq_api.py          # 数据获取接口
│   └── asset_empyrical_set.py  # 风险指标计算
├── res/                 # 结果输出
│   ├── final_report.png
│   └── report.json
├── test/              # 测试模块 ~/re_average下运行
│   ├── test_investment_reaverage.py         # 测试 
│   └── text                                 # 可编辑文本，暂存信息
├── investment_rebalance.py      # 主程序 ~/Qwen-Agent下运行
└── investment_rebalance_bp.py   # 主程序backup


```

## **1. 数据获取 (`utils/data_jq_api.py`)**
- 通过 **聚宽 API** (`get_price()`) 获取资产的历史数据，包括：
  - **黄金 (`gold`)**
  - **债券 (`bond`)**
  - **中证 500 指数 (`strategy`)**
  - **沪深 300 指数 (`benchmark`)**
- 这些数据会存储到 **`re_average/data/`** 目录下的 CSV 文件中，以便后续计算。

---

## **2. 资产再平衡策略 (`re_average/investment_rebalance.py`)**
- **数据读取**  
  读取 `data/` 目录下的历史价格数据，并进行预处理。
- **执行月末再平衡策略**
  - 计算黄金、债券的持仓比例，并进行调整。
  - 计算新的资产净值，模拟投资组合的变化。
- **计算净值曲线**
  - 计算 **策略净值** 和 **基准净值**。
  - 生成投资组合的历史净值数据。

---

## **3. 计算风险指标 (`utils/asset_empyrical_set.py`)**
- **计算单一时点的风险指标**
  - **夏普比率**：衡量策略的收益风险比
  - **最大回撤**：衡量策略的最大损失幅度
  - **信息比率**：衡量策略相对于基准的超额收益稳定性
- **滚动计算风险指标**
  - 采用**滑动窗口**方式（默认 `90` 天），计算：
    - **滚动夏普比率**
    - **滚动最大回撤**
    - **滚动信息比率**
  - **如果资产交易天数不足 90 天**，则调整窗口大小，取最大可用天数。

---

## **4. 可视化 (`investment_rebalance.py`)**
### **生成四张曲线图**
1. **策略净值曲线**  
   - 展示 **策略净值** vs **基准净值** 走势。
2. **资产配置曲线**
   - 原本是资产持仓比例，现在改为 **不同资产净值曲线**。
3. **滚动最大回撤曲线**  
   - 展示 **策略与基准的最大回撤情况**。
4. **滚动夏普比率曲线**  
   - 展示 **策略与基准的夏普比率变化情况**。


---

## **总结**
**一个资产配置策略回测框架**，它包含：
1. **数据获取**（从聚宽 API 获取历史数据）
2. **资产再平衡策略**（模拟黄金 & 债券投资）
3. **风险指标计算**（夏普比率、最大回撤等）
4. **可视化分析**（策略净值、滚动风险指标曲线）
