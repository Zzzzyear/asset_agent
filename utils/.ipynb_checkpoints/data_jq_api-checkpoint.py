import os
import jqdatasdk as jq
import pandas as pd
from datetime import datetime

# 认证信息
user, pwd = '18610934225', 'Pingan112'
jq.auth(user, pwd)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, '../data')


def fetch_stock_close_price(code: str, start_date: str, end_date: str) -> str:
    """增强版数据获取函数，返回文件路径"""
    # 验证代码有效性
    try:
        security = jq.get_security_info(code)
        print(f"验证通过: {security.display_name} ({code})")
    except Exception as e:
        raise ValueError(f"无效代码 {code}: {str(e)}")

    # 创建数据目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 生成文件路径
    file_name = f"{code.replace('.', '_')}_close.csv"
    file_path = os.path.join(SAVE_DIR, file_name)

    # 获取数据
    try:
        df = jq.get_price(
            code,
            start_date=start_date,
            end_date=end_date,
            frequency='daily',
            fields=['close']
        )
        # 规范化数据格式
        df = df.reset_index()[['index', 'close']].rename(columns={'index': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    except Exception as e:
        raise RuntimeError(f"获取 {code} 数据失败: {str(e)}")

    # 保存文件
    try:
        df.to_csv(file_path, index=False)
        print(f"数据已保存至: {file_path}")
        return file_path
    except Exception as e:
        raise IOError(f"保存文件失败: {str(e)}")


if __name__ == "__main__":
    # 测试代码
    test_path = fetch_stock_close_price('000001.XSHE', '2024-01-01', '2024-05-31')
    print(pd.read_csv(test_path).head())
