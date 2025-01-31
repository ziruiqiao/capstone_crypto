import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sentiment_predict.sentiment_tools_variables import logging

logger = logging.getLogger(__name__)


def crypto_price_prediction(coin):
    """
    输入加密货币代号，返回加密货币的价格预测结果。
    :param coin: 加密货币代码（如'BTC'）
    :return: 预测结果列表（包括历史数据和预测数据）
    """
    def validate_crypto_symbol(symbol):
        """动态验证交易对有效性"""
        formatted_symbol = f"t{symbol.upper()}USD"
        test_url = f'https://api-pub.bitfinex.com/v2/candles/trade:1h:{formatted_symbol}/hist'
        params = {"limit": 1, "sort": -1}
        
        try:
            response = requests.get(test_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data[0]) >= 6:
                    return formatted_symbol
            return None
        except requests.exceptions.RequestException:
            return None
    
    def fetch_crypto_data(symbol, valid_symbol):
      """获取指定加密货币数据"""
      logger.debug(f"\n正在获取 {symbol} 的过去7天每小时数据...")
      
      end_date = datetime.now()
      start_date = end_date - timedelta(days=7)
      
      start_ts = int(start_date.timestamp() * 1000)
      end_ts = int(end_date.timestamp() * 1000)
      
      url = f'https://api-pub.bitfinex.com/v2/candles/trade:1h:{valid_symbol}/hist'
      params = {"start": start_ts, "end": end_ts, "sort": 1, "limit": 1000}
      
      try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
          
        if not data:
            logger.error("警告：获取到空数据集")
            return None
              
        df = pd.DataFrame(data, columns=["timestamp", "open", "close", "high", "low", "volume"])
        df['change'] = (df['close'] - df['open']) / df['open'] * 100
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('datetime', inplace=True)  # 设置 datetime 为索引
        df = df.sort_index()
        return df[['open', 'high', 'low', 'close', 'volume', 'change']]
    
      except requests.exceptions.HTTPError as e:
        logger.error(f"API错误：{str(e)}")
        logger.error(f"响应内容：{response.text[:200]}")
        return None
      except Exception as e:
        logger.error(f"获取数据失败：{str(e)}")
        return None

    # 获取有效交易对
    valid_symbol = validate_crypto_symbol(coin)
    if valid_symbol is None:
        logger.error(f"错误：{coin} 不是有效的交易对")
        return None

    # 获取数据
    df = fetch_crypto_data(coin, valid_symbol)
    
    if df is None or df.empty:
        logger.error("无法获取有效数据")
        return None
    
    # 数据处理和预测
    features = ['open', 'high', 'low', 'close', 'volume', 'change']
    recent_days = 7 * 24
    data_nt = df[features].dropna().iloc[-recent_days:]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_nt)
    
    input_steps = 168  # 过去 7 天
    output_steps = 24  # 预测 1 天
    X_latest = np.expand_dims(scaled_data, axis=0)
    
    model = load_model("market_data_predict/lstm_model.keras")
    prediction_scaled = model.predict(X_latest)
    
    prediction_actual = scaler.inverse_transform(
        prediction_scaled.reshape(-1, len(features))
    ).reshape(output_steps, len(features))

    # 构造预测结果 DataFrame
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1),  # 关键修复：使用 df 而非 data
        periods=output_steps, 
        freq='H'
    )
    
    predicted_df = pd.DataFrame(
        index=future_dates, 
        data=prediction_actual, 
        columns=features
    )

        # 获取最后 2 天的历史数据
    history_days = 2 * 24
    historical = df.iloc[-history_days:]  # 使用 df 而非 data

    # 可视化结果
    plt.figure(figsize=(15, 10))
    plot_features = ['close', 'change', 'volume']
    titles = ['Price', 'Change', 'Volume']

    for i, (feature, title) in enumerate(zip(plot_features, titles), 1):
        plt.subplot(3, 1, i)
        
        # 绘制历史数据
        plt.plot(historical.index, historical[feature], label='History (Last 2 Days)', color='blue')
        
        # 绘制预测数据
        plt.plot(predicted_df.index, predicted_df[feature], marker='o', linestyle='--', color='red', label='Prediction (Next 1 Day)')
        
        plt.title(f'{coin} {title} Prediction')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'data/{coin}_prediction_visualization.png')
    plt.show()

    # 只保留 close, change, volume
    selected_features = ['close', 'change', 'volume']
    last_history_time = historical.index[-1]
    last_history_data = historical.loc[[last_history_time], selected_features]
    last_history_data['type'] = 'history'

    last_prediction_time = predicted_df.index[-1]
    last_prediction_data = predicted_df.loc[[last_prediction_time], selected_features]
    last_prediction_data['type'] = 'prediction'

    final_df = pd.concat([last_history_data, last_prediction_data])
    final_df = final_df.reset_index()
    final_df.columns.values[0] = 'datetime'

    last_column = final_df.columns[-1]
    final_df = final_df[[last_column] + [col for col in final_df.columns if col != last_column]]

    predict_results = final_df.to_dict()


    logger.debug(f"\n最终数据已保存至 {coin}_last_hour_prediction.csv")
    
    return predict_results
