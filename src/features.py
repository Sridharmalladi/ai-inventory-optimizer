import pandas as pd

def add_time_features(df):
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    return df

def add_rolling_features(df):
    df = df.sort_values(by=['Product ID', 'Date'])
    df['RollingDemand7'] = df.groupby('Product ID')['Units Sold'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df['RollingDemand14'] = df.groupby('Product ID')['Units Sold'].transform(lambda x: x.rolling(14, min_periods=1).mean())
    return df

def add_lag_features(df):
    df['Lag_1'] = df.groupby('Product ID')['Units Sold'].shift(1)
    return df
