import numpy as np

def CreateTimeSeries(df):
    """
    Create Time Series features based Date Index
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['week'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    df['DayOfYear'] = df.index.dayofyear

    return df

def add_lags(df,
             lags,
             column,
             num):
    df = df.copy()
    for i in lags:
        df[f'Lags{i}'] = df[column].shift(i*num)

    return df

def MoreFeatures(df,
                 column,
                 n_diff):
    df = df.copy()
    df['Diff'] = df[column].diff(n_diff)
    df['RollingMean'] = df[column].rolling(window=7).mean()
    df['SinWeek'] = np.sin(2 * np.pi * df['week']/52)
    df['CosWeek'] = np.cos(2 * np.pi * df['week']/52)

    return df
