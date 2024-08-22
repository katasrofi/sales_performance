def CreateTimeSeries(df):
    """
    Create Time Series features based Date Index
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['DayOfYear'] = df.index.dayofyear

    return df
