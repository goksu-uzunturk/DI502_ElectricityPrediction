import pandas as pd
import numpy as np


def break_datetime(df):
    df['timestamp']= pd.to_datetime(df['timestamp'])
    df[['year','weekofyear','dayofweek']]= np.uint16(df['timestamp'].dt.isocalendar())
    df['month']= np.uint8(df['timestamp'].dt.month)
    df['hour']= np.uint8(df['timestamp'].dt.hour)
    return df