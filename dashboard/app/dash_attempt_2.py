import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from DI502_ElectricityPrediction.notebooks import utils
from IPython.display import Image
import os

train_data = utils.load_data()

train_data = utils.nan_weather_filler(train_data)

# Convert 'timestamp' to datetime
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])

# Create new columns
train_data['hour'] = train_data['timestamp'].dt.hour.astype(np.uint8)
train_data['month'] = train_data['timestamp'].dt.month.astype(np.uint8)
train_data['week'] = train_data['timestamp'].dt.isocalendar().week.astype(np.uint8)  # or train_data['timestamp'].dt.isocalendar().week.astype(np.uint8)
train_data['is_weekday'] = train_data['timestamp'].dt.weekday < 5


train_data['meter_reading_log']=np.log1p(train_data['meter_reading'])
merged_data = train_data.copy()


# Assuming train_data is your DataFrame
hour_mean = train_data.groupby('hour')['meter_reading_log'].mean()

# Create Seaborn line plot
hour_mean.plot(kind='line', color='skyblue')

# Convert Seaborn plot to Plotly figure
fig_hour_mean = make_subplots(rows=1, cols=1)

# Add Seaborn data to Plotly subplot
trace_hour_mean = go.Scatter(x=hour_mean.index, y=hour_mean.values, mode='lines', line=dict(color='skyblue'))
fig_hour_mean.add_trace(trace_hour_mean)

# Update layout with axis labels
fig_hour_mean.update_layout(
    xaxis=dict(title='Hour'),
    yaxis=dict(title='Mean of Log Meter readings'),
    paper_bgcolor='lavender',
    showlegend=False
)
train_data[['year','weekofyear','dayofweek']]= np.uint16(train_data['timestamp'].dt.isocalendar())


plt.figure(figsize=(8, 8))

day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
day_df = train_data.groupby(['dayofweek']).meter_reading_log.mean().reset_index()

# Create Seaborn plot
p = sns.lineplot(x=day_df['dayofweek'], y=day_df['meter_reading_log'], color='purple')
p.set_xticks(range(len(day_df['dayofweek'])))
p.set_xticklabels(day_labels)
plt.xlabel('Days of the week')
plt.ylabel("Log of Meter readings")

# Convert Seaborn plot to Plotly figure
fig8 = make_subplots(rows=1, cols=1)

# Add Seaborn data to Plotly subplot
trace = go.Scatter(x=day_df['dayofweek'], y=day_df['meter_reading_log'], mode='lines', line=dict(color='purple'))
fig8.add_trace(trace)

# Update layout if necessary
fig8.update_layout(xaxis=dict(title='Days of the week'),
                    yaxis=dict(title='Log of Meter readings'),
                   paper_bgcolor='lavender',
                    showlegend=False)


month_mean=train_data.groupby('month')['meter_reading_log'].mean()

month_mean.plot(kind='line',color='skyblue')

# Plot the line chart using Plotly Express
month_mean = px.line(
    x=month_mean.index,
    y=month_mean.values,
    labels={'x': 'Month', 'y': 'Mean Meter Reading'},
    title='Monthly Mean Meter Reading',
    line_shape='linear',  # Set to 'linear' for a line plot
    render_mode='svg'  # Use 'svg' for better rendering in Dash
)
plt.figure(figsize=(15, 8))
sns.set(style="whitegrid")



