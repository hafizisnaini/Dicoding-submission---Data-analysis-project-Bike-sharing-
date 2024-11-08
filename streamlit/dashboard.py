import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime
import streamlit as st


path = kagglehub.dataset_download("lakshmi25npathi/bike-sharing-dataset")
dfDay = pd.read_csv(path + "/day.csv")

dfDay.hum *= 100
dfDay.windspeed *= 67
dfDay.yr += 2011
dfDay.atemp = 66*dfDay.atemp - 16
dfDay.temp = 47*dfDay.temp - 8

q25, q75 = np.percentile(dfDay.hum, 25), np.percentile(dfDay.hum, 75)
iqr = q75 - q25
cut_off = iqr * 1.5
minimum, maximum = q25 - cut_off, q75 + cut_off

outliers = np.where(dfDay.hum < minimum)
dfDay = dfDay.drop(dfDay.index[outliers])

dfDay.rename(columns={
  'dteday': 'dateday',
  'yr': 'year',
  'mnth': 'month',
  'cnt': 'count'
}, inplace=True)

dfDay['dateday'] = pd.to_datetime(dfDay['dateday'])
dfDay['year'] = dfDay['dateday'].dt.year
dfDay['weathersit'] = dfDay['weathersit'].map({
  1: 'Clear/Partly Cloudy',
  2: 'Misty/Cloudy',
  3: 'Light Snow/Rain',
  4: 'Severe Weather'
})

monthlyBiker = dfDay.resample(rule='ME', on='dateday').agg({
  "casual": "sum",
  "registered": "sum",
  "count": "sum"
})

monthlyBiker.index = monthlyBiker.index.strftime('%b-%y')
monthlyBiker = monthlyBiker.reset_index()

monthlyBiker.rename(columns={
  "dateday": "yearmonth",
  "count": "total_rides",
}, inplace=True)

grouped_by_month = dfDay.groupby('month')
aggregated_stats_by_month = grouped_by_month['count'].agg(['max', 'min', 'mean', 'sum'])

grouped_by_weather = dfDay.groupby('weathersit')
aggregated_stats_by_weather = grouped_by_weather['count'].agg(['max', 'min', 'mean', 'sum'])

grouped_by_holiday = dfDay.groupby('holiday')
aggregated_stats_by_holiday = grouped_by_holiday['count'].agg(['max', 'min', 'mean', 'sum'])

grouped_by_weekday = dfDay.groupby('weekday')
aggregated_stats_by_weekday = grouped_by_weekday['count'].agg(['max', 'min', 'mean'])

grouped_by_workingday = dfDay.groupby('workingday')
aggregated_stats_by_workingday = grouped_by_workingday['count'].agg(['max', 'min', 'mean'])

grouped_by_season = dfDay.groupby('season')
aggregated_stats_by_season = grouped_by_season.agg({
  'casual': 'mean',
  'registered': 'mean',
  'count': ['max', 'min', 'mean']
})

aggregated_stats_by_season = dfDay.groupby('season').agg({
  'temp': ['max', 'min', 'mean'],
  'atemp': ['max', 'min', 'mean'],
  'hum': ['max', 'min', 'mean']
})

min_date = dfDay["dateday"].min()
max_date = dfDay["dateday"].max()

st.sidebar.header("Filter:")
start_date, end_date = st.sidebar.date_input(
  label="Date",
  min_value=min_date,
  max_value=max_date,
  value=[min_date, max_date]
)

st.sidebar.header("Connect with me:")
st.sidebar.markdown("Hafiz Isnaini")
st.sidebar.markdown("For inquiries and collaborations, feel free to contact me on [LinkedIn](https://www.linkedin.com/in/hafiz-isnaini/?trk=public_profile_browsemap)")
st.sidebar.markdown("---")
st.sidebar.markdown("[Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)")

dfMain = dfDay[
  (dfDay["dateday"] >= str(start_date)) &
  (dfDay["dateday"] <= str(end_date))
]

st.title("Bike Sharing Dashboard")
st.markdown("##")

col1, col2, col3 = st.columns(3)

with col1:
  st.metric("Total Rides", value=dfMain['count'].sum())

with col2:
  st.metric("Total Casual Rides", value=dfMain['casual'].sum())

with col3:
  st.metric("Total Registered Rides", value=dfMain['registered'].sum())

st.markdown("---")


#Visualization

fig = px.scatter(dfDay, x='atemp', y='count', color='year',
                 title='Total Bike Rider by RealFeel Temperature',
                 labels={'atemp': 'Temperature (°C)', 'count': 'Total Riders'},
                 hover_name='year')

years = dfDay['year'].unique()
for year in years:
    df_year = dfDay[dfDay['year'] == year]
    trendline = px.scatter(df_year, x='atemp', y='count', trendline='lowess')
    fig.add_trace(trendline.data[1])

st.plotly_chart(fig, use_container_width=True)


fig = px.scatter(dfDay, x='hum', y='count', color='year',
                 title='Total Bike Rider by Humidity',
                 labels={'hum': 'Humidity (%)', 'count': 'Total Riders'},
                 hover_name='year')

for year in years:
    df_year = dfDay[dfDay['year'] == year]
    trendline = px.scatter(df_year, x='hum', y='count', trendline='lowess')
    fig.add_trace(trendline.data[1])

st.plotly_chart(fig, use_container_width=True)


dayMean = dfDay.groupby(['year', 'weekday'])['casual'].mean().reset_index()

fig = px.line(dayMean, x='weekday', y='casual', color='year',
              title='Average Casual Bike Rider by Weekday',
              labels={'weekday': 'Day', 'casual': 'Total Casual Biker', 'year': 'Year'})

fig.update_xaxes(tickvals=[0, 1, 2, 3, 4, 5, 6], ticktext=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

fig.for_each_trace(
    lambda trace: trace.update(name=str(int(float(trace.name)))) if trace.name.isdigit() else None
)

st.plotly_chart(fig, use_container_width=True)


dayMean = dfDay.groupby(['year', 'weekday'])['registered'].mean().reset_index()

fig = px.line(dayMean, x='weekday', y='registered', color='year',
              title='Average Registered Bike Rider by Weekday',
              labels={'weekday': 'Day', 'registered': 'Total Registered Biker', 'year': 'Year'})

fig.update_xaxes(tickvals=[0, 1, 2, 3, 4, 5, 6], ticktext=['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

fig.for_each_trace(
    lambda trace: trace.update(name=str(int(float(trace.name)))) if trace.name.isdigit() else None
)

st.plotly_chart(fig, use_container_width=True)


monthMean = dfDay.groupby(['year', 'month'])['count'].mean().reset_index()

fig = px.line(monthMean, x='month', y='count', color='year',
              title='Average Bike Rider by Month',
              labels={'month': 'Month', 'count': 'Total Biker', 'year': 'Year'})

fig.for_each_trace(
    lambda trace: trace.update(name=str(int(float(trace.name)))) if trace.name.isdigit() else None
)

st.plotly_chart(fig, use_container_width=True)


season_mapping = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
dfDay['season'] = dfDay['season'].map(season_mapping)

dfDay['season'] = pd.Categorical(dfDay['season'], categories=['Spring', 'Summer', 'Fall', 'Winter'], ordered=True)

seasonMean = dfDay.groupby(['year', 'season'])['count'].mean().reset_index()

fig = px.line(seasonMean, x='season', y='count', color='year',
              title='Average Casual Bike Rider by Season',
              labels={'season': 'Season', 'count': 'Total Biker', 'year': 'Year'})

fig.update_xaxes(categoryorder='array', categoryarray=['Spring', 'Summer', 'Fall', 'Winter'])

fig.for_each_trace(
    lambda trace: trace.update(name=str(int(float(trace.name)))) if trace.name.isdigit() else None
)

st.plotly_chart(fig, use_container_width=True)


st.caption('Created by Hafiz Isnaini')