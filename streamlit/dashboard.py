import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import streamlit as st


path = kagglehub.dataset_download("lakshmi25npathi/bike-sharing-dataset")
dfDay = pd.read_csv(path + "/day.csv")

dfDay.hum *= 100
dfDay.windspeed *= 67
dfDay.yr += 2011
dfDay.atemp = 66*dfDay.atemp - 16
dfDay.temp = 47*dfDay.temp - 8

dfDay.rename(columns={
  'dteday': 'dateday',
  'yr': 'year',
  'mnth': 'month',
  'cnt': 'count'
}, inplace=True)

dfDay['dateday'] = pd.to_datetime(dfDay['dateday'])
dfDay['weekday'] = dfDay['dateday'].dt.day_name()
dfDay['year'] = dfDay['dateday'].dt.year
dfDay['season'] = dfDay['season'].map({
  1: 'Spring',
  2: 'Summer',
  3: 'Fall',
  4: 'Winter',
})
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

monthlyBiker['total_rides'] = monthlyBiker['casual'] + monthlyBiker['registered']
fig = px.bar(monthlyBiker,
             x='yearmonth',
             y=['casual', 'registered', 'total_rides'],
             barmode='group',
             color_discrete_sequence=["#F07167", "#FDFCDC", "#0081A7"],
             title="Monthly Bike Rides Trends in Recent Years",
             labels={'casual': 'Casual Rentals', 'registered': 'Registered Rentals', 'cnt': 'Total Rides', 'variable': 'User Type'})

fig.update_layout(xaxis_title='Month', yaxis_title='Total Rentals',
                  xaxis=dict(showgrid=False, showline=True, linecolor='rgb(204, 204, 204)', linewidth=2, mirror=True),
                  yaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor='rgb(204, 204, 204)', linewidth=2, mirror=True),
                  plot_bgcolor='rgba(255, 255, 255, 0)',
                  showlegend=True,
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

st.plotly_chart(fig, use_container_width=True)

fig = px.box(dfDay, x='weathersit', y='count', color='weathersit', 
             title='Bike Rides Distribution Based on Weather Condition',
             labels={'weathersit': 'Weather Condition', 'count': 'Total Rentals'})

st.plotly_chart(fig, use_container_width=True)

fig1 = px.box(dfDay, x='workingday', y='count', color='workingday',
              title='Bike Rides Clusters by Working Day',
              labels={'workingday': 'Working Day', 'count': 'Total Rentals'},
              color_discrete_sequence=['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF0000'])
fig1.update_xaxes(title_text='Working Day')
fig1.update_yaxes(title_text='Total Rentals')

fig2 = px.box(dfDay, x='holiday', y='count', color='holiday',
              title='Bike Rides Clusters by Holiday',
              labels={'holiday': 'Holiday', 'count': 'Total Rentals'},
              color_discrete_sequence=['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF0000'])
fig2.update_xaxes(title_text='Holiday')
fig2.update_yaxes(title_text='Total Rentals')

fig3 = px.box(dfDay, x='weekday', y='count', color='weekday',
              title='Bike Rides Clusters by Weekday',
              labels={'weekday': 'Weekday', 'count': 'Total Rentals'},
              color_discrete_sequence=['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF0000'])
fig3.update_xaxes(title_text='Weekday')
fig3.update_yaxes(title_text='Total Rentals')

st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)

fig = px.scatter(dfDay, x='atemp', y='count', color='season',
                 title='Bike Rides Clusters by Season and Temperature',
                 labels={'atemp': 'Temperature (°C)', 'count': 'Total Rentals'},
                 color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                 hover_name='season')

st.plotly_chart(fig, use_container_width=True)

seasonal_usage = dfDay.groupby('season')[['registered', 'casual']].sum().reset_index()

fig = px.bar(seasonal_usage, x='season', y=['registered', 'casual'],
             title='Bike Rides Counts by Season',
             labels={'season': 'Season', 'value': 'Total Rentals', 'variable': 'User Type'},
             color_discrete_sequence=["#00FF00","#0000FF"], barmode='group')

st.plotly_chart(fig, use_container_width=True)
st.caption('Created by Hafiz Isnaini')