import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Display an image
image = Image.open("innova.png")
st.image(image, use_column_width=True)

# Write a title and description
st.write("""
# Innova Hackathon

Data Preprocessing steps as Data Visualization, Data Analysis & Data Manipulation.

***
""")

st.subheader("1. Basic Analysis")

@st.cache_data
def load_data():
    df = pd.read_excel("innova.xlsx")
    df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'])
    return df

df = load_data()


st.write("### DataFrame Head")
st.dataframe(df.head())

st.write("### DataFrame Shape")
st.write(df.shape)

st.write("### DataFrame Tail")
st.dataframe(df.tail())

st.write("### Data Types")
st.write(df.dtypes)

st.subheader("2. Missing Values")

unique_servers = df["SERVER_NAME"].unique()
st.write(f"### Unique Server Names\n{unique_servers}")
st.write("Note: There is only one server name.")

st.write("### Missing Values Count")
st.write(df.isnull().sum())

st.write("### Rows with Missing Values")
st.dataframe(df[df.isnull().any(axis=1)])

st.write("### Number of Duplicated Rows")
st.write(df.duplicated().sum())

if st.button("Line Plots for Downloads and Uploads"):
    df = df.set_index("TIME_STAMP")
    
    st.write("### DOWNLOAD Line Plot")
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["DOWNLOAD"], label="DOWNLOAD")
    plt.xlabel("TIME_STAMP")
    plt.ylabel("DOWNLOAD")
    plt.title("DOWNLOAD Over Time")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

    st.write("### UPLOAD Line Plot")
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["UPLOAD"], label="UPLOAD", color='orange')
    plt.xlabel("TIME_STAMP")
    plt.ylabel("UPLOAD")
    plt.title("UPLOAD Over Time")
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

if st.button("Daily Transactions"):
    df["TIME_STAMP"] = pd.to_datetime(df["TIME_STAMP"])
    df = df.set_index("TIME_STAMP")
    
    num_cols = df.select_dtypes(include='float64').columns
    
    for col in num_cols:
        daily_mean = df[col].resample('D').mean()
        st.write(f"### Daily Mean of {col}")
        plt.figure(figsize=(20, 5))
        plt.plot(daily_mean.index, daily_mean, color="green")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.title(f"Daily Mean of {col}")
        st.pyplot(plt.gcf())
        plt.clf()

if st.button("Weekly Transactions"):
    df["TIME_STAMP"] = pd.to_datetime(df["TIME_STAMP"])
    df = df.set_index("TIME_STAMP")
    
    num_cols = df.select_dtypes(include='float64').columns
    
    for col in num_cols:
        weekly_mean = df[col].resample('W').mean()
        st.write(f"### Weekly Mean of {col}")
        plt.figure(figsize=(20, 5))
        plt.plot(weekly_mean.index, weekly_mean, color="green")
        plt.xlabel("Date")
        plt.ylabel(col)
        plt.title(f"Weekly Mean of {col}")
        st.pyplot(plt.gcf())
        plt.clf()

st.subheader("Turning back to missing values")

df["DOWNLOAD"].interpolate(method="linear", inplace=True)
df["UPLOAD"].interpolate(method="linear", inplace=True)

st.write("### Remaining Missing Values After Interpolation")
remaining_missing = df.isnull().sum()
st.write(remaining_missing)

if st.button("Decomposition -- Original - (Trend + Seasonal) = Residual"):

    df["TIME_STAMP"] = pd.to_datetime(df["TIME_STAMP"])
    df = df.set_index("TIME_STAMP")

    decomposition_download = seasonal_decompose(df['DOWNLOAD'], model='multiplicative', period=288)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
    
    decomposition_download.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    
    decomposition_download.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    
    decomposition_download.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    
    decomposition_download.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()


st.subheader("Feature Engineering")
st.write("### Weeks and Year Extraction")

df = df.reset_index()
df['TIME_STAMP'] = pd.to_datetime(df['TIME_STAMP'])
df['WEEK'] = df['TIME_STAMP'].dt.isocalendar().week
df['YEAR'] = df['TIME_STAMP'].dt.year

st.write("### DataFrame with WEEK and YEAR Columns")
st.dataframe(df.head())

st.subheader("is_holiday feature")
st.write("### Weekends and New Year's Day as Holidays")

df['is_holiday'] = ((df['TIME_STAMP'].dt.weekday >= 5) | ((df['TIME_STAMP'].dt.month == 1) & (df['TIME_STAMP'].dt.day == 1))).astype(int)

st.write("### Rows with Holidays")
st.dataframe(df[df["is_holiday"] == 1].head())

if st.button("Plot Weekdays vs Holidays for Downloads"):
    weekdays = df[df['is_holiday'] == 0]
    holidays = df[df['is_holiday'] == 1]
    
    plt.figure(figsize=(14, 7))
    plt.plot(weekdays.index, weekdays['DOWNLOAD'], label='Weekdays', color='blue')
    plt.plot(holidays.index, holidays['DOWNLOAD'], label='Holidays', color='red')
    
    plt.title('Download Rates: Weekdays vs. Holidays')
    plt.xlabel('Time')
    plt.ylabel('Download')
    plt.legend()
    
    st.pyplot(plt.gcf())
    plt.clf()

if st.button("Upload Transactions | Weekdays vs Holidays"):
    weekdays = df[df['is_holiday'] == 0]
    holidays = df[df['is_holiday'] == 1]
    
    plt.figure(figsize=(14, 7))
    plt.plot(weekdays.index, weekdays['UPLOAD'], label='Weekdays', color='blue')
    plt.plot(holidays.index, holidays['UPLOAD'], label='Holidays', color='red')
    
    plt.title('Upload Rates: Weekdays vs. Holidays')
    plt.xlabel('Time')
    plt.ylabel('Upload')
    plt.legend()
    
    st.pyplot(plt.gcf())
    plt.clf()

st.subheader("Extracting Time_of_Day Feature")
st.write("### Hourly Categoric Intervaled Feature with Classes of Morning, Afternoon, and Night")

def classify_time_of_day(hour):
    if hour < 11:
        return 'Morning'
    elif hour < 18:
        return 'Afternoon'
    else:
        return 'Night'

df['Time_of_Day'] = df['TIME_STAMP'].dt.hour.map(classify_time_of_day)

st.write("### DataFrame with Time_of_Day Column")
st.dataframe(df.head())

if st.button("Plot Time of Day Feature"):
    monthly_df = df.groupby([df['TIME_STAMP'].dt.to_period('M'), 'Time_of_Day'])['DOWNLOAD'].mean().unstack()
    
    plt.figure(figsize=(14, 7))
    monthly_df.plot(kind='bar', stacked=True)
    
    plt.title('Average Download Rates by Time of Day per Month')
    plt.xlabel('Month')
    plt.ylabel('Average Download')
    plt.legend(title='Time of Day')
    
    st.pyplot(plt.gcf())
    plt.clf()

st.subheader("Extracting Month Data")

df['MONTH'] = df['TIME_STAMP'].dt.month

st.write("### DataFrame with MONTH Column")
st.dataframe(df.head())

st.subheader("Bonus Question")

interval = st.selectbox('Select the interval in hours:', [2, 3, 4, 6, 8])

def get_hour_interval(hour, interval):
    """Classifies hours into intervals of the specified length."""
    interval_start = (hour // interval) * interval
    interval_end = interval_start + interval
    return f'{str(interval_start).zfill(2)}:00 - {str(interval_end).zfill(2)}:00'

df['hour'] = df['TIME_STAMP'].dt.hour
df['interval'] = df['hour'].apply(lambda hour: get_hour_interval(hour, interval))

traffic_per_interval = df.groupby('interval')[['UPLOAD', 'DOWNLOAD']].sum()

max_upload_interval = traffic_per_interval['UPLOAD'].idxmax()
max_download_interval = traffic_per_interval['DOWNLOAD'].idxmax()
max_upload_traffic = traffic_per_interval['UPLOAD'].max()
max_download_traffic = traffic_per_interval['DOWNLOAD'].max()

st.write("### Traffic Analysis Per Interval")
st.write(f"**Maximum Upload Interval:** {max_upload_interval} with **{max_upload_traffic} units**")
st.write(f"**Maximum Download Interval:** {max_download_interval} with **{max_download_traffic} units**")

st.write("### Traffic Per Interval")
st.dataframe(traffic_per_interval)
