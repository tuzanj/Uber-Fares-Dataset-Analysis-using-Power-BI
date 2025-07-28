

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, sin, cos, sqrt, asin

# ======================
# 1. DATA LOADING
# ======================
print("Loading dataset...")
df = pd.read_csv('./uber.csv')

# ======================
# 2. INITIAL EXPLORATION
# ======================
print("\n=== Dataset Overview ===")
print(f"Shape: {df.shape} (Rows, Columns)")
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes)
print("\nSummary Statistics:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

# ======================
# 3. DATA CLEANING
# ======================
print("\nCleaning data...")

# A. Remove invalid coordinates
df = df[(df['pickup_longitude'] != 0) & (df['dropoff_longitude'] != 0)]
print(f"Rows after removing invalid coordinates: {len(df)}")

# B. Convert datetime
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# C. Extract datetime features
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_day'] = df['pickup_datetime'].dt.day
df['pickup_month'] = df['pickup_datetime'].dt.month
df['pickup_year'] = df['pickup_datetime'].dt.year
df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek  # 0=Monday, 6=Sunday

# D. Remove fare outliers
df = df[(df['fare_amount'] >= 2.5) & (df['fare_amount'] <= 100)]

# E. Remove unrealistic passenger counts
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

# F. Filter to NYC area only
nyc_min_lon, nyc_max_lon = -74.05, -73.75
nyc_min_lat, nyc_max_lat = 40.63, 40.85

df = df[
    (df['pickup_longitude'].between(nyc_min_lon, nyc_max_lon)) &
    (df['pickup_latitude'].between(nyc_min_lat, nyc_max_lat)) &
    (df['dropoff_longitude'].between(nyc_min_lon, nyc_max_lon)) &
    (df['dropoff_latitude'].between(nyc_min_lat, nyc_max_lat))
]

# G. Calculate trip distance (Haversine formula)
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371 * c  # Earth radius in km
    return km

df['trip_distance_km'] = df.apply(
    lambda row: haversine(
        row['pickup_longitude'], row['pickup_latitude'],
        row['dropoff_longitude'], row['dropoff_latitude']
    ), axis=1
)

# Remove trips with 0 distance or extremely long distances
df = df[(df['trip_distance_km'] > 0.1) & (df['trip_distance_km'] < 50)]

# ======================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ======================
print("\nPerforming EDA...")

# Set style for plots
sns.set_style('whitegrid')
plt.figure(figsize=(15, 10))

# A. Fare amount distribution
plt.subplot(2, 2, 1)
sns.histplot(df['fare_amount'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Fare Amount')
plt.xlabel('Fare Amount ($)')

# B. Passenger count vs fare
plt.subplot(2, 2, 2)
sns.boxplot(x='passenger_count', y='fare_amount', data=df, palette='viridis')
plt.title('Fare Amount by Passenger Count')

# C. Hourly fare patterns
plt.subplot(2, 2, 3)
hourly_avg = df.groupby('pickup_hour')['fare_amount'].mean()
sns.lineplot(x=hourly_avg.index, y=hourly_avg.values, color='green', marker='o')
plt.title('Average Fare by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Fare ($)')

# D. Day of week fare patterns
plt.subplot(2, 2, 4)
dow_avg = df.groupby('pickup_dayofweek')['fare_amount'].mean()
sns.barplot(x=dow_avg.index, y=dow_avg.values, palette='coolwarm')
plt.title('Average Fare by Day of Week')
plt.xlabel('Day of Week (0=Monday)')
plt.ylabel('Average Fare ($)')

plt.tight_layout()
plt.show()

# E. Fare vs distance
plt.figure(figsize=(10, 6))
sns.scatterplot(x='trip_distance_km', y='fare_amount', data=df, alpha=0.5, color='purple')
plt.title('Fare Amount vs Trip Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount ($)')
plt.show()

# ======================
# 5. EXPORT CLEANED DATA
# ======================
print("\nExporting cleaned data...")
df.to_csv('uber_cleaned.csv', index=False)
print("Cleaned data saved as 'uber_cleaned.csv'")

print("\n=== Final Dataset Summary ===")
print(f"Final shape: {df.shape}")
print("\nSample of cleaned data:")
print(df.head())