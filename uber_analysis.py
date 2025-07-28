import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
df = pd.read_csv('./uber_cleaned.csv')

# 1. Descriptive Statistics
print("\n=== Basic Statistics ===")
print("fare_amount:")
print(f"- Mean: ${df['fare_amount'].mean():.2f}")
print(f"- Median: ${df['fare_amount'].median():.2f}")
print(f"- Mode: ${df['fare_amount'].mode()[0]:.2f}")
print(f"- Standard Deviation: ${df['fare_amount'].std():.2f}")
print(f"- Range: ${df['fare_amount'].min():.2f} to ${df['fare_amount'].max():.2f}")
print(f"- IQR (Q3-Q1): ${df['fare_amount'].quantile(0.75) - df['fare_amount'].quantile(0.25):.2f} (${df['fare_amount'].quantile(0.25):.2f} to ${df['fare_amount'].quantile(0.75):.2f})")

print("\ntrip_distance_km:")
print(f"- Mean: {df['trip_distance_km'].mean():.2f} km")
print(f"- Median: {df['trip_distance_km'].median():.2f} km")
print(f"- Mode: {df['trip_distance_km'].mode()[0]:.2f} km")
print(f"- Standard Deviation: {df['trip_distance_km'].std():.2f} km")
print(f"- Range: {df['trip_distance_km'].min():.2f} km to {df['trip_distance_km'].max():.2f} km")
print(f"- IQR (Q3-Q1): {df['trip_distance_km'].quantile(0.75) - df['trip_distance_km'].quantile(0.25):.2f} km ({df['trip_distance_km'].quantile(0.25):.2f} km to {df['trip_distance_km'].quantile(0.75):.2f} km)")

print("\npassenger_count:")
print(f"- Mean: {df['passenger_count'].mean():.1f} passengers")
print(f"- Median: {df['passenger_count'].median()} passenger")
print(f"- Mode: {df['passenger_count'].mode()[0]} passenger")

# Quartiles
print("\n=== Quartiles ===")
print("fare_amount:")
print(f"- 25th percentile (Q1): ${df['fare_amount'].quantile(0.25):.2f}")
print(f"- 50th percentile (Median): ${df['fare_amount'].quantile(0.5):.2f}")
print(f"- 75th percentile (Q3): ${df['fare_amount'].quantile(0.75):.2f}")

print("\ntrip_distance_km:")
print(f"- 25th percentile (Q1): {df['trip_distance_km'].quantile(0.25):.2f} km")
print(f"- 50th percentile (Median): {df['trip_distance_km'].quantile(0.5):.2f} km")
print(f"- 75th percentile (Q3): {df['trip_distance_km'].quantile(0.75):.2f} km")

# Outlier Identification
print("\n=== Outlier Identification ===")
fare_outliers = df[df['fare_amount'] > df['fare_amount'].quantile(0.75) + 1.5*(df['fare_amount'].quantile(0.75)-df['fare_amount'].quantile(0.25))]
print(f"- High fare outliers (>${fare_outliers['fare_amount'].min():.2f}) identified in {len(fare_outliers)} trips ({len(fare_outliers)/len(df)*100:.1f}% of data)")

distance_outliers = df[df['trip_distance_km'] > df['trip_distance_km'].quantile(0.75) + 1.5*(df['trip_distance_km'].quantile(0.75)-df['trip_distance_km'].quantile(0.25))]
print(f"- Long distance outliers (> {distance_outliers['trip_distance_km'].min():.1f} km) identified in {len(distance_outliers)} trips ({len(distance_outliers)/len(df)*100:.1f}% of data)")

# 2. Visualizations
plt.figure(figsize=(15, 10))

# Fare Amount Distribution
plt.subplot(2, 2, 1)
sns.histplot(df['fare_amount'], bins=30, kde=True)
plt.title('Distribution of Fare Amounts')
plt.xlabel('Fare Amount ($)')
plt.ylabel('Frequency')

# Boxplot of Fares by Time of Day
plt.subplot(2, 2, 2)
sns.boxplot(x='pickup_hour', y='fare_amount', data=df)
plt.title('Fare Amounts by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Fare Amount ($)')
plt.xticks(rotation=45)

# Fare Amount vs. Distance Traveled
plt.subplot(2, 2, 3)
sns.scatterplot(x='trip_distance_km', y='fare_amount', data=df, alpha=0.5)
plt.title('Fare Amount vs. Distance Traveled')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount ($)')

# Average Fare by Hour of Day
plt.subplot(2, 2, 4)
hourly_fares = df.groupby('pickup_hour')['fare_amount'].mean().reset_index()
sns.lineplot(x='pickup_hour', y='fare_amount', data=hourly_fares, marker='o')
plt.title('Average Fare by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Average Fare ($)')
plt.xticks(range(0, 24))

plt.tight_layout()
plt.show()

# 3. Correlation Analysis
print("\n=== Correlation Analysis ===")
corr_matrix = df[['fare_amount', 'trip_distance_km', 'passenger_count', 'pickup_latitude', 'pickup_year']].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Specific correlations
print(f"\n- Correlation between fare amount and distance: {df['fare_amount'].corr(df['trip_distance_km']):.2f}")
print(f"- Correlation between passenger count and fare amount: {df['passenger_count'].corr(df['fare_amount']):.2f}")
print(f"- Correlation between fare amount and pickup latitude: {df['fare_amount'].corr(df['pickup_latitude']):.2f}")
print(f"- Correlation between fare amount and year: {df['fare_amount'].corr(df['pickup_year']):.2f}")

# Additional Visualizations
plt.figure(figsize=(15, 5))

# Passenger Count vs Fare
plt.subplot(1, 3, 1)
sns.boxplot(x='passenger_count', y='fare_amount', data=df)
plt.title('Fare Amount by Passenger Count')
plt.xlabel('Passenger Count')
plt.ylabel('Fare Amount ($)')

# Fare Over Years
plt.subplot(1, 3, 2)
yearly_fares = df.groupby('pickup_year')['fare_amount'].mean().reset_index()
sns.lineplot(x='pickup_year', y='fare_amount', data=yearly_fares, marker='o')
plt.title('Average Fare Over Years')
plt.xlabel('Year')
plt.ylabel('Average Fare ($)')

# Distance vs Fare with regression line
plt.subplot(1, 3, 3)
sns.regplot(x='trip_distance_km', y='fare_amount', data=df, scatter_kws={'alpha':0.3})
plt.title('Distance vs Fare with Regression Line')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount ($)')

plt.tight_layout()
plt.show()

# Key Insights Summary
print("\n=== Key Insights ===")
print("1. The average Uber ride costs about ${:.2f} for a {:.1f}km trip".format(df['fare_amount'].mean(), df['trip_distance_km'].mean()))
print("2. Fares are highest during off-peak hours (late night/early morning)")
print("3. Distance is the strongest predictor of fare amount (r = {:.2f})".format(df['fare_amount'].corr(df['trip_distance_km'])))
print("4. About {:.1f}% of trips are potential outliers with unusually high fares or distances".format(len(fare_outliers)/len(df)*100))
print("5. Fares appear to have increased over the years covered in the dataset (r = {:.2f})".format(df['fare_amount'].corr(df['pickup_year'])))