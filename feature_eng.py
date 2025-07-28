import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('./uber_cleaned.csv')

# Convert pickup_datetime to datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# 3a. Create new analytical features from timestamps
# Extract hour, day, month
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month

# Day of week (0=Monday, 6=Sunday)
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek

# Create day type (weekday/weekend)
df['day_type'] = df['day_of_week'].apply(lambda x: 'weekend' if x >= 5 else 'weekday')

# Create time of day categories
def get_time_of_day(hour):
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'
    
df['time_of_day'] = df['hour'].apply(get_time_of_day)

# Create peak/off-peak indicator
def is_peak(hour, day_of_week):
    # Weekday rush hours (7-10am and 4-8pm)
    if day_of_week < 5:  # Weekday
        if (7 <= hour < 10) or (16 <= hour < 20):
            return 'peak'
    return 'off-peak'

df['peak_hour'] = df.apply(lambda x: is_peak(x['hour'], x['day_of_week']), axis=1)

# 3b. Encode categorical variables
# List of categorical features to encode
categorical_features = ['day_type', 'time_of_day', 'peak_hour']

# Initialize label encoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical feature
for feature in categorical_features:
    df[feature + '_encoded'] = label_encoder.fit_transform(df[feature])

# One-hot encoding alternative (uncomment if preferred)
# df = pd.get_dummies(df, columns=categorical_features, prefix=categorical_features)

# 3c. Save enhanced dataset
# Drop original datetime column if no longer needed
df.drop('pickup_datetime', axis=1, inplace=True)

# Save to new CSV file
df.to_csv('uber_enhanced.csv', index=False)

print("Feature engineering completed. Enhanced dataset saved as 'uber_enhanced.csv'")
print("\nNew features added:")
print("- hour, day, month (extracted from timestamp)")
print("- day_of_week (0-6 where 0=Monday)")
print("- day_type (weekday/weekend)")
print("- time_of_day (morning/afternoon/evening/night)")
print("- peak_hour (peak/off-peak)")
print("- Encoded versions of categorical variables")

# Display sample of new features
print("\nSample of new features:")
print(df[['hour', 'day', 'month', 'day_of_week', 'day_type', 'time_of_day', 'peak_hour']].head())