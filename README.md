# Uber Fares Dataset Analysis using Power BI

## 📁 Project Overview

This project analyzes the Uber Fares Dataset using Python for data cleaning and Power BI for data visualization. The goal is to derive insights into fare patterns, peak hours, and ride behavior.

---

## 📊 Tools Used

- Python (Pandas, NumPy)
- Power BI Desktop
- DAX (Data Analysis Expressions)
- GitHub for documentation

---

## 🔍 Methodology

### 1. Data Preparation

- Loaded 200,000 Uber fare records.
- Dropped null entries.
- Converted `pickup_datetime` to datetime object.
- Created new features: hour, day, month, weekday, peak/off-peak indicator.

### 2. Feature Engineering

```python
df['hour'] = df['pickup_datetime'].dt.hour
df['peak_time'] = df['hour'].apply(lambda x: 'Peak' if 7 <= x <= 9 or 16 <= x <= 19 else 'Off-Peak')
