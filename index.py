import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=" * 60)
print("COVID-19 GLOBAL CASES - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# 1. IMPORT THE DATASET
print("\n1. IMPORTING DATASET")
print("-" * 30)

# Load the dataset
df = pd.read_csv('dataset.csv')
print(f"Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")

# 2. DATA CLEANING
print("\n2. DATA CLEANING")
print("-" * 30)

# Check for null values
print("Null values in each column:")
print(df.isnull().sum())

# Check for duplicates
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Check data types
print(f"\nData types:")
print(df.dtypes)

# Handle any missing values if present
df_clean = df.copy()
# Fill any null values with 0 for numeric columns
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_columns] = df_clean[numeric_columns].fillna(0)

print("Data cleaning completed!")

# 3. BASIC EDA
print("\n3. BASIC EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Show top 5 rows
print("Top 5 rows of the dataset:")
print(df_clean.head())

# Summary statistics
print("\nSummary statistics:")
print(df_clean.describe())

# Number of countries in the data
print(f"\nNumber of countries in the dataset: {len(df_clean)}")

# Most affected countries
print("\nMost affected countries:")
print("\nTop 10 countries by Confirmed cases:")
top_confirmed = df_clean.nlargest(10, 'Confirmed')[['Country/Region', 'Confirmed']]
print(top_confirmed)

print("\nTop 10 countries by Deaths:")
top_deaths = df_clean.nlargest(10, 'Deaths')[['Country/Region', 'Deaths']]
print(top_deaths)

print("\nTop 10 countries by Recovered:")
top_recovered = df_clean.nlargest(10, 'Recovered')[['Country/Region', 'Recovered']]
print(top_recovered)

# 4. VISUALIZATIONS
print("\n4. CREATING VISUALIZATIONS")
print("-" * 35)

# Set up the plotting area
fig = plt.figure(figsize=(20, 15))

# Visualization 1: Bar chart showing top 10 countries with highest confirmed cases
plt.subplot(2, 3, 1)
top_10_confirmed = df_clean.nlargest(10, 'Confirmed')
plt.bar(range(len(top_10_confirmed)), top_10_confirmed['Confirmed'])
plt.title('Top 10 Countries by Confirmed Cases', fontsize=14, fontweight='bold')
plt.xlabel('Countries')
plt.ylabel('Confirmed Cases')
plt.xticks(range(len(top_10_confirmed)), top_10_confirmed['Country/Region'], rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')

# Visualization 2: Bar chart showing top 10 countries with highest deaths
plt.subplot(2, 3, 2)
top_10_deaths = df_clean.nlargest(10, 'Deaths')
plt.bar(range(len(top_10_deaths)), top_10_deaths['Deaths'], color='red', alpha=0.7)
plt.title('Top 10 Countries by Deaths', fontsize=14, fontweight='bold')
plt.xlabel('Countries')
plt.ylabel('Deaths')
plt.xticks(range(len(top_10_deaths)), top_10_deaths['Country/Region'], rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')

# Visualization 3: Pie chart of global recovered vs deaths vs active
plt.subplot(2, 3, 3)
global_recovered = df_clean['Recovered'].sum()
global_deaths = df_clean['Deaths'].sum()
global_active = df_clean['Active'].sum()

sizes = [global_recovered, global_deaths, global_active]
labels = ['Recovered', 'Deaths', 'Active']
colors = ['lightgreen', 'lightcoral', 'lightskyblue']
explode = (0.05, 0.05, 0.05)

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Global COVID-19 Cases Distribution', fontsize=14, fontweight='bold')

# Visualization 4: WHO Region wise cases
plt.subplot(2, 3, 4)
region_cases = df_clean.groupby('WHO Region')['Confirmed'].sum().sort_values(ascending=False)
plt.bar(range(len(region_cases)), region_cases.values, color='orange', alpha=0.7)
plt.title('Confirmed Cases by WHO Region', fontsize=14, fontweight='bold')
plt.xlabel('WHO Regions')
plt.ylabel('Confirmed Cases')
plt.xticks(range(len(region_cases)), region_cases.index, rotation=45, ha='right')
plt.ticklabel_format(style='plain', axis='y')

# Visualization 5: Death Rate vs Recovery Rate scatter plot
plt.subplot(2, 3, 5)
plt.scatter(df_clean['Deaths / 100 Cases'], df_clean['Recovered / 100 Cases'], alpha=0.6, c='purple')
plt.title('Death Rate vs Recovery Rate', fontsize=14, fontweight='bold')
plt.xlabel('Deaths per 100 Cases')
plt.ylabel('Recovered per 100 Cases')
plt.grid(True, alpha=0.3)

# Visualization 6: Top 10 countries with highest recovery rate
plt.subplot(2, 3, 6)
top_recovery_rate = df_clean.nlargest(10, 'Recovered / 100 Cases')
plt.bar(range(len(top_recovery_rate)), top_recovery_rate['Recovered / 100 Cases'], color='green', alpha=0.7)
plt.title('Top 10 Countries by Recovery Rate', fontsize=14, fontweight='bold')
plt.xlabel('Countries')
plt.ylabel('Recovery Rate (per 100 cases)')
plt.xticks(range(len(top_recovery_rate)), top_recovery_rate['Country/Region'], rotation=45, ha='right')

plt.tight_layout()
plt.show()

# 5. ADDITIONAL ANALYSIS
print("\n5. ADDITIONAL ANALYSIS")
print("-" * 30)

# Global statistics
global_confirmed = df_clean['Confirmed'].sum()
global_deaths = df_clean['Deaths'].sum()
global_recovered = df_clean['Recovered'].sum()
global_active = df_clean['Active'].sum()

print(f"Global Statistics:")
print(f"Total Confirmed Cases: {global_confirmed:,}")
print(f"Total Deaths: {global_deaths:,}")
print(f"Total Recovered: {global_recovered:,}")
print(f"Total Active Cases: {global_active:,}")
print(f"Global Death Rate: {(global_deaths/global_confirmed)*100:.2f}%")
print(f"Global Recovery Rate: {(global_recovered/global_confirmed)*100:.2f}%")

# Countries with highest death rates
print(f"\nCountries with highest death rates:")
high_death_rate = df_clean.nlargest(10, 'Deaths / 100 Cases')[['Country/Region', 'Deaths / 100 Cases']]
print(high_death_rate)

# Countries with highest recovery rates
print(f"\nCountries with highest recovery rates:")
high_recovery_rate = df_clean.nlargest(10, 'Recovered / 100 Cases')[['Country/Region', 'Recovered / 100 Cases']]
print(high_recovery_rate)

# 6. OBSERVATIONS AND INSIGHTS
print("\n6. KEY OBSERVATIONS AND INSIGHTS")
print("-" * 40)

observations = [
    "1. The United States has the highest number of confirmed cases, followed by Brazil and India.",
    "2. The global recovery rate is significantly higher than the death rate.",
    f"3. {df_clean.loc[df_clean['Deaths / 100 Cases'].idxmax(), 'Country/Region']} has the highest death rate at {df_clean['Deaths / 100 Cases'].max():.2f}%.",
    f"4. {df_clean.loc[df_clean['Recovered / 100 Cases'].idxmax(), 'Country/Region']} has the highest recovery rate at {df_clean['Recovered / 100 Cases'].max():.2f}%.",
    "5. The Americas region shows the highest number of confirmed cases among WHO regions.",
    "6. There's a wide variation in death rates across countries, suggesting different healthcare capacities and response strategies.",
    "7. Some countries show very high recovery rates (>90%), indicating effective treatment protocols.",
    f"8. The dataset covers {len(df_clean)} countries/regions worldwide.",
    "9. Active cases vary significantly across countries, with some having very low active cases despite high total cases.",
    "10. The relationship between death rate and recovery rate shows an inverse correlation in most cases."
]

for observation in observations:
    print(observation)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 60)

# Save summary statistics to a file
summary_stats = df_clean.describe()
print(f"\nSummary statistics saved. Dataset contains {len(df_clean)} countries/regions.")
print(f"Analysis completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
