''' Primary libraries to import '''

import numpy as np # for numerical operations
import pandas as pd # for data loading, grouping and aggregation
import matplotlib.pyplot as plt # basic plots
# import seaborn as sns # nicer visuals

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


'''
Data Loading and Initial Exploration:
    - Load the dataset 
    - Display dataset structure and size
    - Display the first 5 rows and check the data types of all columns.
    - Missing values.
'''

# Loading the dataset into a pandas DataFrame

df = pd.read_csv("query-water-efficiency-data (1).csv")


# Create a function which store and display information related to dataset inspection

def inspect_data(filepath):
    
    df = pd.read_csv(filepath)

    print("\nFirst look at the dataset:")
    print(df.head())
    print("\nDataset shape:")
    print(f"{df.shape[0]} rows × {df.shape[1]} columns")
    print("\nData types:")
    print(df.dtypes)
    print("\nChecking for NaN values:")
    print(df.isna().sum())
    print(f"\nNumber of countries: {df['country'].nunique()}")

    countries_list = df["country"].unique()
    print("\nCountries represented in the dataset:")
    for country in countries_list:
        print(country)

    return df


# Diplay the function

df = inspect_data("query-water-efficiency-data (1).csv")

# Standardize variable names by removing the avg_ prefix to improve readability

df.columns = df.columns.str.replace("^avg_", "", regex=True) # Remove avg_
df.columns


# Coverage days by country to understand the time span of observations per country 

df["date"] = pd.to_datetime(df["date"]) # datetime object
coverage = (
    df.groupby("country")["date"]
      .agg(start_date="min", end_date="max") # group by country and extract date range
)
coverage["coverage_days"] = (
    coverage["end_date"] - coverage["start_date"]
).dt.days + 1 # compute numbers of days covered
coverage.sort_values("coverage_days", ascending=False) # sort them by coverage lentgh

print(coverage)

# Bar chart to diplay date coverage distribution in a more visual way

coverage["coverage_days"].plot(
    kind="bar",
    figsize=(12, 4)
)

plt.ylabel("Number of days covered")
plt.xlabel("Country")
plt.title("Temporal Data Coverage by Country")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


"""
- Variable-Level Descriptive Analysis
    - Understand each variable independently before making any comparisons.

Focus on basic statistics, distribution and outliers detection

"""

## 1st: Define the CORE variables ##
# Why? Make the analysis repruducible 

# Main outcome variables
wue_vars = [
    "wue_fixed",
    "wue_indirect"
    ]

# Climate variables
climate_vars = [
    "temperature",
    "humidity",
    "wetbulb_temperature",
    "wind_speed",
    "precipitation"
]

# Energy variables (contextual)
energy_vars = [
    "total_fossil_twh",
    "total_renewables_twh",
    "total_energy_twh"
]

core_vars = wue_vars + climate_vars + energy_vars


## Full descriptive statistics ##

df[core_vars].describe().T

# Summary table

summary_stats = pd.DataFrame({
    "mean": df[core_vars].mean(),
    "median": df[core_vars].median(),
    "std": df[core_vars].std(),
    "min": df[core_vars].min(),
    "max": df[core_vars].max()
})

summary_stats



## Outliers identification ##
# - understand whether extreme values exist
# - check if they are rare and plausible
# - flag variables that may need special care later

outlier_summary = []

for col in core_vars:
    series = df[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1 # iqr = interquantile range method. It capture typical variability

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr # everything below hte lower bound or above the upper one has to be treated as an outlier.

    n_outliers = ((series < lower) | (series > upper)).sum() # counts how many observations fall outside the bounds
    pct_outliers = n_outliers / len(series) * 100 # % of the data they represent

    outlier_summary.append({
        "variable": col,
        "lower_bound": lower,
        "upper_bound": upper,
        "n_outliers": n_outliers,
        "pct_outliers": pct_outliers
    })

outlier_df = pd.DataFrame(outlier_summary)
outlier_df



# Outlier analysis shows that most variables have a small and expected proportion of extreme values. 
# Higher outlier shares in precipitation and energy variables reflect distributional characteristics and country-level heterogeneity rather than data quality issues. 
# No observations were removed at this stage

# - Distribution check 
# Purpose: understand shape (skewness), spread, and extreme values

dist_table = df[core_vars].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
dist_table["skew"] = df[core_vars].skew(numeric_only=True)
dist_table[["mean","std","min","1%","5%","50%","95%","99%","max","skew"]].round(4)

# Histograms: visual check for skewness, spread
import matplotlib.pyplot as plt

for col in core_vars:
    plt.figure()
    df[col].hist(bins=60)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()
    





