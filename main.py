''' Primary libraries to import '''

import numpy as np # for numerical operations
import pandas as pd # for data loading, grouping and aggregation
import matplotlib.pyplot as plt # basic plots


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


# Display the function

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


""" 
    Wet-bulb temperature values appear inconsistent with physical expectations.
# According to the dataset documentation, this variable is recorded in degrees Celsius.
# However, descriptive statistics and hourly profiles suggest values that are unusually high 
# relative to dry-bulb temperature.
#
# Since wet-bulb temperature should generally be less than or equal to air temperature,
# we perform diagnostic checks to verify whether the variable may actually be recorded in Fahrenheit.
#
# The following tests evaluate this hypothesis. 

"""


(df["wetbulb_temperature"] > df["temperature"]).mean() * 100 # Wet-bulb results to be always higher than temperature (dry-bulb). This is not supposed to happen.

df[["temperature", "wetbulb_temperature"]].describe() # Compare magnitudes. Strongly suggesting a °F scale.

df["wetbulb_temperature"] = (df["wetbulb_temperature"] - 32) * 5/9 # Proceed to convert the variable in Fahrenheit

df[["temperature", "wetbulb_temperature"]].describe() # Now the variable values look more plausible.

df["wetbulb_temperature"].describe()  

# Recompute the summary stat

summary_stats = pd.DataFrame({
    "mean": df[core_vars].mean(),
    "median": df[core_vars].median(),
    "std": df[core_vars].std(),
    "min": df[core_vars].min(),
    "max": df[core_vars].max()
})

print(summary_stats)



""" 
    Outliers identification 
# - understand whether extreme values exist
# - check if they are rare and plausible
# - flag variables that may need special care later

"""

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

for col in core_vars:
    plt.figure()
    df[col].hist(bins=60)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()
    


"""
- Temporal validation
    -Variables behaving smoothly over time?
    -Diurnal patterns make physical sense?
    -No strange jumps or aggregation artifacts exist

"""

## Compute Hourly Means

hourly_profile = (
    df.groupby("hour")[climate_vars].mean()
)

hourly_profile

hourly_profile.plot(subplots=True, figsize=(10,8), sharex=True) # Using subplots since variables have different scales

plt.suptitle("Hourly Mean Profiles — Climate Variables")
plt.xticks(range(0,24))
plt.tight_layout()
plt.show()

# The dataset exhibits coherent temporal dynamics consistent with physical expectations, supporting its suitability for comparative and modeling analyses.

"""
Climate Region Structure and Comparison

Goal: Identify systematic geographic differences.
We then explore spatial patterns:
• Compare mean and distribution of avg_wue_fixed across climate regions (Desert, Savanna, Rainforest...).
• Identify regions that tend to exhibit higher or lower water usage efficiency.
Results remain descriptive and comparative.

"""

#Climate Region Structure and Comparison
"""
Goal: Identify systematic geographic differences.
We explore spatial patterns by:
- validating country-to-region assignments
- comparing climate conditions across regions
- comparing countries within the same region
- comparing water usage efficiency across regions

"""
# ------------------------------
# 1) Country -> Climate Region mapping (data quality check)
# ------------------------------
# Ensure each country belongs to only one climate region

country_region_map = (
    df[["country", "climate_region"]]
    .drop_duplicates()
    .sort_values(["climate_region", "country"])
    .reset_index(drop=True)
)

# Start display index from 1 (presentation-friendly)
country_region_map.index = country_region_map.index + 1

print("Country -> Climate Region mapping:")
print(country_region_map)

# Check if any country appears in more than one climate region
region_count_per_country = df.groupby("country")["climate_region"].nunique()
multi_region_countries = region_count_per_country[region_count_per_country > 1]

if len(multi_region_countries) == 0:
    print("\nData quality check passed: each country belongs to exactly one climate region.")
else:
    print("\nWarning: some countries appear in multiple climate regions:")
    print(multi_region_countries)


# ------------------------------
# 2) Descriptive statistics for climate variables by region
# ------------------------------
# Compare temperature, humidity, wet-bulb temperature, wind speed, precipitation

region_climate_stats = (
    df.groupby("climate_region")[climate_vars]
    .agg(["mean", "median", "std", "min", "max"])
    .round(2)
)

print("\nDescriptive statistics for climate variables by climate region:")
print(region_climate_stats)


# ------------------------------
# 3) Region ranking by average climate values (all observations)
# ------------------------------
# Which region is hotter / windier / more humid?

region_means = (
    df.groupby("climate_region")[climate_vars]
    .mean()
    .round(2)
)

print("\nMean climate values by climate region:")
print(region_means)

for var in climate_vars:
    print(f"\nRanking by average {var} (highest to lowest):")
    print(region_means[var].sort_values(ascending=False))


# ------------------------------
# 4) Country-level means (equal-weight country comparison)
# ------------------------------
# This avoids over-weighting countries with more observations

country_climate_means = (
    df.groupby(["climate_region", "country"])[climate_vars]
    .mean()
    .reset_index()
    .round(2)
)

# Start display index from 1 (presentation-friendly)
country_climate_means.index = country_climate_means.index + 1

print("\nCountry-level climate means (first rows):")
print(country_climate_means.head())

equal_weight_region_means = (
    country_climate_means.groupby("climate_region")[climate_vars]
    .mean()
    .round(2)
)

print("\nEqual-weight mean climate values by region (recommended for country comparison):")
print(equal_weight_region_means)

for var in climate_vars:
    print(f"\nEqual-weight ranking by {var} (highest to lowest):")
    print(equal_weight_region_means[var].sort_values(ascending=False))

"""
Climate Region Structure and Comparison

Goal: Identify systematic geographic differences.
We then explore spatial patterns:
• Compare mean and distribution of avg_wue_fixed across climate regions (Desert, Savanna, Rainforest...).
• Identify regions that tend to exhibit higher or lower water usage efficiency.
Results remain descriptive and comparative.
"""

# ------------------------------
# 5) Differences between countries within each climate region
# ------------------------------
# Compare each country to its regional average (based on country means)

region_avg_for_country = (
    country_climate_means.groupby("climate_region")[climate_vars]
    .transform("mean")
)

country_vs_region = country_climate_means.copy()

for var in climate_vars:
    country_vs_region[f"{var}_diff_vs_region"] = (
        country_climate_means[var] - region_avg_for_country[var]
    ).round(2)

print("\nCountry differences from regional averages (first rows):")
print(country_vs_region.head(10))


# ------------------------------
# 6) Descriptive comparison of water usage efficiency by region
# ------------------------------
# Descriptive only (no causal interpretation)

wue_var = "wue_fixed"

region_wue_stats = (
    df.groupby("climate_region")[wue_var]
    .agg(["mean", "median", "std", "min", "max"])
    .round(4)
    .sort_values("mean", ascending=False)
)

print(f"\nDescriptive statistics for {wue_var} by climate region:")
print(region_wue_stats)

region_wue_percentiles = (
    df.groupby("climate_region")[wue_var]
    .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    .round(4)
)

print(f"\nPercentile summary for {wue_var} by climate region:")
print(region_wue_percentiles)


# ------------------------------
# 7) Visual comparison: average climate by region (equal-weight)
# ------------------------------

for var in climate_vars:
    plt.figure(figsize=(8, 4))
    equal_weight_region_means[var].sort_values(ascending=False).plot(kind="bar")
    plt.title(f"Average {var} by Climate Region (Equal-Weight Countries)")
    plt.ylabel(var)
    plt.xlabel("Climate Region")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ------------------------------
# 8) Visual comparison: WUE distribution across regions
# ------------------------------

plt.figure(figsize=(10, 5))
df.boxplot(column=wue_var, by="climate_region", grid=False)
plt.title(f"Distribution of {wue_var} Across Climate Regions")
plt.suptitle("")
plt.xlabel("Climate Region")
plt.ylabel(wue_var)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()