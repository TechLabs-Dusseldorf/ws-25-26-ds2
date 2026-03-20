import pandas as pd
from src.utils import logging
from src.config import core_vars


"""
Data Loading and First Exploration:
    - Load the dataset 
    - Display dataset structure and size
    - Display the first 5 rows and check the data types of all columns.
    - Missing values.
"""

def inspect_data(filepath):
    df = pd.read_csv(filepath)

    # Clean column names first
    df.columns = df.columns.str.replace("^avg_", "", regex=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    logging.info("Dataset structure:")
    df.info()

    logging.info("Descriptive statistics:\n%s", df.describe())

    logging.info("First look at the dataset:\n%s", df.head())

    logging.info("Dataset shape: %s rows × %s columns", df.shape[0], df.shape[1])

    logging.info("Data types:\n%s", df.dtypes)

    logging.info("Checking for NaN values:\n%s", df.isna().sum())

    logging.info("Number of countries: %s", df["country"].nunique())

    logging.info("Countries represented in the dataset:")
    for country in df["country"].unique():
        logging.info(country)

    return df

""" 
Wet-bulb temperature values appear inconsistent with physical expectations.
According to the dataset documentation, this variable is recorded in degrees Celsius.
However, descriptive statistics and hourly profiles suggest values that are unusually high 
relative to dry-bulb temperature.

Since wet-bulb temperature should generally be less than or equal to air temperature,
we perform diagnostic checks to verify whether the variable may actually be recorded in Fahrenheit.

The following tests evaluate this hypothesis. 
"""

def ensure_wetbulb_in_celsius(df, threshold=5.0, verbose=True):
    # Compute percentage of violations
    violation_rate = (df["wetbulb_temperature"] > df["temperature"]).mean() * 100

    if verbose:
        logging.info(f"Wetbulb > Temperature in {violation_rate:.2f}% of observations.")

    # Convert if violation rate suggests wrong units
    if violation_rate > threshold:
        if verbose:
            logging.info("Detected likely Fahrenheit scale. Converting to Celsius.")
        df["wetbulb_temperature"] = (df["wetbulb_temperature"] - 32) * 5 / 9
    else:
        if verbose:
            logging.info("Wetbulb values appear consistent with Celsius scale.")

    return violation_rate

"""

## Outliers identification with IQR rule (Interquartile Range method) ##
# - understand whether extreme values exist
# - check if they are rare and plausible
# - flag variables that may need special care later

"""


def identify_outliers_iqr(df, verbose=True):
    outlier_summary = []

    for col in core_vars:

        if col not in df.columns:
            continue

        series = df[col].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1 # iqr = interquantile range method. It capture typical variability

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr # everything below hte lower bound or above the upper one has to be treated as an outlier

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

    if verbose:
        logging.info("\nOutlier detection (IQR method):")
        logging.info(outlier_df)

    return outlier_df