from src.utils import logging
import pandas as pd
import matplotlib.pyplot as plt
from src.config import core_vars, figures_dir, climate_vars


"""

Compute and plot temporal coverage (in days) per country.

Goal: understand the observations time span. 
- Do all countries have the same observation period?

"""
def analyze_country_coverage(df, plot=True):
    # Ensure datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Compute coverage per country
    coverage = (
        df.groupby("country")["date"]
        .agg(start_date="min", end_date="max")
    )

    # Calculate number of days
    coverage["coverage_days"] = (
        coverage["end_date"] - coverage["start_date"]
    ).dt.days + 1

    # Sort countries by observation period
    coverage = coverage.sort_values("coverage_days", ascending=False)

    if plot:
        coverage["coverage_days"].plot(
            kind="bar",
            figsize=(12, 4)
        )
        plt.ylabel("Number of days covered")
        plt.xlabel("Country")
        plt.title("Temporal Data Coverage by Country")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(figures_dir / "country_coverage.png", dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(coverage)

    return coverage


"""

Compute descriptive statistics for the core variables of the analysis (EDA), including water use effectiveness (dependent var) and key climate indicators (independent vars). 
They provide an overview of the central tendencies and variability of these variables across the dataset, 
helping identify outliers and assess the overall distribution of climate indicators.

"""

def compute_core_descriptive_stats(df, verbose=True):

    available_vars = []

    for v in core_vars:
        if v in df.columns:
            available_vars.append(v)
        else:
            logging.info(f"Variable is not a column of df: {v}")

    logging.info(f"These are the available vars: {available_vars}")

    # Overall statistics
    full_describe = df[available_vars].describe().T

    # Statistics by climate region
    summary_stats = (
        df.groupby("climate_region")[available_vars]
        .agg(["mean", "median", "std", "min", "max"])
    )

    if verbose:
        logging.info("\nCore variable descriptive statistics by climate region:")
        logging.info(summary_stats)

    return summary_stats, full_describe


"""

# - Distribution check 
# Purpose: understand shape (skewness), spread, and extreme values

"""
def analyze_variable_distributions(df, plot=True, verbose=True):

    available_vars = [v for v in core_vars if v in df.columns]

    # Distribution statistics
    dist_table = df[available_vars].describe(
        percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
    ).T

    dist_table["skew"] = df[available_vars].skew(numeric_only=True)

    dist_table = dist_table[
        ["mean", "std", "min", "1%", "5%", "50%", "95%", "99%", "max", "skew"]
    ].round(4)

    if verbose:
        logging.info("\nDistribution summary:")
        logging.info(dist_table)

    # Visualization
    if plot:
        df[available_vars].hist(
            bins=60,
            figsize=(12, 8),
            layout=(3, 3)
        )
        plt.suptitle("Distribution of Core Variables")
        plt.tight_layout()
        plt.savefig(figures_dir / "distribution_check.png", dpi=300, bbox_inches="tight")
        plt.close()

    return dist_table


"""

- Temporal validation of climate variables
    -Variables behaving smoothly over time?
    -Diurnal patterns make physical sense?
Goal: Confirms data behaves reasonably before comparisons.

"""

def analyze_hourly_climate_profiles(df, plot=True, verbose=True):

    available_vars = [v for v in climate_vars if v in df.columns]

    # Compute hourly means
    hourly_profile = df.groupby("hour")[available_vars].mean()

    if verbose:
        logging.info("\nHourly climate profiles:")
        logging.info(hourly_profile)

    # Visualization
    if plot:
        hourly_profile.plot(
            subplots=True,
            figsize=(10, 8),
            sharex=True
        )

        plt.suptitle("Hourly Mean Profiles — Climate Variables")
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig(figures_dir / "climate_temporal_validation.png", dpi=300, bbox_inches="tight")
        plt.close()

    return hourly_profile