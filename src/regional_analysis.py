import pandas as pd
from src.utils import logging
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import figures_dir

"""

Climate Region Structure and Comparison

Goal: Identify systematic geographic differences.
- Do climate regions differ systematically in their climate conditions and in WUE?

We then explore spatial patterns:
• Compare mean and distribution of avg_wue_fixed across climate regions (Desert, Savanna, Rainforest...).
• Identify regions that tend to exhibit higher or lower water usage effectiveness.
Results remain descriptive and comparative.

"""

def analyze_climate_regions(df, climate_vars, wue_var="wue_fixed", verbose=True):

    # 1) Country -> Climate Region mapping check
    country_region_map = (
        df[["country", "climate_region"]]
        .drop_duplicates()
        .sort_values(["climate_region", "country"])
        .reset_index(drop=True)
    )

    region_count_per_country = df.groupby("country")["climate_region"].nunique() # how many distinct climate regions each country has in the dataset
    multi_region_countries = region_count_per_country[region_count_per_country > 1] # each country should belong to exactly one climate region

    # 2) Equal-weight climate comparison
    country_climate_means = (
        df.groupby(["climate_region", "country"])[climate_vars]
        .mean()
        .reset_index()
        .round(2)
    )

    equal_weight_region_means = (
        country_climate_means.groupby("climate_region")[climate_vars]
        .mean()
        .round(2)
    )

    # 3) WUE comparison by region
    region_wue_stats = (
        df.groupby("climate_region")[wue_var]
        .agg(["mean", "median", "std", "min", "max"])
        .round(4)
        .sort_values("mean", ascending=False)
    )

    results = {
        "country_region_map": country_region_map,
        "multi_region_countries": multi_region_countries,
        "equal_weight_region_means": equal_weight_region_means,
        "region_wue_stats": region_wue_stats,
    }

    if verbose:
        logging.info("Country -> Climate Region mapping:")
        logging.info(country_region_map)

        if len(multi_region_countries) == 0:
            logging.info("\nData quality check passed: each country belongs to exactly one climate region.")
        else:
            logging.info("\nWarning: some countries appear in multiple climate regions:")
            logging.info(multi_region_countries)

        logging.info("\nEqual-weight mean climate values by region:")
        logging.info(equal_weight_region_means)

        logging.info(f"\nDescriptive statistics for {wue_var} by climate region:")
        logging.info(region_wue_stats)

    return results

# Outputs visualization
def plot_climate_region_results(df, results, wue_var="wue_fixed"):

    equal_weight_region_means = results["equal_weight_region_means"]

    # Standardize climate means across regions for visual comparability
    standardized_region_means = (
        equal_weight_region_means - equal_weight_region_means.mean()
    ) / equal_weight_region_means.std()

    # Heatmap of climate means by region
    plt.figure(figsize=(10, 5))
    sns.heatmap(
        standardized_region_means,
        annot=equal_weight_region_means,
        cmap="coolwarm",
        fmt=".2f",
        center=0,
        cbar_kws={"label": "Standardized value (z-score)"}
    )
    plt.title("Climate Region Comparison (Standardized Heatmap)")
    plt.ylabel("Climate Region")
    plt.xlabel("Climate Variable")
    plt.tight_layout()
    plt.savefig(figures_dir / "heatmap_region_means_standardized.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Boxplot of WUE by region
    plt.figure(figsize=(10, 5))
    df.boxplot(column=wue_var, by="climate_region", grid=False)
    plt.title(f"Distribution of {wue_var} Across Climate Regions")
    plt.suptitle("")
    plt.xlabel("Climate Region")
    plt.ylabel(wue_var)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(figures_dir / "boxplot_wue_region.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_climate_region_analysis(df, climate_vars, wue_var="wue_fixed", plot=True, verbose=True):

    results = analyze_climate_regions(
        df=df,
        climate_vars=climate_vars,
        wue_var=wue_var,
        verbose=verbose
    )

    if plot:
        plot_climate_region_results(
            df=df,
            results=results,
            wue_var=wue_var
        )

    return results


"""

Climate Sensitivity Exploration.

Purpose: Examine relationships between climate and WUE.
We analyze how WUE varies with physical climate drivers:
• Explore relationships between wue_fixed and climate_vars.
• Use scatter plots and correlation measures to assess directional patterns.
This phase still motivates later modeling and does not asserting causality.

"""

def analyze_climate_sensitivity(df, climate_vars, wue_var="wue_fixed", verbose=True):
    """
    Compute within-region correlations between WUE and climate variables.
    """

    rows = []

    for region, group in df.groupby("climate_region"):
        corr_matrix = group[climate_vars + [wue_var]].corr()
        wue_correlations = corr_matrix[wue_var].drop(wue_var)

        row = {"climate_region": region}
        for var in climate_vars:
            row[var] = wue_correlations[var]

        rows.append(row)

    results_df = pd.DataFrame(rows).set_index("climate_region")

    if verbose:
        logging.info("\nClimate sensitivity of WUE by region:")
        logging.info(results_df)

    return {"correlation_matrix": results_df}


def plot_climate_correlation_heatmap(results):

    corr_matrix = results["correlation_matrix"]

    plt.figure(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Climate Sensitivity of WUE by Region")
    plt.ylabel("Climate Region")
    plt.xlabel("Climate Variable")
    plt.tight_layout()
    plt.savefig(figures_dir / "climate_wue_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()


"""
This plot shows how WUE changes along climate conditions.
Each point represents a country, positioned by its average climate values.
It helps visualize whether countries in wetter or warmer climates tend to have higher WUE.
For example, rainforest countries may appear in the high-humidity, high-WUE area, while Mediterranean countries may appear in the lower-humidity, lower-WUE area.

In this way, the plot complements the correlation heatmap by showing where countries actually lie in the climate space, not just the numerical correlations

"""

def plot_wue_climate_gradients(df, wue_var="wue_fixed"):

    # Country averages
    country_means = (
        df.groupby(["country", "climate_region"])[
            ["temperature", "humidity", wue_var]
        ]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, climate_var in zip(axes, ["humidity", "temperature"]):
        sns.scatterplot(
            data=country_means,
            x=climate_var,
            y=wue_var,
            hue="climate_region",
            palette="tab10",
            s=90,
            ax=ax
        )

        ax.set_xlabel(climate_var.capitalize())
        ax.set_ylabel("WUE")
        ax.set_title(f"WUE vs {climate_var.capitalize()}")

    plt.tight_layout()
    plt.savefig(figures_dir / "wue_climate_gradients.png", dpi=300, bbox_inches="tight")
    plt.close()


# Wrapper function that runs the full sensitivity step. It Run both:
   # 1. correlation heatmap by region
   # 2. scatterplots of WUE vs climate variables
def run_climate_sensitivity_analysis(df, climate_vars, wue_var="wue_fixed", plot=True, verbose=True):

    results = analyze_climate_sensitivity(
        df=df,
        climate_vars=climate_vars,
        wue_var=wue_var,
        verbose=verbose
    )

    if plot:
        plot_climate_correlation_heatmap(results)
        plot_wue_climate_gradients(df, wue_var=wue_var)

    return results