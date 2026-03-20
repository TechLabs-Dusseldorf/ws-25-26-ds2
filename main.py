from src.utils import setup_logging
import logging
from src.config import data_path, climate_vars
from src.preprocessing import inspect_data, ensure_wetbulb_in_celsius, identify_outliers_iqr
from src.descriptive_analysis import (
        analyze_country_coverage, 
        compute_core_descriptive_stats, 
        analyze_variable_distributions, 
        analyze_hourly_climate_profiles
)
from src.regional_analysis import run_climate_region_analysis, run_climate_sensitivity_analysis
from src.regression_analysis import (
        check_climate_variable_correlation,
        run_regression_residuals,
        run_wue_regression_by_region,
 )


def main():
    setup_logging()
    logging.info("Starting analysis...")

    # Data loading & preprocessing

    df = inspect_data(data_path)

    ensure_wetbulb_in_celsius(df, threshold=5.0, verbose=True)
    outlier_df = identify_outliers_iqr(df, verbose=True)

    # Descriptive analysis

    coverage = analyze_country_coverage(df, plot=True)
    summary_stats, full_describe = compute_core_descriptive_stats(df, verbose=True)
    dist_table = analyze_variable_distributions(df, plot=True, verbose=True)
    hourly_profile = analyze_hourly_climate_profiles(df, plot=True, verbose=True)

    # Regional analysis

    region_results = run_climate_region_analysis(
        df,
        climate_vars=climate_vars,
        wue_var="wue_fixed",
        plot=True,
        verbose=True
    )
    sensitivity_results = run_climate_sensitivity_analysis(
        df,
        climate_vars=climate_vars,
        wue_var="wue_fixed",
        plot=True,
        verbose=True
    )

    # Regression analysis

    climate_corr = check_climate_variable_correlation(
        df,
        climate_vars=climate_vars,
        plot=True,
        verbose=True
    )
    run_regression_residuals(df)
    run_wue_regression_by_region(df)

if __name__ == "__main__":
    main()