import numpy as np
import pandas as pd
from src.utils import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from src.config import figures_dir

"""  
Correlation analysis indicates a potential multicollinearity issue among climate variables. 
Wet-bulb temperature shows an extremely high correlation with WUE (0.97–0.99) across all climate regions. 
This may partly reflect the fact that wet-bulb temperature is derived from temperature and humidity, making these variables related.

Before proceeding with regression analysis, we examine the correlations among climate variables themselves to assess the degree of multicollinearity and ensure more reliable model estimation. 
"""

def check_climate_variable_correlation(df, climate_vars, plot=True, verbose=True):

    # Compute correlation matrix
    climate_corr = df[climate_vars].corr()

    if verbose:
        logging.info("\nCorrelation between climate variables:")
        logging.info(climate_corr)

    if plot:
        plt.figure(figsize=(6, 5))

        sns.heatmap(
            climate_corr,
            annot=True,
            cmap="coolwarm",
            center=0
        )

        plt.title("Correlation Between Climate Variables")

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()

        plt.savefig(
            figures_dir / "climate_variables_correlation.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    return climate_corr


"""
Regression analysis. The first model will be run excluding "wetbulb_temperature" from the independent variables 
to avoid multicollinearity with "temeprature" and "humidity". We first check for residuals distribution.

"""
def run_regression_residuals(df):
    # Add quadratic terms
    df["wetbulb_temperature_sq"] = df["wetbulb_temperature"] ** 2
    df["temperature_sq"] = df["temperature"] ** 2
    df["humidity_sq"] = df["humidity"] ** 2

    y = df["wue_fixed"]
    X = df[[
        "temperature",
        "humidity",
        "temperature_sq",
        "humidity_sq",
        "wind_speed",
        "precipitation"
    ]]

    data = pd.concat([X, y], axis=1).dropna()

    X = data[[
        "temperature",
        "humidity",
        "temperature_sq",
        "humidity_sq",
        "wind_speed",
        "precipitation"
    ]]
    y = data["wue_fixed"]

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.title("Histogram of Residuals")
    plt.tight_layout()
    plt.savefig(figures_dir / "lr_wue_residuals_histogram.png", dpi=300, bbox_inches="tight")
    plt.close()

    """ 
    Residuals are normally distributed, meaning the distribution of errors is approximately normal.
    We will now test for constant variance

    """
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        x=predictions,
        y=residuals,
        alpha=0.4,
        ax=ax
    )

    ax.set(
        xlabel="Predicted WUE",
        ylabel="Residuals",
        title="Residuals vs Predicted WUE"
    )

    ax.axhline(
        y=0,
        color="black",
        linestyle="--"
    )

    plt.tight_layout()
    plt.savefig(figures_dir / "residuals_vs_predictions_lr_wue.png", dpi=300, bbox_inches="tight")
    plt.close()


""" 
Residual diagnostics reveal a clear nonlinear pattern between predicted values and residuals, 
indicating that the relationship between climate conditions and cooling efficiency cannot be fully captured by a simple linear specification. 
This suggests that nonlinear climate effects, particularly related to temperature and humidity, may influence data center cooling efficiency.

"""

def run_wue_regression_by_region(df):
    # To Make sure quadratic terms exist
    df["temperature_sq"] = df["temperature"] ** 2
    df["humidity_sq"] = df["humidity"] ** 2

    regions = df["climate_region"].unique()
    n_regions = len(regions)

    n_cols = 3
    n_rows = (n_regions + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # If there is only one row/one plot, make axes always iterable
    if n_regions == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    last_i = -1

    for i, (region, data_region) in enumerate(df.groupby("climate_region")):
        last_i = i

        logging.info("\n==============================")
        logging.info(f"CLIMATE REGION: {region}")
        logging.info("==============================")

        X = data_region[
            [
                "temperature",
                "humidity",
                "temperature_sq",
                "humidity_sq",
                "wind_speed",
                "precipitation",
            ]
        ]

        y = data_region["wue_fixed"]

        data = pd.concat([X, y], axis=1).dropna()
        X = data[X.columns]
        y = data["wue_fixed"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        logging.info("EVALUATION METRICS")
        logging.info(f"R2: {r2:.4f}")
        logging.info(f"MSE: {mse:.4f}")
        logging.info(f"RMSE: {rmse:.4f}")

        coeff_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        })

        logging.info("\nFeature Coefficients:")
        logging.info("\n%s", coeff_df)

        ax = axes[i]
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            color="red",
            linestyle="--"
        )
        ax.set_title(f"{region} (R² = {r2:.2f})")
        ax.set_xlabel("Actual WUE")
        ax.set_ylabel("Predicted WUE")

    # Remove empty subplot panels
    for j in range(last_i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Regression Fit: Actual vs Predicted by Climate Region", y=1.02)
    plt.tight_layout()
    plt.savefig(
        figures_dir / "regression_actual_vs_predicted_by_region.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()