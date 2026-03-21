# Results

## Project goal

This project investigates how climatic conditions influence **Water Usage Effectiveness (WUE)** in African data centers.  
The goal is to identify the main climate drivers of WUE and evaluate whether their impact varies across climate regions.

---

## Why this matters

Data centers rely on cooling systems, which are strongly affected by environmental conditions.  
In water-constrained regions, inefficient cooling increases resource consumption.

Understanding the relationship between climate and WUE helps:
- identify suitable locations for data centers  
- assess environmental risks  
- improve sustainable infrastructure planning  

---

## Data overview

The dataset contains **382,968 observations** across multiple African countries and climate regions.

It includes:
- **WUE (target variable)**
- Climate variables:
  - temperature  
  - humidity  
  - wet-bulb temperature  
  - wind speed  
  - precipitation  
- Temporal features (date, hour)
- Geographic information (country, climate region)

---

## Data validation and preprocessing

### Wet-bulb temperature consistency

Wet-bulb temperature values appeared inconsistent with physical expectations.  
Since wet-bulb temperature should not exceed air temperature, a diagnostic check was performed.

A high violation rate suggested the variable was recorded in **Fahrenheit instead of Celsius**, and values were converted accordingly.

---

### Outlier detection

Outliers were identified using the **IQR method**.  
Extreme values were present but represented a small proportion of the data and were considered plausible.

---

## Descriptive analysis

### Temporal coverage

Compute and plot temporal coverage (in days) per country.

Goal: understand the observations time span. 
- Do all countries have the same observation period?

![Country coverage](figures/country_coverage.png)

Countries differ in observation periods.  
Some countries (Algeria, Morocco, South Africa, Tunisia and Libya) have close to **two years of data**, while others have around **one year**, indicating an unbalanced dataset.

---

### Variable distributions

Distribution check. 
- Purpose: understand shape (skewness), spread and extreme values

![Distribution of variables](figures/distribution_check.png)

Climate variables show:
- skewed distributions  
- heavy tails  
- presence of extreme values  

This supports the inclusion of nonlinear terms in the analysis.

---

### Hourly climate profiles

Temporal validation of climate variables.
- Variables behaving smoothly over time?
- Diurnal patterns make physical sense?

Goal: Confirms data behaves reasonably before comparisons.

![Hourly climate profiles](figures/climate_temporal_validation.png)

Hourly patterns are smooth and consistent, indicating:
- realistic temporal behavior  
- no aggregation issues  

---

## Regional analysis

### Climate comparison across regions

Goal: Identify systematic geographic differences.
- Do climate regions differ systematically in their climate conditions and in WUE?

Explore spatial patterns:
- Compare mean and distribution of dependent variable across climate regions (Desert, Savanna, Rainforest...).
- Identify regions that tend to exhibit higher or lower WUE.

![Regional heatmap](figures/heatmap_region_means_standardized.png)

Climate conditions are generally more intense in Rainforest region, with consistently higher values across variables, while Mediterranean region exhibit comparatively lower levels.

---

### WUE distribution by region

![WUE by region](figures/boxplot_wue_region.png)

WUE differs across climate regions, indicating that environmental conditions might influence cooling efficiency.

Regions with extreme heat (Desert) or high humidity (Rainforest) tend to show less favorable WUE values.

---

## Climate sensitivity

Goal: Examine relationships between climate and WUE.

Analyze how WUE varies with physical climate drivers:
- Explore relationships between WUE and climate variables.
- Use scatter plots and correlation measures to assess directional patterns.

![Climate sensitivity heatmap](figures/climate_wue_matrix.png)

Temperature and humidity are the dominant drivers of WUE.

Their influence varies by region:
- stronger temperature effects in Mediterranean and Steppe climates  
- stronger humidity effects in Desert region

The analysis indicates a potential multicollinearity issue among climate variables. 
Wet-bulb temperature shows an extremely high correlation with WUE (0.97–0.99) across all climate regions. 
This may partly reflect the fact that wet-bulb temperature is derived from temperature and humidity, making these variables related.

---

## Climate gradients

This plot shows how WUE changes along climate conditions.
Each point represents a country, positioned by its average climate values.
It helps visualize whether countries in wetter or warmer climates tend to have higher WUE.
For example, rainforest countries may appear in the high-humidity, high-WUE area, while Mediterranean countries may appear in the lower-humidity, lower-WUE area.

In this way, the plot complements the correlation heatmap by showing where countries actually lie in the climate space, not just the numerical correlations


![WUE vs climate variables](figures/wue_climate_gradients.png)

Scatterplots confirm:
- a clear relationship between WUE and climate variables  
- nonlinear patterns, especially for temperature and humidity  

---

## Regression analysis

After exploring climate patterns and regional differences, the next step is to quantify how strongly climate conditions influence Water Usage Effectiveness (WUE).

The goal of the regression analysis is not only to confirm that relationships exist, but to answer:

Which climate variables have the strongest impact on WUE?
Are these relationships linear or nonlinear?
Do these effects differ across climate regions?
Understanding these relationships helps assess how sensitive data center cooling efficiency is to environmental conditions and which factors matter most in practice.

### Climate variable correlation

Before proceeding with regression analysis, we examine the correlations among climate variables themselves to assess the degree of multicollinearity and ensure more reliable model estimation. 



![Climate variable correlation](figures/climate_variables_correlation.png)

Climate variables such as temperature and humidty show some degree of correlation with wetbulb temperature. To avoid multicollinearity issues, wetbulb temperature will be excluded from the regression model.

---

### Residual diagnostics

![Residual histogram](figures/lr_wue_residuals_histogram.png)

Residuals are approximately normally distributed.

![Residuals vs predictions](figures/residuals_vs_predictions_lr_wue.png)

Residual patterns indicate nonlinear relationships, suggesting that simple linear models are not sufficient.

Quadratic terms for temperature and humidity are included to capture potential nonlinear effects, as extreme climate conditions may impact cooling efficiency differently than moderate ones.

By running the regression separately for each climate region, we aim to identify regional differences in climate sensitivity and understand whether certain environments are more favorable for efficient operation.


![Regression by region](figures/regression_actual_vs_predicted_by_region.png)
![Regression coefficients](figures/regression_coefficients_by_region.png)

Regression results show:
- explanatory power of climate variables  
- influence of temperature and humidity  
- evidence of nonlinear effects  

Regional regressions confirm that climate sensitivity varies across regions.

---

## Main findings

- **Temperature and humidity are the main drivers of WUE**
- Climate effects are **nonlinear**, especially at extreme values  
- The relationship between climate and WUE varies across regions
  
Mediterranean and Steppe regions offer the most favorable conditions for water-efficient data center operation, due to their combination of moderate temperatures and lower humidity. In contrast, Rainforest and Desert regions present the greatest challenges, driven by persistent humidity and extreme heat, respectively.  


---

## Conclusion

Climate plays a critical role in determining water efficiency in data centers.

These results highlight the importance of:
- considering regional climate differences  
- accounting for nonlinear environmental effects  
- selecting locations with favorable climate conditions  

Overall, the analysis shows that **moderate and stable climates are most suitable for water-efficient data center operation**.
