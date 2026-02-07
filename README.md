## Welcome Team 2 to your Project Phase

# 🌍 Water Usage Efficiency of Data Centers in Africa

## 📖 Overview
This project analyzes how **climate conditions affect onsite water usage efficiency (WUE)** of data centers across **African climate regions**.  
The goal is to provide data-driven insights to support **sustainable data center planning and policy decisions**.

---

## 👥 Who We Are
We are a **sustainability consultancy** supporting governments and institutions in assessing **data center efficiency**.

**Mission:**  
Use **data science** to evaluate how **climate conditions impact data center sustainability**.

---

## 🎯 Research Question
> How do climate conditions influence onsite data center water usage efficiency across African climate zones?

---

## 📊 Data & Variables

### DataSet
https://huggingface.co/datasets/PengfeiLi/WaterEfficientDatasetForAfricanCountries

### Outcome Variable
- `WUE_FixedApproachDirect (L/kWh)` — onsite water usage efficiency

### Climate Predictors
- `temperature` (°C)
- `humidity` (%)
- `wetbulb_temperature` (°C)
- `precipitation` (mm)
- `wind_speed` (m/s)
- `climate_region` (e.g. Desert, Rainforest)

---

## 🔄 Data Aggregation
Original data is highly granular (**city × hour**).  
To improve usability and performance, data is aggregated to:

- **Country level**
- **Monthly frequency**

`city` is used only for aggregation and then dropped.

---

## 🧠 Analysis Plan
- Data cleaning and harmonization  
- Descriptive statistics and distributions  
- Comparisons across climate regions  
- Visualizations of climate vs. WUE  
- **Bonus:** simple regression linking climate variables to WUE  

---

## ⚠️ Challenges & Solutions

### Challenges
- Large dataset size  
- High temporal and spatial granularity  
- Heterogeneous data structure  

### Solutions
- Careful variable selection  
- Aggregation to **country × month**  
- Unified data format  

---

## ✍️ Authors
- Adrian Maloku  
- Luca Pozzi  
- Memduh Talha Köksal  
- Negin Jaraei  

 