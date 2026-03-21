## Welcome Team 2 to your Project Phase

# :droplet: Cooling the Cloud: How Climate shapes Water Usage Effectiveness (WUE) in African Data Centers 

## 📖 Overview
This project analyzes how **climate conditions affect onsite water usage effectiveness (WUE)** of data centers across **African climate regions**.  
The goal is to provide data-driven insights to support **sustainable data center planning and policy decisions**.

---

## 👥 Who We Are
We are a **sustainability non-profit consultancy** supporting governments and institutions in assessing **data center cooling efficiency**.

**Mission:**  
Use **data science** to evaluate how **climate conditions impact data center sustainability**.

---

## 🎯 Research Question
> How do climate conditions influence cooling efficiency in data centers across different African climate regions?


## 🗂️ Project Structure

- `src/` – contains all source code for the analysis  
  - `config.py` – defines file paths and variable groups used across the project  
  - `preprocessing.py` – data loading, validation and preprocessing steps  
  - `descriptive_analysis.py` – EDA and summary statistics  
  - `regional_analysis.py` – comparison of climate regions and sensitivity analysis  
  - `regression_analysis.py` – regression models and diagnostics  
  - `utils.py` – helper functions (e.g. logging setup)  

- `data/` – input dataset  

- `figures/` – generated plots saved during the analysis  

- `results.md` – summary of findings and visual outputs  

- `main.py` – main file that runs the full analysis pipeline  

- `requirements.txt` – project dependencies

## 🛠 Installation & Setup

### **System Requirements**
- Python 3.8+
- Git
- pip 
- *(Optional)* Virtual environment manager (`venv`, `virtualenv`, or `conda`)  

### **1. Clone the Repository**
```bash
git clone git@github.com:TechLabs-Dusseldorf/ws-25-26-ds2.git
cd ws-25-26-ds2
```

### **2. Create a virtual environment (optional but recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### **3. Install dependencies**

1. Ensure you're in the project root directory.  
2. Install required packages using:

```bash
pip install -r requirements.txt
```

### 4. Running the Python Scripts

1. Run the Python scripts using:

Run calculations:

```bash
python main.py 
```
---

## ✍️ Authors 
- Luca Pozzi  
- Negin Jaraei  
