from pathlib import Path


# Base paths

base_dir = Path(__file__).resolve().parent.parent

data_dir = base_dir / "data"
figures_dir = base_dir / "figures"
final_figures_dir = figures_dir / "final"


# Data file

data_path = data_dir / "query-water-efficiency-data.csv"

"""

Grouping related variables into lists.
This makes the code easier to reuse later.
Instead of repeatedly writing column names everywhere we can refers to them as groups.

"""

wue_var = [
    "wue_fixed",
]

climate_vars = [
    "temperature",
    "humidity",
    "wetbulb_temperature",
    "wind_speed",
    "precipitation",
]

energy_vars = [
    "total_fossil_twh",
    "total_renewables_twh",
    "total_energy_twh",
]

core_vars = wue_var + climate_vars # The "core_vars" list includes those variables that will later be used in the regression