 
import xarray as xr
import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
import numpy as np
from os import listdir
import os
from os.path import isfile, join

# Load as an xarray Dataset, making sure the filename is correct
# xr_data = xr.open_dataset('Assignment 2\Part 1 sea surfaced data\bi_2007.nc')
xr_data = xr.open_dataset('bi_2007.nc', engine='netcdf4')

# Inspect the xarray dataset structure
print(xr_data)

 
files = [f for f in listdir(".") if isfile(join(".", f))]

 
variable_names = {
    "bi" : "burning_index_g",
    "fm1000" : "dead_fuel_moisture_1000hr", 
    "rmax" : "relative_humidity",
    "fm100" : "dead_fuel_moisture_100hr",
    "pr" : "precipitation_amount",
    "vpd" : "mean_vapor_pressure_deficit",
    "sph" : "specific_humidity",
    "tmmx" : "air_temperature",
    "pet" : "potential_evapotranspiration",
    "etr" : "potential_evapotranspiration",
    "tmmn" : "air_temperature",
    "rmin" : "relative_humidity",
    "srad" : "surface_downwelling_shortwave_flux_in_air",
}

 
datasets = []
for file in files:
    if file.endswith(".nc"):
        key = file.split("_")[0]
        ds = xr.open_dataset(file, engine='netcdf4').rename({variable_names[key] : key})
        datasets.append(ds)

 
dates = ["2007-06-12", "2007-06-17", "2007-07-17", "2007-07-18", "2007-07-19", "2007-07-27", "2007-07-28", "2007-08-03", "2007-08-08", "2007-08-17"]

 
len(datasets)

 
merged_ds = xr.merge(datasets)

 
merged_ds

 
def standardize_data(data_array):
    """Standardizes the data by subtracting the mean and dividing by the standard deviation."""
    data_array_copied = data_array
    return (data_array_copied - data_array_copied.mean()) / data_array_copied.std()

 
def construct_variable_datsets(date):
    variable_datasets = {}
    for variable in variable_names.keys():
        variable_datasets[variable] = merged_ds[variable].sel(day=date)
    return variable_datasets

 
import ipywidgets as widgets
from IPython.display import display

def create_Visualizations(date) -> None:
    variable_ds = construct_variable_datsets(date=date)

    # Drought index
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['drought_index'] = (1 - standardize_data(variable_ds['pr']))/(1 + standardize_data(variable_ds['pet']) + standardize_data(variable_ds['vpd']))

    variable_ds['drought_index'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs={'label': 'Drought Index'}
    )

    ax.set_title("Drought Index")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    
    folder_name = "Drought Index"
    filename = f"{folder_name}/drought_index_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Water balance index
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['water_balance_index'] = standardize_data(variable_ds['pr']) - standardize_data(variable_ds['pet'])

    variable_ds['water_balance_index'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Water balance index'}
    )

    ax.set_title("Water balance index")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Water Balance Index"
    filename = f"{folder_name}/water_balance_index_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Fire Danger Index
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['fdi'] = (standardize_data(variable_ds['tmmx']) * (100 - standardize_data(variable_ds['rmax'])))/(standardize_data(variable_ds['fm1000']))

    variable_ds['fdi'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Fire Danger Index'}
    )

    ax.set_title("Fire Danger index")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Fire Danger Index"
    filename = f"{folder_name}/fire_danger_index_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Potential Evapotranspiration Deficit
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['pde'] = standardize_data(variable_ds['pet']) - standardize_data(variable_ds['pr'])

    variable_ds['pde'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Potential Evapotranspiration Deficit'}
    )

    ax.set_title("Potential Evapotranspiration Deficit")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Potential Evapotranspiration Deficit"
    filename = f"{folder_name}/potential_evapotranspiration_deficit_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Vapour pressure deficit to fuel moisture ratio

    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['vpd_fm'] = standardize_data(variable_ds['vpd'])/standardize_data(variable_ds['fm1000'])

    variable_ds['vpd_fm'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Vapour pressure deficit to fuel moisture ratio'}
    )

    ax.set_title("Vapour pressure deficit to fuel moisture ratio")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Vapour Pressure Deficit To Fuel Moisture Ratio"
    filename = f"{folder_name}/fuel_moisture_deficit_to_fuel_moisture_ratio_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Energy Balance Index
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['ebi'] = standardize_data(variable_ds['srad']) * standardize_data(variable_ds['tmmx'])/1000

    variable_ds['ebi'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Energy Balance Index'}
    )

    ax.set_title("Energy Balance Index")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Energy Balance Index"
    filename = f"{folder_name}/energy_balance_index_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Thermal Stress Index
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['tsi'] = standardize_data(variable_ds['tmmx']) - standardize_data(variable_ds['tmmn'])

    variable_ds['tsi'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Thermal Stress Index'}
    )

    ax.set_title("Thermal Stress Index")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Thermal Stress Index"
    filename = f"{folder_name}/thermal_stress_index_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")
     # plt.show()

    # Relative Humidity Stress Index
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['rhsi'] = standardize_data(variable_ds['rmax']) - standardize_data(variable_ds['rmin'])

    variable_ds['rhsi'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Relative Humidity Stress Index'}
    )

    ax.set_title("Relative Humidity Stress Index")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Relative Humidity Stress Index"
    filename = f"{folder_name}/relaticve_humidity_stress_index_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")
     # plt.show()

    # Precipitation Efficiency Index
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['pei'] = standardize_data(variable_ds['pr'])/(standardize_data(variable_ds['pet']) + standardize_data(variable_ds['vpd']))

    variable_ds['pei'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Precipitation Efficiency Index'}
    )

    ax.set_title("Precipitation Efficiency Index")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Precipitation Efficiency Index"
    filename = f"{folder_name}/precipitation_efficiency_index_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Evapotranspiration to precipitation ratio
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['etp'] = standardize_data(variable_ds['pet'])/standardize_data(variable_ds['pr'])

    variable_ds['etp'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Evapotranspiration to precipitation ratio'}
    )

    ax.set_title("Evapotranspiration to precipitation ratio")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Evapotranspiration To Precipitation Ratio"
    filename = f"{folder_name}/evapotranspiration_to_precipitation_ratio_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Shortwave Radiation Efficiency
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['sre'] = standardize_data(variable_ds['srad'])/standardize_data(variable_ds['pet'])

    variable_ds['sre'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Shortwave Radiation Efficiency'}
    )

    ax.set_title("Shortwave Radiation Efficiency")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Shortwave Radiation Efficiency"
    filename = f"{folder_name}/shortwave_radiation_efficiency_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Precipitation to Specific Humidity Ratio (P/Sph)
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['pshr'] = standardize_data(variable_ds['pr'])/standardize_data(variable_ds['sph'])

    variable_ds['pshr'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Precipitation to Specific Humidity Ratio'}
    )

    ax.set_title("Precipitation to Specific Humidity Ratio")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude") 

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Precipitation To Specific Humidity Ratio"
    filename = f"{folder_name}/precipitation_to_specific_humidity_ratio_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

    # Fuel Moisture to Precipitation Ratio (FM/P)
    fig, ax = plt.subplots(figsize=(10, 6))

    variable_ds['pshr'] = standardize_data(variable_ds['fm1000'])/standardize_data(variable_ds['pr'])

    variable_ds['pshr'].plot(
        ax=ax,
        cmap='inferno',
        cbar_kwargs = {'label': 'Fuel Moisture to Precipitation Ratio'}
    )

    ax.set_title("Fuel Moisture to Precipitation Ratio")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.suptitle(f'Sea surfaced data for {date}')

    plt.tight_layout()

    folder_name = "Fuel Moisture To Precipitation Ratio"
    filename = f"{folder_name}/fuel_moisture_to_precipitation_ratio_{date}.png"  # predefined naming convention

    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)
    
    # Save the figure
    fig.savefig(filename, dpi=300)
    print(f"Image saved as {filename}")

     # plt.show()

 
for date in dates:
    create_Visualizations(date=date)

# 1. Drought Index (or Drought Severity Index)
# Description: This index combines precipitation, evapotranspiration, and vapor pressure deficit to estimate drought severity. It reflects the balance between available water (precipitation) and demand (evapotranspiration and vapor pressure deficit).
# Usage: The drought index is useful for identifying areas experiencing drought conditions. It could be visualized as a heatmap for specific dates or over time.
# 2. Fire Danger Index (FDI)
# Description: The Fire Danger Index quantifies the risk of fire based on meteorological variables like temperature, fuel moisture, and relative humidity. It is a commonly used metric in wildfire forecasting.
# Usage: It could be visualized over time or spatially to assess fire risk in specific regions.
# 3. Potential Evapotranspiration Deficit (PED)
# Description: This feature measures the difference between potential evapotranspiration (PET) and the actual precipitation (pr). A higher deficit indicates that the environment is losing more water through evaporation than is replenished by precipitation.
# Usage: It could be used to highlight areas that are water-stressed or have insufficient precipitation relative to their evapotranspiration needs.
# 4. Vapor Pressure Deficit to Fuel Moisture Ratio (VPD/Fuel Moisture Ratio)
# Description: The ratio of vapor pressure deficit (vpd) to fuel moisture content (e.g., fm1000 or fm100) can provide insight into the potential for fire spread. High VPD and low fuel moisture typically indicate high fire danger.
# Usage: This ratio could be visualized to show areas at risk for large wildfires.
# 5. Energy Balance Index
# Description: This index gives an approximation of the energy available for evapotranspiration by considering radiation and temperature. It combines solar radiation and temperature to give an estimate of the energy available for plant growth or evaporation.
# Usage: The energy balance could help assess regions with the highest energy input for evaporation, potentially indicating areas where crops or vegetation might be stressed due to excessive evaporation.
# 6. Thermal Stress Index
# Description: Thermal stress on plants or crops can be derived by comparing the daily maximum temperature (tmmx) to the daily minimum temperature (tmmn). A large difference between the two could indicate significant thermal stress.
# Usage: This index could be visualized to show areas experiencing extreme temperature fluctuations, which may stress vegetation or crops.
# 7. Relative Humidity Stress Index
# Description: This index compares the difference between maximum (rmax) and minimum relative humidity (rmin) to determine the level of atmospheric moisture stress. A large difference indicates more atmospheric moisture variation, which may affect plant transpiration and potential fire risk.
# Usage: This index could be mapped spatially to show regions where the atmosphere is highly variable, affecting both vegetation stress and fire risk.
# 8. Precipitation Efficiency Index
# Description: The precipitation efficiency index assesses how efficiently precipitation is being converted into actual water storage, based on the relationship between precipitation and evapotranspiration. A higher ratio indicates more efficient water use.
# Usage: This index can be used to identify areas where precipitation is well-utilized, as well as areas where evaporation or vapor pressure deficit outweighs the benefit of rainfall.
# 9. Evapotranspiration to Precipitation Ratio (ET/P)
# Description: This ratio compares the potential evapotranspiration (pet) to the daily accumulated precipitation (pr) to identify areas where the demand for water exceeds supply.
# Usage: This ratio could highlight areas with high water demand (high evapotranspiration) compared to the available water (precipitation), which is useful for agricultural and water resource management.
# 10. Shortwave Radiation Efficiency
# Description: This feature measures how efficiently shortwave radiation (srad) is being used for energy (e.g., evaporation or plant growth). The higher the ratio, the more efficient the radiation is being used.
# Usage: This could be useful in identifying regions with high solar energy but low evapotranspiration efficiency, possibly indicating areas suitable for solar power generation or those with potential water stress.
# 11. Precipitation to Specific Humidity Ratio (P/Sph)
# Description: The ratio between precipitation (pr) and specific humidity (sph) could provide insight into how much water vapor is available in the atmosphere relative to the actual precipitation falling.
# Usage: This ratio might be used to understand atmospheric moisture levels in relation to the actual precipitation and help model weather patterns or predict the risk of drought.
# 12. Fuel Moisture to Precipitation Ratio (FM/P)
# Description: The ratio of fuel moisture (e.g., fm1000 for 1000-hour fuel moisture) to precipitation (pr) helps understand how quickly vegetation or fuel dries out relative to the precipitation received.
# Usage: This could help forecast fire behavior by examining how quickly fuels dry out compared to rainfall.

# Stories and inferences

# 1. Drought Index (Drought Severity Index)
# 
# Story: A drought index heatmap could reveal regions where water scarcity is at critical levels, providing an early warning for drought conditions. Areas in deeper colors may indicate extreme drought, where precipitation is significantly below evapotranspiration and vapor pressure demands.
# 
# Inference: Areas with high drought severity likely need water conservation measures or agricultural adjustments to cope with potential crop stress. Over time, tracking this index could highlight the persistence or intensification of droughts, particularly in arid or semi-arid zones, where this might signal a longer-term change in water availability.
# 
# 2. Fire Danger Index (FDI)
# 
# Story: The FDI map would show real-time or forecasted hotspots where conditions are ideal for wildfires. Higher indices suggest that vegetation or natural fuel is dry enough for ignition, with potential areas of extreme risk highlighted for resource allocation.
# 
# Inference: In fire-prone regions, a rising FDI could indicate the need for proactive fire prevention measures. Monitoring FDI over time can help assess whether fire risk is seasonally recurring or exacerbated by unusual weather conditions, allowing for more targeted interventions in high-risk areas.
# 
# 3. Potential Evapotranspiration Deficit (PED)
# 
# Story: Mapping the PED reveals regions experiencing water deficits, where evapotranspiration needs outpace the available precipitation. Higher values show areas under stress from insufficient rainfall relative to their environmental water loss.
# 
# Inference: Persistent high PED values may signal areas vulnerable to drought and may prompt water management strategies. This map could be crucial for agricultural planning, as high PED regions may require irrigation or water-conserving crop varieties to ensure productivity.
# 
# 4. Vapor Pressure Deficit to Fuel Moisture Ratio (VPD/Fuel Moisture Ratio)
# 
# Story: Visualizing this ratio could show regions with a high potential for fire spread due to low moisture content in vegetation combined with high atmospheric dryness.
# 
# Inference: In areas where VPD is high and fuel moisture is low, the probability of fire igniting and spreading increases, indicating a need for active monitoring and resource mobilization for wildfire prevention. Tracking this ratio could also reveal seasonal trends, helping to plan for times of heightened fire risk.
# 
# 5. Energy Balance Index
# 
# Story: An energy balance map reveals areas with high solar energy availability relative to temperature, which may correlate with high evapotranspiration rates.
# 
# Inference: Regions with high energy balance values may experience significant plant water loss due to evapotranspiration, potentially stressing crops or vegetation. Identifying these regions helps inform agricultural management practices like shading or irrigation, particularly in high-temperature zones.
# 
# 6. Thermal Stress Index
# 
# Story: The thermal stress index highlights regions experiencing significant daily temperature swings, with higher values in areas of extreme temperature variation.
# 
# Inference: Large daily temperature differences can stress vegetation and potentially lead to reduced plant growth or crop yields. This map could identify regions where crop selection and agricultural practices should adapt to mitigate thermal stress.
# 
# 7. Relative Humidity Stress Index
# 
# Story: This index would map areas where atmospheric moisture variation is high, indicating potential stress for plant life and higher fire risk in drier zones.
# 
# Inference: Regions with large fluctuations between maximum and minimum relative humidity might experience more intense plant transpiration cycles and increased fire susceptibility. This map could inform both agricultural practices and fire management efforts, particularly in semi-arid regions.
# 
# 8. Precipitation Efficiency Index
# 
# Story: The precipitation efficiency index shows how effectively precipitation translates into water storage, with higher values indicating efficient water usage.
# 
# Inference: Areas with low precipitation efficiency may experience frequent runoff or limited water retention, signaling the need for soil conservation practices. This index can also highlight areas with good water use efficiency, useful for locating regions ideal for water-intensive crops.
# 
# 9. Evapotranspiration to Precipitation Ratio (ET/P)
# 
# Story: Visualizing the ET/P ratio helps identify regions where water demand significantly exceeds precipitation, indicating water stress.
# 
# Inference: Regions with a high ET/P ratio may face agricultural water shortages and could benefit from water-conserving techniques. Over time, these areas might signal longer-term trends in water scarcity, useful for regional water resource planning and crop suitability studies.
# 
# 10. Shortwave Radiation Efficiency
# 
# Story: This map would highlight regions where solar radiation is used efficiently for plant growth and evaporation, with higher values indicating more productive use of available sunlight.
# 
# Inference: Regions with high radiation efficiency may indicate areas suitable for crop growth or solar energy installations. Conversely, low efficiency areas might suggest the need for irrigation or vegetation cover adjustments to optimize growth.
# 
# 11. Precipitation to Specific Humidity Ratio (P/Sph)
# 
# Story: This ratio shows how precipitation relates to atmospheric moisture content, with higher values indicating efficient moisture release.
# 
# Inference: Areas with high values could indicate regions where precipitation effectively reduces atmospheric dryness, possibly lowering drought risk. Low values, however, may signal inefficient moisture conversion, potentially leading to a water vapor surplus that doesn’t translate to rainfall.
# 
# 12. Fuel Moisture to Precipitation Ratio (FM/P)
# 
# Story: The FM/P ratio map would show regions where vegetation moisture rapidly dries out compared to recent precipitation.
# 
# Inference: High FM/P ratios suggest regions where fuel moisture decreases quickly relative to precipitation, signaling high wildfire risk. Tracking these areas over time helps inform fire management strategies, especially in drought-prone areas where dry spells are frequent.