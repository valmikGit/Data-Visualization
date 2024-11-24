# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA  # For time series forecasting (ARIMA)
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Google-Playstore-Preprocessed.csv')

# Convert 'Released' and 'Last Updated' to datetime
df['Released'] = pd.to_datetime(df['Released'])
df['Last Updated'] = pd.to_datetime(df['Last Updated'])

# Calculate the total installs (you can combine 'Minimum Installs' and 'Maximum Installs' if necessary)
df['Total Installs'] = df['Minimum Installs'].astype(int) + (df['Maximum Installs'].astype(int) - df['Minimum Installs'].astype(int)) // 2

# Handle missing values, if any
df.fillna(0, inplace=True)

# --- Visualization: Line Charts for Install and Rating Trends ---
def plot_install_rating_trends():
    # Grouping by date to observe trends over time
    df_grouped = df.groupby(df['Released']).agg({'Total Installs': 'sum', 'Rating': 'mean'}).reset_index()

    # Plotting installs and ratings
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Line chart for installs
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Installs', color='tab:blue')
    ax1.plot(df_grouped['Released'], df_grouped['Total Installs'], color='tab:blue', label='Total Installs')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Creating a second y-axis for Rating
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Rating', color='tab:green')
    ax2.plot(df_grouped['Released'], df_grouped['Rating'], color='tab:green', label='Rating')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title('App Growth: Installs and Ratings Over Time')
    plt.show()

# --- Visualization: Stacked Area Chart for App Growth by Category ---
def plot_stacked_area_chart():
    # Grouping by category and date to create stacked area chart
    df_category_grouped = df.groupby([df['Released'], 'Category']).agg({'Total Installs': 'sum'}).reset_index()

    # Pivoting for stacked area chart
    df_pivot = df_category_grouped.pivot_table(index='Released', columns='Category', values='Total Installs', aggfunc='sum').fillna(0)

    # Breaking the plot into subplots for each category
    categories = df_pivot.columns
    num_categories = len(categories)
    num_plots = 5  # Show 5 categories per plot for better readability

    for i in range(0, num_categories, num_plots):
        # Select a subset of categories for this plot
        subset_categories = categories[i:i + num_plots]
        df_subset = df_pivot[subset_categories]

        fig, ax = plt.subplots(figsize=(10, 6))  # Adjusting the figure size
        df_subset.plot.area(ax=ax, stacked=True, cmap="Set2")

        # Set the title and labels
        ax.set_title(f'App Growth by Category ({i+1} to {i+num_plots})', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Total Installs', fontsize=12)

        # Adjusting the legend position and font size
        ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()  # Ensures the plot fits within the figure area
        plt.show()

# --- Trend Analysis & Time Series Forecasting (ARIMA) ---
def forecast_arima(df_arima):
    # ARIMA model requires the data to be stationary; first, we check for stationarity

    # Differencing the data (if necessary) to make it stationary
    df_arima_diff = df_arima.diff().dropna()  # Differencing to make it stationary

    # Fit an ARIMA model (using parameters (p=1, d=1, q=1) for simplicity, but you can optimize these)
    model_arima = ARIMA(df_arima_diff, order=(1, 1, 1))
    model_arima_fit = model_arima.fit()

    # Make predictions (forecast next 180 days)
    forecast_arima = model_arima_fit.forecast(steps=180)

    # Convert forecast back to the original scale by adding the differenced values
    forecast_arima_cumsum = forecast_arima.cumsum() + df_arima.iloc[-1]['Total Installs']

    # Ensure forecast_dates and forecast_arima_cumsum have the same length
    forecast_arima_cumsum = forecast_arima_cumsum[:180]  # Ensure it's exactly 180 values

    # Create a date range for the forecasted period (180 days)
    forecast_dates = pd.date_range(df_arima.index[-1] + pd.Timedelta(days=1), periods=180, freq='D')

    # Plot the forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(df_arima.index, df_arima['Total Installs'], label='Historical Installs')
    plt.plot(forecast_dates, forecast_arima_cumsum, label='Forecasted Installs', color='orange')
    plt.title('App Growth Forecast (ARIMA)')
    plt.xlabel('Date')
    plt.ylabel('Total Installs')
    plt.legend()
    plt.show()

# --- Feedback Loop 1: Refining Forecast Based on User Input ---
def feedback_loop_1(df_grouped):
    # User interaction: Get new data from the user
    new_installs = int(input("Enter the new total installs (e.g., 750000): "))
    new_date = input("Enter the release date for this data (YYYY-MM-DD): ")
    new_data = {'Released': pd.to_datetime(new_date), 'Total Installs': new_installs}  # User inputs the new data
    new_df = pd.DataFrame([new_data])

    # Append new data to the original dataframe and re-train the model
    df_grouped_updated = pd.concat([df_grouped, new_df], ignore_index=True)
    df_arima_updated = df_grouped_updated[['Released', 'Total Installs']].set_index('Released')

    # Retrain the ARIMA model with updated data
    forecast_arima(df_arima_updated)

    return df_grouped_updated

# --- Feedback Loop 2: Re-forecasting Based on App Updates ---
def feedback_loop_2(df_grouped):
    # Simulate feedback based on app updates (User interaction for updates)
    update_installs = int(input("Enter the new install count after the app update: "))
    update_date = input("Enter the date of the update (YYYY-MM-DD): ")
    update_data = {'Released': pd.to_datetime(update_date), 'Total Installs': update_installs}
    df_updated = pd.concat([df_grouped, pd.DataFrame([update_data])], ignore_index=True)

    # Re-train the ARIMA model with the updated data
    df_arima_update = df_updated[['Released', 'Total Installs']].set_index('Released')
    forecast_arima(df_arima_update)

    return df_updated

# Run the program: Display visualizations and run the feedback loops
plot_install_rating_trends()
plot_stacked_area_chart()

# Prepare the data for ARIMA
df_grouped = df.groupby(df['Released']).agg({'Total Installs': 'sum', 'Rating': 'mean'}).reset_index()

# Run feedback loops and update data
df_grouped = feedback_loop_1(df_grouped)
df_grouped = feedback_loop_2(df_grouped)

print("Program execution completed.")