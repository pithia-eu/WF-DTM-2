import os
import subprocess
import uuid
from datetime import datetime, timedelta
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi.responses import FileResponse
import matplotlib.pyplot as plt

description = """
To compare observed atmospheric densities from the Starlette satellite with modeled densities from DTM2020 for a specified time interval.
"""

tags_metadata = [
    {
        "name": "Run Workflow",
        "description": "Return the comparison of observed atmospheric densities from the Starlette satellite with modeled densities from DTM2020 for a specified time interval.",
    },
]

DATA_URL="https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
AP_TO_KP = {
    0:0,
    2:0.33,
    3:0.66,
    4:1,
    5:1.33,
    6:1.66,
    7:2,
    9:2.33,
    12:2.66,
    15:3,
    18:3.33,
    22:3.66,
    27:4,
    32:4.33,
    39:4.66,
    48:5,
    56:5.33,
    67:5.66,
    80:6,
    94:6.33,
    111:6.66,
    132:7,
    154:7.33,
    179:7.66,
    207:8,
    236:8.33,
    300:8.66,
    400:9,
}

# Get the full path to the directory containing the FastAPI script
script_dir = os.path.dirname(os.path.abspath(__file__))
# The path to the directory containing the runs input files
runs_dir = os.path.join(script_dir, 'runs')
# Starlette data file is in the script directory Starlette_2000-2023_PITHIA.dat
starlette_data_file = os.path.join(script_dir, 'Starlette_2000-2023_PITHIA.dat')
dtm_model_executable = os.path.join(script_dir, 'Model_DTM2020F107Kp_1p')

# Create the runs directory if it does not exist
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)
# The path to the directory containing the output files
output_dir = os.path.join(script_dir, 'output')
# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

app = FastAPI(title='DTM2020 Workflow 2 API',
              description=description,
              version="1.0",
              openapi_tags=tags_metadata
              )

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def plot_starlette_data(data, output_dir, filename, start_date, end_date):
    max_density = max(data["Observed_density"].max(), data["DTM2020_Density"].max()) * 1e17
    min_ratio = min(data["Observed_to_Modeled_Ratio"].min(), data["Observed_to_Modeled_Ratio"].min()*1e17, data["Observed_to_Modeled_Ratio"].min()*1e17)*0.9
    # Create a combined plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot densities on the left y-axis
    ax1.plot(data["timestamp"], data["Observed_density"], label="Observed Density", linestyle='-', marker='o',
             color="blue")
    ax1.plot(data["timestamp"], data["DTM2020_Density"], label="DTM2020 Density", linestyle='-', marker='o',
             color="green")
    ax1.set_xlabel(f"Timeseries from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    ax1.set_ylabel("Density (kg/m^3)", color="blue")
    ax1.legend(loc="upper left")
    ax1.tick_params(axis='y', labelcolor="blue")
    # Set the minimum y-axis value to 0, but no maximum value
    ax1.set_ylim(min_ratio/1e17, max_density/1e17*1.1)

    # Create a second y-axis for ratios
    ax2 = ax1.twinx()
    ax2.plot(data["timestamp"], data["Observed_to_Modeled_Ratio"], label="Observed-to-Modeled Ratio", linestyle='--',
             color="orange")
    ax2.axhline(y=1, color='grey', linestyle='-', label="Perfect Agreement (y=1)")
    ax2.set_ylabel("Observed-to-Modeled Ratio", color="orange")
    ax2.legend(loc="upper right")
    ax2.tick_params(axis='y', labelcolor="orange")
    ax2.set_ylim(min_ratio, max_density*1.1)  # Fixing the range for the second y-axis

    # Set the x-axis labels to be the dates YYYY-MM-DD, don't display the time
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

    # Title and grid
    plt.title("Timeseries of Observed and DTM2020 Densities with Observed-to-Modeled Ratio")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the plot to a file
    plot_file = os.path.join(output_dir, filename)
    plt.savefig(plot_file)
    plt.close()

@app.get("/run_workflow", tags=["Run Workflow"])
async def run_workflow(
        start_date: str = Query(..., description="Start Date in the format 'YYYY-MM-DD', e.g. 2024-01-01."),
        end_date: str = Query(..., description="End Date in the format 'YYYY-MM-DD', e.g. 2024-01-02.")):
    exe_start_time = datetime.now()
    # Validate the start and end dates, the format should be 'YYYY-MM-DD' and the end date should be greater than the start date
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        if start_date > end_date:
            raise HTTPException(status_code=400, detail="End Date should be greater than Start Date.")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Please use the format 'YYYY-MM-DD'.")
    # Use UUID to generate a unique identifier for the run
    run_id = str(uuid.uuid4())
    # Construct the filename for the run, using the start and end dates, e.g. '2024-01-01_2024-01-02.input'
    run_filename = f'{run_id}.input'
    run_filepath = os.path.join(runs_dir, run_filename)

    # Get the fm, f1, akp1, akp3 values for the run
    # Load the data from the URL
    df = pd.read_csv(DATA_URL, skiprows=40, sep='\s+', names=["Year", "Month", "Day", "Days", "Days_M", "Bsr", "dB",
                        "Kp1", "Kp2", "Kp3", "Kp4", "Kp5", "Kp6", "Kp7", "Kp8",
                        "ap1", "ap2", "ap3", "ap4", "ap5", "ap6", "ap7", "ap8",
                        "Ap", "SN", "F10.7obs", "F10.7adj", "D"])
    # Add the date index to the dataframe
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    # Filter the dataframe based on the start and end dates, keeping only the relevant columns, from previous 80 days of start date to end date (inclusive)
    previous_81_days = (start_date - timedelta(days=80)).strftime('%Y-%m-%d')
    df = df[(df['Date'] >= previous_81_days) & (df['Date'] <= end_date.strftime('%Y-%m-%d'))]
    # For each day from the start date to the end date, calculate the average of the ap values for the previous 80 days
    ap_values = {}
    for date in pd.date_range(start=start_date, end=end_date):
        previous_80_days = (date - timedelta(days=80)).strftime('%Y-%m-%d')
        previous_1_day = (date - timedelta(days=1)).strftime('%Y-%m-%d')
        # fm is the average of the F10.7obs values for the previous 80 days + the current day
        fm = df[(df['Date'] >= previous_80_days) & (df['Date'] <= date)]['F10.7obs'].mean()
        # f1 is the value of the F10.7obs for the previous day
        f1 = df[df['Date'] == previous_1_day]['F10.7obs'].values[0]
        akp3 = {}
        for i in range (0, 9):
            # AP_TO_KP[min(AP_TO_KP.keys(), key=lambda x:abs(x-previous_1_day_ap))]
            # We need to get the last 24 hours of ap values
            # if i == 0: get the previous day's ap value ap8, ap7, ap6, ap5, ap4, ap3, ap2, ap1
            # if i == 1: get the previous day's ap value ap8, ap7, ap6, ap5, ap4, ap3, ap2, ap1, and the current day's ap1
            # if i == 2: get the previous day's ap value ap8, ap7, ap6, ap5, ap4, ap3, ap2, ap1, and the current day's ap1, ap2
            # ... and so on, and calculate the average ap value
            prev_ap_values = []
            for j in range(8, i, -1):
                prev_ap_values.append(df[df['Date'] == previous_1_day][f'ap{j}'].values[0])
            if i > 0:
                for k in range(1, i+1):
                    prev_ap_values.append(df[df['Date'] == date.strftime('%Y-%m-%d')][f'ap{k}'].values[0])
            # print("AP Values:", ap_values)
            mean_ap = sum(prev_ap_values) / len(prev_ap_values)
            akp3[i] = AP_TO_KP[min(AP_TO_KP.keys(), key=lambda x:abs(x-mean_ap))]
        akp1 = {}
        akp1[0] = df[df['Date'] == previous_1_day]['Kp8'].values[0]
        for i in range(1, 8):
            akp1[i] = df[df['Date'] == date][f'Kp{i}'].values[0]
        # print(akp3)
        # print(akp1)
        ap_values[date.strftime('%Y-%m-%d')] = {
            'fm': fm,
            'f1': f1,
            'akp3': akp3,
            'akp1': akp1
        }
    # print("AP Values:", ap_values)
    # Load the Starlette data, which has no header and is separated by whitespace
    # The columns are: Year, Month, Day, decimal day-of-year, altitude (km), local solar time (hr), latitude (deg), longitude (deg), observed density (g/cm3)
    # e.g. 2000  1  2    2.04514 805.980  10.86  -2.25  147.68  0.186E-16
    starlette_df = pd.read_csv(starlette_data_file, sep='\s+', header=None, names=["Year", "Month", "Day", "Decimal_day", "Altitude", "Solar_time", "Latitude", "Longitude", "Observed_density"])
    # Create a new index from the year month and day columns, called Date
    starlette_df['Date'] = pd.to_datetime(starlette_df[['Year', 'Month', 'Day']])
    # Filter the starlette dataframe based on the start and end dates
    starlette_df = starlette_df[(starlette_df['Date'] >= start_date) & (starlette_df['Date'] <= end_date)]
    # Drop the Date column
    starlette_df.drop(columns=['Date'], inplace=True)
    # Add the datetime index to the starlette dataframe, considering the "Year","Month","Day" and Hour, Minute, Second from "Decimal_day" columns, precision of 1 second
    # The decimal day is the day of the year with the time as a decimal, e.g. 2.04514 is the 2nd day of the year at 1:05:30
    # Get the timestamp from the decimal day of the year
    starlette_df['timestamp'] = pd.to_datetime(starlette_df['Year'].astype(str) + '-' + starlette_df['Month'].astype(str) + '-' + starlette_df['Day'].astype(str) + ' ' + (starlette_df['Decimal_day'] % 1 * 24).astype(int).astype(str) + ':' + ((starlette_df['Decimal_day'] % 1 * 24 * 60) % 60).astype(int).astype(str) + ':' + (((starlette_df['Decimal_day'] % 1 * 24 * 60) % 60 * 60) % 60).astype(int).astype(str))

    # Loop through the starlette data, calculate the hour from the decimal day, and get the fm, f1, akp1 (for that hour), akp3 values
    starlette_data = []
    # Add a new column for the density
    starlette_df['DTM2020_Density'] = 0.0
    # Add a new column for the observed to modeled density ratio
    starlette_df['Observed_to_Modeled_Ratio'] = 0.0
    # Also add the fm, f1, akp1, akp3 columns to the starlette data
    starlette_df['fm'] = 0.0
    starlette_df['f1'] = 0.0
    starlette_df['akp1'] = 0.0
    starlette_df['akp3'] = 0.0
    runs = []
    for index, row in starlette_df.iterrows():
        # Get the hour from the decimal day
        hour = (row["Decimal_day"] % 1) * 24
        hour = round(hour, 2)
        # Every 3 hours, the akp1 value changes, check which hour interval the current hour falls into.  Get the integer value of the hour/3
        hour = int(hour/3)
        # Get the fm, f1, akp1, akp3 values for the row
        date = datetime(year=int(row['Year']), month=int(row['Month']), day=int(row['Day']))
        fm, f1, akp1, akp3 = ap_values[date.strftime('%Y-%m-%d')]['fm'], ap_values[date.strftime('%Y-%m-%d')]['f1'], ap_values[date.strftime('%Y-%m-%d')]['akp1'][int(hour)], ap_values[date.strftime('%Y-%m-%d')]['akp3'][int(hour)]
        starlette_data.append([row['Year'], row['Month'], row['Day'], int(hour), row['Altitude'], row['Solar_time'], row['Latitude'], row['Longitude'], row['Observed_density'], fm, f1, akp1, akp3])
        input_string = f"""{row["Decimal_day"]} {round(fm,3)} {f1} {akp1} {akp3} {row["Altitude"]} {row["Solar_time"]} {row["Latitude"]} {row["Longitude"]}"""
        runs.append((index, input_string))
        # Write the input string to the run file
        with open(run_filepath, 'a') as f:
            # Clear the file before writing
            f.truncate(0)
            f.write(input_string)
        # Run the Model_DTM2020F107Kp_1p, command is 'Model_DTM2020F107Kp_1p < run_file', and print the output
        # Run it as subprocess
        dtm_model_command = f'{dtm_model_executable} < {run_filepath}'
        # Get the stdout
        stdout = subprocess.run(dtm_model_command, shell=True, capture_output=True).stdout
        # Output example: 'sol_DTM2020_F107_Kp_iterT  (SWAMI Operational Model. Drivers: F107 and Kp)      , facteur mutiplicat\n\n  1.673184426124853E-017\n'
        # Get the density value from the stdout, e.g. 1.673184426124853E-017
        density = float(stdout.split()[-1])
        # Update the density value in the starlette dataframe
        starlette_df.at[index, 'DTM2020_Density'] = density
        # Calculate the observed to modeled density ratio
        ratio = row['Observed_density'] / density
        # Update the ratio value in the starlette dataframe
        starlette_df.at[index, 'Observed_to_Modeled_Ratio'] = ratio
        # Update the fm, f1, akp1, akp3 values in the starlette dataframe
        starlette_df.at[index, 'fm'] = fm
        starlette_df.at[index, 'f1'] = f1
        starlette_df.at[index, 'akp1'] = akp1
        starlette_df.at[index, 'akp3'] = akp3

    # Plot the data
    plot_starlette_data(starlette_df, output_dir, f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}_starlette_plot.png', start_date, end_date)
    starlette_plot_file = os.path.join(output_dir, f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}_starlette_plot.png')

    # Remove the "timestamp" column from the starlette dataframe
    starlette_df.drop(columns=['timestamp'], inplace=True)
    # Save the updated starlette dataframe to a new file
    starlette_output_file = os.path.join(output_dir, f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}_starlette_output.dat')
    starlette_df.to_csv(starlette_output_file, sep=' ', index=False)
    # Save the runs data to a new file
    runs_output_file = os.path.join(output_dir, f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}_runs_output.dat')
    with open(runs_output_file, 'w') as f:
        for index, input_string in runs:
            f.write(f'{input_string}\n')
    # Delete the run file
    os.remove(run_filepath)
    # Return the starlette data that allows the user to download the file directly. Stream the file to the user
    exe_end_time = datetime.now()
    exe_time = exe_end_time - exe_start_time

    # Zip the output files and plot, and return the zip file
    zip_file = os.path.join(output_dir, f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}_output.zip')
    # Zip the output files
    subprocess.run(['zip', '-j', zip_file, starlette_output_file, starlette_plot_file])
    # Delete the individual files
    os.remove(starlette_output_file)
    os.remove(runs_output_file)
    os.remove(starlette_plot_file)
    print(f"Execution time: {exe_time}")
    return FileResponse(zip_file, filename=f'{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}_output.zip', media_type='application/octet-stream')
