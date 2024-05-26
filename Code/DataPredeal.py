import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
import numpy as np


def normalize_and_interpolate_excel(file_path, output_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Perform min-max normalization on each column
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Define all years
    all_years = [f'fidx10NN{year}' for year in range(1988, 2015)]

    # Create a DataFrame with all years and merge existing data
    df_full = pd.DataFrame(index=df_normalized.index, columns=all_years)
    df_full.update(df_normalized)

    # Perform cubic spline interpolation for each row
    df_interpolated = pd.DataFrame(index=df_full.index, columns=df_full.columns)
    for index, row in df_full.iterrows():
        x_existing = np.arange(len(row))[~row.isna()]
        y_existing = row.dropna().values

        # Create a cubic spline interpolation function
        cs = CubicSpline(x_existing, y_existing)

        # Generate interpolation results
        y_interpolated = cs(np.arange(len(row)))

        df_interpolated.loc[index] = y_interpolated

    # Save the normalized and interpolated data to a new Excel file
    df_interpolated.to_excel(output_path, index=False)



input_file = '' # Input xlsx file path
output_file = ''  # Output xlsx file path
normalize_and_interpolate_excel(input_file, output_file)
