import os
import csv
import netCDF4 as nc

# Folder path to search for .NC files
folder_path = r"C:\Users\might\Downloads\GUVI\netCDF files"

# Create a folder to store CSV files
output_folder = os.path.join(folder_path, "Auroral boundary LATITUDE and LONGITUDE")
os.makedirs(output_folder, exist_ok=True)

# Get a list of files with the .NC extension
file_list = [file for file in os.listdir(folder_path) if file.endswith(".NC")]

# Function to get value or empty string for a given index and data
def get_data(index, data):
    if index < len(data):
        return data[index]
    else:
        return ""

# Process each .NC file
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)

    # Open the netCDF file
    dataset = nc.Dataset(file_path)

    # Read the variables
    latitude_data = dataset.variables["NORTH_GEOGRAPHIC_LATITUDE"][:]
    longitude_data = dataset.variables["NORTH_GEOGRAPHIC_LONGITUDE"][:]
    polar_latitude_data = dataset.variables["NORTH_POLAR_GEOGRAPHIC_LATITUDE"][:]
    polar_longitude_data = dataset.variables["NORTH_POLAR_GEOGRAPHIC_LONGITUDE"][:]
    south_latitude_data = dataset.variables["SOUTH_GEOGRAPHIC_LATITUDE"][:]
    south_longitude_data = dataset.variables["SOUTH_GEOGRAPHIC_LONGITUDE"][:]
    south_polar_latitude_data = dataset.variables["SOUTH_POLAR_GEOGRAPHIC_LATITUDE"][:]
    south_polar_longitude_data = dataset.variables["SOUTH_POLAR_GEOGRAPHIC_LONGITUDE"][:]

    # Read the attribute "STOPPING_TIME"
    stopping_time = dataset.getncattr("STOPPING_TIME")

    # Create a CSV file for writing
    csv_file_name = f"Auroral boundary LATITUDE and LONGITUDE_{stopping_time}.csv"
    csv_file_path = os.path.join(output_folder, csv_file_name)

    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write the data to the CSV file
        writer.writerow(["Index", "NORTH_GEOGRAPHIC_LATITUDE", "NORTH_GEOGRAPHIC_LONGITUDE",
                         "NORTH_POLAR_GEOGRAPHIC_LATITUDE", "NORTH_POLAR_GEOGRAPHIC_LONGITUDE",
                         "SOUTH_GEOGRAPHIC_LATITUDE", "SOUTH_GEOGRAPHIC_LONGITUDE",
                         "SOUTH_POLAR_GEOGRAPHIC_LATITUDE", "SOUTH_POLAR_GEOGRAPHIC_LONGITUDE"])

        max_data_length = max(len(latitude_data), len(longitude_data), len(polar_latitude_data),
                              len(polar_longitude_data), len(south_latitude_data), len(south_longitude_data),
                              len(south_polar_latitude_data), len(south_polar_longitude_data))

        # Write the data row by row using the helper function
        for index in range(max_data_length):
            row = [index + 1]
            row.append(get_data(index, latitude_data))
            row.append(get_data(index, longitude_data))
            row.append(get_data(index, polar_latitude_data))
            row.append(get_data(index, polar_longitude_data))
            row.append(get_data(index, south_latitude_data))
            row.append(get_data(index, south_longitude_data))
            row.append(get_data(index, south_polar_latitude_data))
            row.append(get_data(index, south_polar_longitude_data))

            writer.writerow(row)

    # Close the netCDF file
    dataset.close()

print("Process completed successfully!")
