import pandas as pd
from os import listdir

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

devX = find_csv_filenames('data/DevAttX', suffix=".csv")
devY = find_csv_filenames('data/DevAttY', suffix=".csv")

print(len(devX))
print(len(devY))



print(devX)

VPgeo_XFrame = pd.read_csv(f"data/DevAttX/{devX[0]}")
VPgeo_YFrame = pd.read_csv(f"data/DevAttY/{devY[0]}")
column_dict = {j:f"Feature_{i+1}" for i,j in enumerate(VPgeo_XFrame)}
print(column_dict)
Y_map = {j:f"Label" for i,j in enumerate(VPgeo_YFrame)}
VPgeo_YFrame.rename(columns=Y_map, inplace=True)
VPgeo_XFrame.rename(columns=column_dict, inplace=True)
print(VPgeo_XFrame.columns)
print(VPgeo_YFrame.columns)
VPgeo_XFrame['Label'] = VPgeo_YFrame['Label']
VPgeo_XYFrame = VPgeo_XFrame
print(len(VPgeo_XYFrame.columns))
