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
# VPgeo Data Frame 1
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
VPgeo_XYFrame.to_csv("data/DevAttXY/VPgeo.csv")

# VPaan Data Frame 2
VPaan_XFrame = pd.read_csv(f"data/DevAttX/{devX[1]}")
VPaan_YFrame = pd.read_csv(f"data/DevAttY/{devY[1]}")
column_dict = {j:f"Feature_{i+1}" for i,j in enumerate(VPaan_XFrame)}
print(column_dict)
Y_map = {j:f"Label" for i,j in enumerate(VPaan_YFrame)}
VPaan_YFrame.rename(columns=Y_map, inplace=True)
VPaan_XFrame.rename(columns=column_dict, inplace=True)
print(VPaan_XFrame.columns)
print(VPaan_YFrame.columns)
VPaan_XFrame['Label'] = VPaan_YFrame['Label']
VPaan_XYFrame = VPaan_XFrame
print(len(VPaan_XYFrame.columns))
VPaan_XYFrame.to_csv("data/DevAttXY/VPaan.csv")

print(devX)
# VPaak Data Frame 3
VPaak_XFrame = pd.read_csv(f"data/DevAttX/{devX[2]}")
VPaak_YFrame = pd.read_csv(f"data/DevAttY/{devY[2]}")
column_dict = {j:f"Feature_{i+1}" for i,j in enumerate(VPaak_XFrame)}
print(column_dict)
Y_map = {j:f"Label" for i,j in enumerate(VPaak_YFrame)}
VPaak_YFrame.rename(columns=Y_map, inplace=True)
VPaak_XFrame.rename(columns=column_dict, inplace=True)
print(VPaak_XFrame.columns)
print(VPaak_YFrame.columns)
VPaak_XFrame['Label'] = VPaak_YFrame['Label']
VPaak_XYFrame = VPaak_XFrame
print(len(VPaak_XYFrame.columns))
VPaak_XYFrame.to_csv("data/DevAttXY/VPaak.csv")

print(devX)
# VPaap Data Frame 4
VPaap_XFrame = pd.read_csv(f"data/DevAttX/{devX[3]}")
VPaap_YFrame = pd.read_csv(f"data/DevAttY/{devY[3]}")
column_dict = {j:f"Feature_{i+1}" for i,j in enumerate(VPaap_XFrame)}
print(column_dict)
Y_map = {j:f"Label" for i,j in enumerate(VPaap_YFrame)}
VPaap_YFrame.rename(columns=Y_map, inplace=True)
VPaap_XFrame.rename(columns=column_dict, inplace=True)
print(VPaap_XFrame.columns)
print(VPaap_YFrame.columns)
VPaap_XFrame['Label'] = VPaap_YFrame['Label']
VPaap_XYFrame = VPaap_XFrame
print(len(VPaap_XYFrame.columns))
VPaap_XYFrame.to_csv("data/DevAttXY/VPaap.csv")

# VPaak Data Frame 5
VPaak_XFrame = pd.read_csv(f"data/DevAttX/{devX[4]}")
VPaak_YFrame = pd.read_csv(f"data/DevAttY/{devY[4]}")
column_dict = {j:f"Feature_{i+1}" for i,j in enumerate(VPaak_XFrame)}
print(column_dict)
Y_map = {j:f"Label" for i,j in enumerate(VPaak_YFrame)}
VPaak_YFrame.rename(columns=Y_map, inplace=True)
VPaak_XFrame.rename(columns=column_dict, inplace=True)
print(VPaak_XFrame.columns)
print(VPaak_YFrame.columns)
VPaak_XFrame['Label'] = VPaak_YFrame['Label']
VPaak_XYFrame = VPaak_XFrame
print(len(VPaak_XYFrame.columns))
VPaak_XYFrame.to_csv("data/DevAttXY/VPaak.csv")
