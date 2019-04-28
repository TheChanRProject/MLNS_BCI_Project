import pandas as pd
from os import listdir

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

devY = find_csv_filenames('data/DevAttY', suffix=".csv")

df = pd.concat([pd.read_csv(f"data/DevAttY/{i}") for i in devY])

print(len(df))

df.to_csv("data/Merged/merged_DevAttentionY.csv")
