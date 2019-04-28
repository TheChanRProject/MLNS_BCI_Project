import pandas as pd
from os import listdir

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

devX = find_csv_filenames('data/DevAttX', suffix=".csv")

df = pd.concat([pd.read_csv(f"data/DevAttX/{i}") for i in devX])

print(len(df))

print(df.shape)
column_dict = {j:f"Feature_{i+1}" for i,j in enumerate(df)}
print(column_dict)
df.rename(columns=column_dict, inplace=True)

df.to_csv("data/Merged/merged_labeled_DevAttentionX.csv")
