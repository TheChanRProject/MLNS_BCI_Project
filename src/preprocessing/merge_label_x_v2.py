import pandas as pd
from os import listdir

a = pd.read_csv("train/VPaat_DevAttentionX.csv")
b = pd.read_csv("train/VPjat_DevAttentionX.csv")
c = pd.read_csv("train/VPaaq_DevAttentionX.csv")
d = pd.read_csv("train/VPgeo_DevAttentionX.csv")
e = pd.read_csv("train/VPaak_DevAttentionX.csv")
f = pd.read_csv("train/VPjaq_DevAttentionX.csv")
g = pd.read_csv("train/VPaar_DevAttentionX.csv")

df = pd.concat([a,b,c,d,e,f,g])

print(len(df))

print(df.shape)
column_dict = {j:f"Feature_{i+1}" for i,j in enumerate(df)}
print(column_dict)
df.rename(columns=column_dict, inplace=True)

df.to_csv("train/merged_labeled_DevAttentionX_v2.csv")
