import pandas as pd
from os import listdir

a = pd.read_csv("train/VPaat_DevAttentionY.csv")
b = pd.read_csv("train/VPjat_DevAttentionY.csv")
c = pd.read_csv("train/VPaaq_DevAttentionY.csv")
d = pd.read_csv("train/VPgeo_DevAttentionY.csv")
e = pd.read_csv("train/VPaak_DevAttentionY.csv")
f = pd.read_csv("train/VPjaq_DevAttentionY.csv")
g = pd.read_csv("train/VPaar_DevAttentionY.csv")

df = pd.concat([a,b,c,d,e,f,g])

print(len(df))

print(df.shape)
column_dict = {j:f"Output_{i+1}" for i,j in enumerate(df)}
print(column_dict)
df.rename(columns=column_dict, inplace=True)

df.to_csv("train/merged_labeled_DevAttentionY_v2.csv")
