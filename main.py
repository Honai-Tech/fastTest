import numpy as np
import pandas as pd

url = 'https://raw.githubusercontent.com/Honai-Tech/fastTest/main/adverse-effects.csv'
df = pd.read_csv(url)

df.head()
# print(df.head())