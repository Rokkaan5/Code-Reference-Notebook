import pandas as pd

# %%
df = pd.read_fwf("HW4_data.txt", sep=" ",header=None,)
df.head()
