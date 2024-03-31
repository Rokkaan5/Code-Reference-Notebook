import pandas as pd

# %%
data = pd.read_fwf("HW5_Data.txt", sep=" ",header=None,)
data.head()

# %%
signal = pd.read_fwf("HW5_Signal.txt",sep=" ",header=None)
signal.head()

# %%
BI = pd.read_fwf("HW5_BlurredImage.txt",sep=" ",header=None)
BI.head()

# %%
transfer = pd.read_fwf("HW5_TransferFunc.txt",sep=" ",header=None)
transfer.head()

# %%
