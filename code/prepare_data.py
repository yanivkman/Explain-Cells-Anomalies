import numpy as np
import pandas as pd

from constants import *


'''
This function receives list of paths to plates' csv files and returns an indexed pandas dataframe of
all of them.
'''
def load_csvs(path_of_csvs):
    df_all: pd.DataFrame
    for path in path_of_csvs:
        df_temp: pd.DataFrame = pd.read_csv(path) # The colon is there to indicate the type
        df_all = pd.concat([df_all, df_temp], ignore_index=True)

    return df_all



