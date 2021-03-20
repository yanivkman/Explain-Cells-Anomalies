import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from constants import INDECIES_ARRAY, S_STD, S_MinMax


def load_csvs(path_of_csvs):
    """
    This function receives list of paths to plates' csv files and returns an indexed pandas dataframe of
    all of them. Also this function sets the indecies to the data.
    """
    df_all: pd.DataFrame = None
    for path in path_of_csvs:
        df_temp: pd.DataFrame = pd.read_csv(path) # The colon is there to indicate the type.
        df_all = pd.concat([df_all, df_temp], ignore_index=True)
        df_all = df_all.set_index(INDECIES_ARRAY)   # Create all the indecies.
                                                    # And make them appear at the begining of the row.

    return df_all


def prepare_for_autoencoder(df):
    """
    This function prepares the data for the autoencoder. Setting the indecies columns, removing lines
    containing nans and changing the type of the data to fit the type the autoencoder is working with.
    df: dataFrame to prepare.
    returns: the dataframe ready to be used in the autoencoder.
    """
    df = df.dropna() # remove lines with nan values (there are other ways to complete data).
    df = np.asarray(df).astype('float32') # TODO: try removing line and document later the affect
    return df


def fit_scaler(df, scale_method):
    """
    This function is fitting a scaler using one of two methods: STD and MinMax
    df: dataFrame to fit on
    scale_method: 'Std' or 'MinMax'
    returns: the scaler acording to the given dataFrame and scale mathod
    """

    if scale_method == S_STD:
        scaler = StandardScaler()
    elif scale_method == S_MinMax:
        scaler = MinMaxScaler()
    else:
        scaler = None

    return scaler


def scale_data(df, scaler):
    """
    This function is scaling a dataFrame acording to the fited scaler.
    df: dataFrame to scale
    scaler: the fited scaler
    returns: the scaled dataFrame
    """
    scaled_df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
    # scaled_df.fillna(0, inplace=True) # TODO: remove if function works
    return scaled_df
