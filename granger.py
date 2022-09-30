from pandas import Series
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller


class GrangerCausality:

    def plot(self, df, x, y, name):
        # print(df.dtypes)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        #ax.set_xlabel("date")
        ax.set_ylabel(name)
        plt.title(name)
        plt.show()


if __name__ == "__main__":
    # ds = xr.open_dataset(r"C:\Users\yzwan\Downloads\sstmonltm.nc")
    # df = ds.to_dataframe()
    df = pd.read_csv(r"C:\Users\yzwan\Downloads\owid-covid-data.csv")
    df = df.dropna()
    gc = GrangerCausality()
    print(df.dtypes)

    gc.plot(df, x=df.date, y=df.new_deaths, name="new_deaths")
    gc.plot(df, x=df.date, y=df.new_cases, name="new_cases")

    # Make a copy of df
    df_cp = df.copy(deep=True)
    print(df['new_deaths'])
    # Check stationary, Transform data
    stationary = -1
    while stationary == -1:
        result = adfuller(df['new_deaths'])
        print("p-value is: ", result[1])
        if result[1] > 0.05:
            print("Time series A is non-stationary")
            df['new_deaths'] = df['new_deaths'] - df['new_deaths'].shift(1)
            df = df.dropna()
        else:
            stationary = 1
            print("Time Series A is stationary")

    stationary = -1
    while stationary == -1:
        result = adfuller(df['new_cases'])
        print("p-value is: ", result[1])
        if result[1] > 0.05:
            print("Time series B is non-stationary")
            df['new_cases'] = df['new_cases'] - df['new_cases'].shift(1)
            df = df.dropna()
        else:
            stationary = 1
            print("Time Series B is stationary")

    # gc.make_it_stationary(df, name="High")
    gc.plot(df, x=df.date, y=df.new_deaths, name="new_deaths")
    gc.plot(df, x=df.date, y=df.new_cases, name="new_cases")

    # Choose lag value:
    df_vars = df_cp[['new_deaths', 'new_cases']]
    model = VAR(df_vars)
    hashmap = dict()
    array_aics = np.array([])
    for i in range(5):
        result = model.fit(i)
        print("lag is: ", i)
        print("aic-value is: ", result.aic)
        array_aics = np.append(array_aics, result.aic)
        hashmap[result.aic] = i

    lag = hashmap[array_aics.min()]
    print("appropriate lag is: ", lag)
    grangercausalitytests(df[['new_deaths', 'new_cases']], maxlag=[lag])
