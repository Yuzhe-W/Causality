from pandas import Series
import numpy as np
import pandas as pd
import xarray as xr
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
    ds = xr.open_dataset('E:\causality\Z500_ens1.nc')
    print(ds)

    lon1 = 12
    lat1 = 185
    dsloc1 = ds.sel(lon=lon1, lat=lat1, method='nearest')
    df1 = ds.to_dataframe()
    # print(df1.dtypes)
    dsloc1['z500t42'].plot()
    print(dsloc1['z500t42'])

    ds.plot
    loni = 25
    lati = 102
    dsloc2 = ds.sel(lon=loni, lat=lati, method='nearest')
    dsloc2['z500t42'].plot()
    print("------concat xarray-------")
    x = xr.concat([dsloc1['z500t42'], dsloc2['z500t42']], dim='y')
    print(x)
    """
    print("------convert the concat to dataframe-------")
    print(x.to_dataframe())
    print(x[0])
    """

    print("------get time series in form of pd Series-----")
    x0_series = Series(x[0])
    x1_series = Series(x[1])
    print(x0_series)
    print("-----PUT TWO SERIES INTO A DATAFRAME-----")
    df = pd.concat([x0_series, x1_series], axis=1)
    print(df)
    # Check stationary, Transform data
    stationary = -1
    while stationary == -1:
        result = adfuller(x[0])
        print("p-value is: ", result[1])
        if result[1] > 0.05:
            print("Time series A is non-stationary")
            x[0] = x[0] - x[0].shift(1)
            x = x.dropna()
        else:
            stationary = 1
            print("Time Series A is stationary")

    stationary = -1
    while stationary == -1:
        result = adfuller(x[1])
        print("p-value is: ", result[1])
        if result[1] > 0.05:
            print("Time series B is non-stationary")
            x[1] = x[1] - x[1].shift(1)
            x = x.dropna()
        else:
            stationary = 1
            print("Time Series B is stationary")

    # Choose lag value:
    print("Choose lag value")
    model = VAR(df[[0,1]])
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

    # Granger check
    grangercausalitytests(df[[0, 1]], maxlag=[lag])
