from pandas import Series
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller


class GrangerCausality:

    def grangerCause(self, array1, array2):

        print("------get time series in form of pd Series-----")
        x0_series = Series(array1)
        x1_series = Series(array2)
        print(x0_series)
        print("-----PUT TWO SERIES INTO A DATAFRAME-----")
        df = pd.concat([x0_series, x1_series], axis=1)
        df_cp = df.copy(deep=True)

        print(df)
        # Check stationary, Transform data
        stationary = -1
        while stationary == -1:
            result = adfuller(df[0])
            print("p-value is: ", result[1])
            if result[1] > 0.05:
                print("Time series A is non-stationary")
                df[0] = df[0] - df[0].shift(1)
                df = df.dropna()
            else:
                stationary = 1
                print("Time Series A is stationary")

        stationary = -1
        while stationary == -1:
            result = adfuller(df[1])
            print("p-value is: ", result[1])
            if result[1] > 0.05:
                print("Time series B is non-stationary")
                df[1] = df[1] - df[1].shift(1)
                df = df.dropna()
            else:
                stationary = 1
                print("Time Series B is stationary")

        # Choose lag value:
        print("Choose lag value")
        model = VAR(df_cp[[0, 1]])
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

        print("check dataframe: ")
        print(df)

        # Granger check
        grangercausalitytests(df[[0, 1]], maxlag=[lag])

if __name__ == "__main__":
    file = 'E:\causality\sst_mon.nc'
    ds = xr.open_dataset(file)
    print(ds)

    lon_1 = 12.5
    lat_1 = 72.5
    lon_2 = 11.5
    lat_2 = 71.5

    lon1 = lon_1
    lat1 = lat_1
    dsloc1 = ds.sel(lon=lon1, lat=lat1, method='nearest')
    df1 = ds.to_dataframe()
    # print(df1.dtypes)
    dsloc1['sst'].plot()
    plt.show()
    print(dsloc1['sst'])

    #ds.plot
    lon2 = lon_2
    lat2 = lat_2
    dsloc2 = ds.sel(lon=lon2, lat=lat2, method='nearest')
    dsloc2['sst'].plot()
    plt.show()
    print("------concat xarray-------")
    x = xr.concat([dsloc1['sst'], dsloc2['sst']], dim='y')
    print(x)

    gc = GrangerCausality()
    gc.grangerCause(x[0],x[1])
