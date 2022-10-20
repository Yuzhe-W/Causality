from pandas import Series
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import ccf


class GrangerCausality:

    def grangerCause(self, array1, array2):

        print("------get time series in form of pd Series-----")
        x0_series = Series(array1)
        x1_series = Series(array2)
        print(x1_series)

        print("------Plot-------")
        x0_series.plot()
        plt.show()
        x1_series.plot()
        plt.show()

        print("-----PUT TWO SERIES INTO A DATAFRAME-----")
        df = pd.concat([x0_series, x1_series], axis=1)
        df = df.dropna()
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

    def nino_index(self, ds):
        # 1580 is the size of time
        area_store = [[] for i in range(1580)]
        for lat_i in range(-5, 6):
            for lon_j in range(170, 191):
                lat = lat_i
                lon = lon_j
                dsloc = ds.sel(lon=lon, lat=lat, method='nearest')
                array = dsloc['sst'].to_numpy()
                # print(array)
                for k in range(len(array)):
                    area_store[k].append(array[k])

        area_mean = []
        for time_idx in range(len(area_store)):
            cur_array = area_store[time_idx]
            area_mean.append(np.mean(cur_array))

        mean_5mon = []
        for idx in range(0, len(area_mean), 5):
            # print(idx)
            if (idx == len(area_mean) - 1):
                break
            five_mon_mean = (area_mean[idx] + area_mean[idx + 1] + area_mean[idx + 2] +
                             area_mean[idx + 3] + area_mean[idx + 4]) / 5
            mean_5mon.append(five_mon_mean)

        # print(mean_5mon)
        return mean_5mon

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\yzwan\Downloads\nino.csv")
    fig, ax = plt.subplots()
    x = df.Year_Mon
    y = df.SST
    ax.plot(x, y)
    ax.set_xlabel("date")
    plt.show()

    file = 'E:\causality\sst_mon.nc'
    ds = xr.open_dataset(file)
    print(ds)

    gc = GrangerCausality()

    lon_1 = 170
    lat_1 = 5
    lon_2 = 190
    lat_2 = -5

    lon1 = lon_1
    lat1 = lat_1
    dsloc1 = ds.sel(lon=lon1, lat=lat1, method='nearest')
    dsloc = dsloc1.sel(time=slice('19500101','20200801'))
    dsloc['sst'].plot()
    plt.show()

    # Calculate nino 3.4 index -faster way
    area_store1 = ds.sel(lon=slice(170, 190), lat=slice(5, -5))
    area_store = area_store1.sel(time=slice('19500101','20200801'))
    area_mean = area_store.mean(dim=('lon', 'lat'))
    mean_5mon = area_mean.rolling(time=5).mean()
    mean_5mon['sst'].plot()
    print(mean_5mon)
    plt.show()

    # correlation
    A = (Series(df['SST']))
    B = (Series(mean_5mon['sst']))
    df1 = pd.concat([A, B], axis=1)
    df1 = df1.dropna()
    print(df1)
    r = np.corrcoef(np.array(df1['SST']), np.array(df1[0]))
    print(r)

    x = xr.concat([dsloc1['sst'], mean_5mon['sst']], dim='y')
    gc.grangerCause(x[0],x[1])
