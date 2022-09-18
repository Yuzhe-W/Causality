import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

class GrangerCausality:

    def plot(self, df, x, y, name):
        #print(df.dtypes)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.title(name)
        plt.show()
    """
        def make_it_stationary(self, df, name):
        stationary = -1
        while (stationary == -1):
            result = adfuller(df[name])
            print("p-value is: ", result[1])
            if (result[1] > 0.05):
                print("Time series is non-stationary")
                df[name] = df[name] - df[name].shift(1)
                df = df.dropna()
            else:
                stationary = 1
                print("Time Series is stationary")
    """


if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\yzwan\Downloads\ONEPIECE.csv")
    gc = GrangerCausality()
    print(df.dtypes)

    #df["rank"] = pd.to_numeric(df['rank'], errors='coerce')
    gc.plot(df, x=df.episode, y=df.start, name="release year")
    gc.plot(df, x=df.episode, y=df.average_rating, name="average_rating")

    #Make a copy of df
    df_cp = df.copy(deep=True)

    #Check stationary, Transform data
    stationary = -1
    while(stationary == -1):
        result = adfuller(df['start'])
        print("p-value is: ", result[1])
        if(result[1] > 0.05):
            print("Time series is non-stationary")
            df['start'] = df['start'] - df['start'].shift(1)
            df = df.dropna()
        else:
            stationary = 1
            print("Time Series is stationary")

    stationary = -1
    while (stationary == -1):
        result = adfuller(df['average_rating'])
        print("p-value is: ", result[1])
        if (result[1] > 0.05):
            print("Time series is non-stationary")
            df['average_rating'] = df['average_rating'] - df['average_rating'].shift(1)
            df = df.dropna()
        else:
            stationary = 1
            print("Time Series is stationary")

    #gc.make_it_stationary(df, name="High")
    gc.plot(df, x=df.episode, y=df.start, name="release year")
    gc.plot(df, x=df.episode, y=df.average_rating, name="average_rating")

    # Choose lag value:
    df_vars = df_cp[['start', 'average_rating']]
    model = VAR(df_vars)
    hashmap = dict()
    array_aics = np.array([])
    for i in range(1, 13):
        result = model.fit(i)
        print("lag is: ", i)
        print("aic-value is: ", result.aic)
        array_aics=np.append(array_aics, result.aic)
        hashmap[result.aic] = i

    lag = hashmap[array_aics.min()]
    print("appropriate lag is: ", lag)
    grangercausalitytests(df[['start', 'average_rating']], maxlag=[lag])

