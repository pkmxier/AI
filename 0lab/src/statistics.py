import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import os


curDir = "C:\\Users\\pkmixer\\Downloads\\AI\\Stocks\\"

names = {}
indicesDir = "C:\\Users\\pkmixer\\Downloads\\indices\\"

for fileName in os.listdir(indicesDir):
    with open(indicesDir + fileName, "r") as indiceName:
        indiceName.readline()
        for line in indiceName:
            pair = line.strip().split("\t")
            if len(pair) == 2:
                ticker, name = pair
            names[ticker.lower()] = name

skipColumns = ["Volume", "OpenInt"]

for fileName in os.listdir(curDir):
    data = pd.read_csv(curDir + fileName, usecols=lambda x: x not in skipColumns)

    index = fileName.split(".")[0]
    if not names.get(index):
        names[index] = index

    scatter_matrix(data, alpha=0.2)
    plt.suptitle(names[index])
    plt.savefig(curDir + "pics\\" + fileName + ".png")
    plt.close()
