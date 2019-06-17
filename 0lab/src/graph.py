import matplotlib.pyplot as plt
import pandas as pd
import os

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


curDir = "C:\\Users\\pkmixer\\Downloads\\AI\\Stocks\\"
skipColumns = ["Volume", "OpenInt"]

for fileName in os.listdir(curDir):
    indice = fileName.split(".")[0]
    if not names.get(indice):
        names[indice] = indice
    data = pd.read_csv(curDir + fileName, usecols=lambda x: x not in skipColumns)

    data.plot("Date")

    plt.ylabel("Price")
    plt.title(names[indice])
    plt.tight_layout()
    plt.xticks(rotation=30)

    plt.savefig(curDir + "pics1\\" + fileName + ".png")
    plt.close()
