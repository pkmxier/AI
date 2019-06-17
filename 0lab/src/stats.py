import matplotlib.pyplot as plt
import pandas as pd
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

skipColumns = ["Date", "OpenInt", "Volume"]

for fileName in os.listdir(curDir):
    data = pd.read_csv(curDir + fileName, usecols=lambda x: x not in skipColumns)

    index = fileName.split(".")[0]
    if not names.get(index):
        names[index] = index

    with open(curDir + "..\\" + fileName + ".stat", "w") as file:
        file.write(data.describe().to_csv())
