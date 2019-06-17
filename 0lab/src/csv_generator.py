from os import listdir

curDir = "C:\\Users\\pkmixer\\Downloads\\AI\\Stocks\\"

outputFile = "stocks.csv"

with open(outputFile, "w") as out:
    out.write("Name,Country,Date,Open,High,Low,Close,Volume,OpenInt\n")

    for fileName in listdir(curDir):
        name, country, format = fileName.split(".")
        with open(curDir + fileName, "r") as stock:
            stock.readline()
            for line in stock.readlines():
                out.write(name + "," + country + "," + line)
