import os
curDir = "C:\\Users\\pkmixer\\Downloads\\AI\\Stocks\\"

for fileName in os.listdir(curDir):
    if os.stat(curDir + fileName).st_size is 0:
        print(fileName)
        os.remove(curDir + fileName)

