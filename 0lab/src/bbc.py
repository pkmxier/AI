import matplotlib.pyplot as plt
from os import listdir
import numpy as np

curDir = "C:\\Users\\pkmixer\\Downloads\\AI\\bbc\\"

for directory in listdir(curDir):
    dictionary = {}
    frequency = []
    for file in listdir(curDir + directory):
        with open(curDir + directory + "\\" + file, "r") as f:
            for line in f.readlines():
                line = line.lower()

                chars = [".", ",", "\'"]
                for char in chars:
                    line = line.replace(char, "")

                line = line.split()

                for word in line:
                    if word not in dictionary:
                        dictionary[word] = 0
                    else:
                        dictionary[word] += 1

    for word in dictionary:
        frequency.append((word, dictionary[word]))

    frequency.sort(key=lambda x: x[1])

    X = []
    Y = []
    for word in frequency[:-10:-1]:
        X.append(word[0])
        Y.append(word[1])

    y_pos = np.arange(len(X))
    plt.bar(y_pos, Y)
    plt.xticks(y_pos, X)
    plt.ylabel("Frequency")
    plt.title("Words frequency in {} news".format(directory))
    plt.savefig(curDir + "..\\bbc_pics\\" + directory + ".png")
    plt.close()
