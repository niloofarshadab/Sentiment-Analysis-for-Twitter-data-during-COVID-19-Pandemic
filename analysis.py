import numpy as np
import time
from os import listdir
from os.path import isfile, join

dir_path = "./data/month/"
file_paths = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f[:4] == 'sent']

geos = {}
positive = {}
negative = {}

def read_file(file_path):
    with open(file_path) as f:
        line = f.readline().strip()

        k = 0
        while(line != ''):

            if(line[0] != '1'):
                #print(line)
                line = f.readline().strip()
                continue
            splt = line.split('\t')

            #id, date, time, language, geo, sentiment, text
            #print(splt)
            date = splt[1][0:7]
            if(splt[5] == "0"):
                positive[date] = positive.get(date, 0) + 1
            else:
                negative[date] = negative.get(date, 0) + 1
            geos[splt[4]] = geos.get(splt[4], 0) + 1
            line = f.readline().strip()
        print(k)

#print('\n'.join(file_paths))
i = 0
for fp in file_paths:
    i += 1
    #print(f"\r{i}/{len(file_paths)}", end="")
    read_file(join(dir_path, fp))
print()
print(geos)

import matplotlib.pyplot as plt
x = []
y1 = []
y2 = []
y3 = []
for k in set(positive.keys()).union(negative.keys()):
    x.append(k)
    y1.append(positive.get(k, 0))
    y2.append(negative.get(k, 0))
    y3.append(positive.get(k, 0) / (positive.get(k, 0) + negative.get(k, 0)))
plt.plot_date(x, y3)
#plt.plot_date(x, y2)
plt.show()
