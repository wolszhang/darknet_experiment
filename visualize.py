import sys
import matplotlib.pyplot as plt

nodes = 0
title = ''
data = []
with open(sys.argv[1], 'r') as inp:
    for line in inp:
        if title == '':
            title = line
        elif nodes == 0:
            nodes = int(line)
        else:
            data.append((int(line.split()[0]), int(line.split()[1])))
layer, depth = {}, {}
layer[0] = 1
depth[1] = [0]
while len(layer) < nodes:
    for (a,b) in data:
        if a in layer and b not in layer:
            layer[b] = layer[a]+1
            if layer[a] +1 in depth:
                depth[layer[a] +1].append(b)
            else:
                depth[layer[a] +1] = [b]
        if b in layer and a not in layer:
            layer[a] = layer[b]+1
            if layer[b] +1 in depth:
                depth[layer[b] +1].append(a)
            else:
                depth[layer[b] +1] = [a]

leng = 10
loc = {}
for key in depth:
    for i, item in enumerate(depth[key]):
        loc[item] = [layer[item], (i+1)*leng/(1+len(depth[key]))]
        plt.plot(layer[item], (i+1)*leng/(1+len(depth[key])), 'o')
for (a,b) in data:
    plt.arrow(loc[a][0], loc[a][1], loc[b][0] - loc[a][0], loc[b][1]-loc[a][1], width=0.02,head_length=0.0,head_width=0.0)
plt.axis('off')
plt.title(title)
plt.show()

