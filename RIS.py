# -*- coding: utf-8 -*-
import numpy as np
import time
import argparse
import random

def get_RRS(G):
    
    source = random.choice(np.unique([G[i][0] for i in range(len(G))]))

    g = np.delete(G, [random.random() > G[i][2] for i in range(len(G))], axis = 0)

    new_nodes, RRS0 = [source], [source]
    RRS = []
    
    while new_nodes:

        temp = np.delete(g, [g[i][1] not in new_nodes for i in range(len(g))], axis = 0)
        
        temp = [temp[i][0] for i in range(len(temp))]

        RRS = list(set(temp + RRS0))

        new_nodes = list(set(RRS) - set(RRS0))

        RRS0 = RRS[:]

    return (RRS)


def RIS(G, k, time_limit):
    start = time.perf_counter()

    R = []

    end = time.perf_counter()
    while end - start < time_limit/2:
        R.append(get_RRS(G))
        end = time.perf_counter()

    for _ in range(k):
        flat_map = [int(item) for subset in R for item in subset]

        counts = np.bincount(flat_map)
        #返回众数
        seed = np.argmax(counts)
        # seed = Counter(flat_map).most_common()[0][0]
        # print(Counter(flat_map).most_common()[0])
        print(seed)

        R = [rrs for rrs in R if seed not in rrs]


if __name__ == '__main__':
    '''
    从命令行读参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-k', '--seedCount', type=int, default='1')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seedCount = args.seedCount
    model = args.model
    time_limit = args.time_limit

    G = np.loadtxt(file_name, skiprows=1)

    RIS(G, seedCount, time_limit)
