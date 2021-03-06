# -*- coding: utf-8 -*-
import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
from numpy import random


random.seed(int(time.time()))
core = 4


class Graph(object):
    def __init__(self, m):
        self.length = m
        self.node = [False for i in range(0, m + 1)]
        self.p = [random.random() for i in range(0, m + 1)]
        self.edge = {}
        self.post_edge = {}
        for i in range(1, m + 1):
            self.edge[i] = list()
            self.post_edge[i] = list()

    def insert(self, a, b, w):
        self.edge[a].append((b, w))
        self.post_edge[b].append((a,w))

    def next_to(self, a):
        return self.edge[a]

    def post_to(self, b):
        return self.post_edge[b]

    def is_active(self, a):
        return self.node[a]

    def be_active(self, a):
        self.node[a] = True

    def reset(self,acti_set):
        self.node = [False for i in range(self.length + 1)]
        for i in acti_set:
            self.node[i] = True
        self.p = [random.random() for i in range(0, self.length + 1)]

    def add_active(self):
        list = []
        for i in range(1, len(self.p)):
            if self.p[i] == 0:
                self.node[i] = True
                list.append(i)
        return list


def read_graph(file):
    f = open(file)
    m, n = map(int, f.readline().split())
    g = Graph(m)
    for i in range(n):
        a, b, w = map(float, f.readline().split())
        g.insert(int(a), int(b), w)
    f.close()
    return g

def read_seed(seed):
    acti_set = []
    f = open(seed)
    line = f.readline().strip('\n')
    while line:
        acti_set.append(int(line))
        line = f.readline().strip('\n')
    f.close()
    return acti_set

def try_active(w):
    p = np.array([w, 1 - w])
    return random.choice([True, False], p=p.ravel())

def OneICSample(g, acti_set):
    count = len(acti_set)
    while len(acti_set) != 0:
        new_acti_set = []
        for i in acti_set:
            for (j, w) in g.next_to(i):
                if not g.is_active(j):
                    if try_active(w):
                        g.be_active(j)
                        new_acti_set.append(j)
        count += len(new_acti_set)
        acti_set = new_acti_set
    return count

def OneLTSample(g, acti_set):
    acti_set.extend(g.add_active())
    count = len(acti_set)
    while len(acti_set) != 0:
        new_acti_set = []
        for i in acti_set:
            for (j, w) in g.next_to(i):
                if not g.is_active(j):
                    total = 0
                    for (k, k_w) in g.post_to(j):
                        if g.is_active(k):
                            total += k_w
                    if total >= g.p[j]:
                        g.be_active(j)
                        new_acti_set.append(j)
        count += len(new_acti_set)
        acti_set = new_acti_set
    return count

def run(file_name, seed, model, time_limit):
    start = time.perf_counter()
    sum = 0
    N = 0
    g = read_graph(file_name)
    acti_set = read_seed(seed)
    end = time.perf_counter()

    while end - start < time_limit/2:
        g.reset(acti_set)
        one = OneLTSample(g, acti_set) if model == 'LT' else OneICSample(g, acti_set)
        sum += one
        N += 1
        end = time.perf_counter()
    return sum/N

if __name__ == '__main__':
    '''
    从命令行读参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='network.txt')
    parser.add_argument('-s', '--seed', type=str, default='seeds.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    time_limit = args.time_limit

    pool = mp.Pool(core)
    result = []

    for i in range(core):
        result.append(pool.apply_async(run, args=(file_name, seed, model, time_limit)))
    pool.close()
    pool.join()
    print(np.mean([res.get() for res in result]))
    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()