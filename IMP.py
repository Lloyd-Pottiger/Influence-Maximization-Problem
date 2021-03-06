# -*- coding: utf-8 -*-
import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
from numpy import random
import math

random.seed(int(time.time()))
Comb = lambda x, y: math.factorial(x) // (math.factorial(y) * math.factorial(x - y))
core = 8

class Graph(object):
    def __init__(self, m, n):
        self.vertex = m
        self.edge = n
        self.p = []
        self.post_edge = [list() for i in range(m + 1)]

    def insert(self, a, b, w):
        self.post_edge[b].append((a,w))

    def post_to(self, b):
        return self.post_edge[b]


def read_graph(file):
    with open(file) as f:
        m, n = map(int, f.readline().split())
        g = Graph(m, n)
        for i in range(n):
            a, b, w = map(float, f.readline().split())
            g.insert(int(a), int(b), w)
    return g

def IMM(g, k, e, l):
    n = g.vertex
    l = l * (1 + math.log(2) / math.log(n))
    R = Sampling(g, k, e, l)
    S_k_star = NodeSelection(R, k)[0]
    return S_k_star

def Sampling(G: Graph, k, e, l):
    R = list()
    LB = 1
    e_ = math.sqrt(2) * e
    n = G.vertex
    alpha = math.sqrt(l * math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (math.log(Comb(n, k)) + l * math.log(n) + math.log(2)))
    lambda_ = (2 + 2 * e_ / 3) * (math.log(Comb(n, k)) + l * math.log(n) + math.log(math.log2(n))) * n / (pow(e_, 2))
    for i in range(1, int(math.log2(n))):
        x = n / pow(2, i)
        theta = lambda_ / x
        cnt = (theta - len(R)) // core
        R = creat_mp(R, cnt)
        F_R = NodeSelection(R, k)[1]
        if n * F_R >= (1 + e_) * x:
            LB = n * F_R / (1 + e_)
            break
    lambda_star = 2 * n * pow((1 - 1 / math.e) * alpha + beta, 2) * pow(e, -2)
    theta = lambda_star / LB
    cnt = theta - len(R)
    if cnt > 0:
        R = creat_mp(R, cnt)
    return R


def creat_mp(R, cnt):
    pool = mp.Pool(core)
    result = []
    for i in range(core):
        result.append(pool.apply_async(get_RR, args=(G, cnt)))
    pool.close()
    pool.join()
    for res in result:
        R.extend(res.get())
    return R


def NodeSelection(R, k):
    S = set()
    rr_dict = {}
    R_S_k = set()
    cnt = [0 for i in range(G.vertex + 1)]
    for i in range (0, len(R)):
        rr = R[i]
        for u in rr:
            if u not in rr_dict:
                rr_dict[u] = set()
            rr_dict[u].add(i)
            cnt[u] += 1
    for i in range(k):
        v = cnt.index(max(cnt))
        S.add(v)
        R_S_k = R_S_k.union(rr_dict[v])
        cur_dict = rr_dict[v].copy()
        for d in cur_dict:
            for n in R[d]:
                cnt[n] -= 1
    return S, len(R_S_k)/len(R)


def get_RR(G, cnt):
    RR = []
    while cnt > 0:
        n = G.vertex
        v = random.randint(1, n)
        all_activity_set = [v]
        activity_set = [v]
        while activity_set:
            new_activity_set = []
            for u in activity_set:
                for (v, w) in G.post_to(u):
                    if v not in all_activity_set:
                        if random.random() <= w:
                            new_activity_set.append(v)
                            all_activity_set.append(v)
            activity_set = new_activity_set
        RR.append(all_activity_set)
        cnt -= 1
    return RR


if __name__ == '__main__':
    '''
    从命令行读参数
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='Test/network.txt')
    parser.add_argument('-k', '--seedCount', type=int, default=5)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    k = args.seedCount
    model = args.model
    time_limit = args.time_limit

    G = read_graph(file_name)

    l = 1
    e = math.sqrt((G.vertex + G.edge) * (k + l) * math.log(G.vertex)/(5e8 * time_limit))
    if G.vertex < 500 and k < 10:
        if e < 0.01:
            e = 0.01
    else:
        if e < 0.08:
            e = 0.08
        elif e < 0.1:
            e = 0.1
    S_k_star = IMM(G, k, e, l)
    for seed in S_k_star:
        print(seed)
    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()
