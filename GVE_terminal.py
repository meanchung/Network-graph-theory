#!usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

'''从文本文件读取图G的数据'''
G = np.loadtxt('GVE.txt', comments='#', delimiter=' ',
               dtype='int', encoding='utf8')
Data = sorted(list(zip(G[0, :], G[1, :], G[2, :])))
E, D1, D2 = zip(*Data)
m = np.max(G[1::, :])
n = len(E)
# k = int(G[3,0])
p = m-1


def getB(k=-1):
    '''
    由图G的数据建立关联矩阵，逐行扫描
    '''
    B = np.zeros(m*n, dtype='int').reshape(m, n)
    i = 0  # 行 代表竖向的V
    while i < m:  # 这里如果用的是p=m-1的话，求出来默认是Bm矩阵
        j = 0  # 列
        while j < n:
            if D1[j] == i+1:  # 加1是为了转化为实际的表达
                B[i][j] = 1
            elif D2[j] == i+1:
                B[i][j] = -1
            else:
                B[i][j] = 0
            j += 1
        i += 1
    if k == -1:
        return B
    else:
        return np.delete(B, k-1, axis=0)  # 传入参数即可求取Bk


def getA():
    """ 从关联矩阵求邻接矩阵A ，按列扫描，注意观察i，j的先后顺序"""
    B = getB()
    A = np.zeros(shape=(m, m), dtype='int')
    for j in range(n):
        va = -1
        vz = -1
        for i in range(m):
            if B[i][j] == 1:
                va = i
            elif B[i][j] == -1:
                vz = i
            else:
                pass
        A[va][vz] = 1
    return A


def getA2():
    """
    教材上的方法
    采用三次遍历的方法求取邻接矩阵，需要计算的次数为m*n*n，当图较大的时候，相当的麻烦
    """
    A2 = np.zeros((m, m), dtype='int')
    i = 0
    while i < m:
        j = 0
        while j < m:
            K = 0
            while K < n:
                if D1[K] == i+1:
                    if D2[K] == j+1:
                        A2[i][j] = 1
                    else:
                        pass
                else:
                    pass
                K += 1
            j += 1
        i += 1
    return A2


def getA3():
    """
    直接读取D1，D2作为二维数组的下标，给对应的位置赋值为1
    """
    A3 = np.zeros((m, m), dtype='int')
    for i in range(n):
        A3[D1[i]-1][D2[i]-1] = 1  # 减1是为了转化为电脑的存储方式
    return A3


def getD():
    '''依据邻接矩阵求道路矩阵'''
    A = getA()
    D = A  # 这里用掉了一次m-1
    for _ in range(m-2):
        A = np.dot(D, A)
        D = A + D
    return D


def getP():
    '''将道路矩阵转化为可达矩阵'''
    P = getD()
    for i in range(m):
        for j in range(m):
            if P[i][j] != 0:
                P[i][j] = 1
            else:
                P[i][j] = 0
    return P


if __name__ == '__main__':
    while True:
        command = input('使用之前请将图的数据存放在GVE.txt的文件中，请输入想要求解的矩阵（Q退出）：\n').upper()
        if command != 'Q':
            if command == 'A':
                print('此图的邻接矩阵A1为：\n', getA(), '\n')
            elif command == 'A2':
                print('此图的邻接矩阵A2为：\n', getA2(), '\n')
            elif command == 'A3':
                print('此图的邻接矩阵A3为：\n', getA3(), '\n')
            elif command == 'D':
                print('此图的道路矩阵为：\n', getD(), '\n')
            elif command == 'P':
                print('此图的可达矩阵为：\n', getP(), '\n')
            elif command == 'B':
                k = input('请输入Bk的下标求取基本关联矩阵，继续输入B求取关联矩阵：\n').upper()
                if k == 'B':
                    print('此图的关联矩阵为：\n', getB(), '\n')
                else:
                    k = int(k)
                    print('此图的基本关联矩阵B%d为：\n' % k, getB(k), '\n')
        else:
            break
