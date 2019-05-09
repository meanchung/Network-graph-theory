#!usr/bin/python
# -*- coding: UTF-8 -*-

import tkinter as tk
from functools import wraps
from os import system
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tkinter.messagebox as msg


'''
图论相关知识的基本代码实现
包括基本的邻接矩阵、基本关联矩阵、基本回路矩阵、
独立割集矩阵、生成树数目，道路矩阵以及简单的绘图等
'''
# 从文本文件读取图G的数据
try:
    G = np.loadtxt('GVE_G.txt', comments='#', delimiter=' ',
                   dtype='int', encoding='utf8')
except OSError:
    msg.showerror(
        '数据文件不存在', ('数据文件不存在，点击确认进行设置（文件应包含三行数据，第一行为弧向量， \
            第二行为弧的始节点，第三行为终节点，以空格隔开，注意末尾不能含有空格）'))
    system('notepad GVE_G.txt')
    G = np.loadtxt('GVE_G.txt', comments='#', delimiter=' ',
                   dtype='int', encoding='utf8')
try:
    Data = sorted(list(zip(G[0, :], G[1, :], G[2, :])))
except IndexError:
    msg.showerror(
        '数据文件错误', '数据文件未正确设置，点击确认进行设置（文件应包含三行数据，第一行为弧向量，第二行为弧的始节点，第三行为终节点，以空格隔开，注意末尾不能含有空格）')
    system('notepad GVE_G.txt')
    G = np.loadtxt('GVE_G.txt', comments='#', delimiter=' ',
                   dtype='int', encoding='utf8')
    Data = sorted(list(zip(G[0, :], G[1, :], G[2, :])))
E, D1, D2 = zip(*Data)
try:
    Ts = np.loadtxt('GVE_T.txt', dtype='int', delimiter=' ',
                    comments='#', encoding='utf8')
except OSError:
    msg.showerror('树枝文件不存在', '未找到树枝文件，点击确定输入树枝\n在一行输入，用空格隔开例如：\n1 3 5 7 8')
    system('notepad GVE_T.txt')
    Ts = np.loadtxt('GVE_T.txt', dtype='int', delimiter=' ',
                    comments='#', encoding='utf8')

edge_label = zip(G[1, :], G[2, :], G[0, :])
m = np.max(G[1::, :])
n = len(E)
p = m-1
result = ''

win = tk.Tk()
win.title('{} X {}网络图相关基本矩阵的求解'.format(m, n))
win.geometry('600x480')
# 创建基本窗口并分割布局
win_up = tk.Frame(win)
win_up0 = tk.Frame(win_up)
win_up1 = tk.Frame(win_up)
win_down = tk.Frame(win)
win_down_left = tk.Frame(win_down)
win_down_right = tk.Frame(win_down)
win_down_left2 = tk.Frame(win_down_left)
win_down_left1 = tk.Frame(win_down_left)
text = tk.Text(win_down_right, font='Times 15 normal')
text.pack(padx=5, pady=13, ipady=10)
win_up.pack(side='top')
win_up0.pack(side='left')
win_up1.pack(side='right')
win_down.pack(side='bottom')
win_down_left.pack(side='left')
win_down_right.pack(side='right')
win_down_left1.pack(side=tk.LEFT)
win_down_left2.pack(side=tk.RIGHT)


def decoratorFunc(aFunc):
    """
    函数装饰器，用于输出结果
    """
    wraps(aFunc)

    def disp(*args, **kwargs):
        text.delete(1.0, tk.END)
        result = aFunc(*args, **kwargs)
        orderE, *_ = getOrderB()
        text.insert('insert', '\n')
        text.insert(tk.INSERT, result)
        text.insert('insert', '\n\n当前弧的排列顺序为：\n')
        text.insert('insert', orderE)
        # return aFunc(*args, **kwargs)
    return disp


def drawG():
    """
    将节点以圆弧排列绘制出图，并将树枝标为红色
    """
    # 创建一个图
    G = nx.DiGraph()  # 现在 G 是空的
    # 添加节点
    nodes = ['V'+str(i) for i in range(1, m+1)]
    G.add_nodes_from(nodes)
    # 也能通过传入一个列表来添加一系列的节点
    # 添加边
    edge_color = []
    for edge, n1, n2 in Data:
        if edge not in Ts:
            G.add_edge('V'+str(n1), 'V'+str(n2), e=edge)
            edge_color.append('blue')
            # weight.append(edge)
        else:
            G.add_edge('V'+str(n1), 'V'+str(n2), e=edge)
            edge_color.append('red')
            # print(edge)
            # weight.append(edge)
    print(edge_color)
    fig = plt.figure('图形可视化')
    fig.suptitle(str(m)+' x '+str(n), fontsize='x-large')
    pos = nx.spring_layout(G)
    edge = nx.draw(G, with_labels=True,
                   pos=pos, node_color='red', font_size=10)
    nx.draw_networkx_edge_labels(
        G, pos=pos, rotate=False)
    nx.draw_networkx_edges(G, pos=pos, edgelist=[x for x in G.edges(
        data=True) if x[2]['e'] in Ts], edge_color='red', width=4, alpha=0.5, arrowstyle='->')
    # fig.set_title(str(m)+'X'+str(n))
    plt.show(edge)


def modifyG():
    """
    打开记事本修改图的数据
    """
    system('notepad GVE_G.txt')


@decoratorFunc
def getB(k=-1):
    '''
    由图G的数据建立关联矩阵，逐行扫描
    '''
    text.insert('insert', '基本关联矩阵为:')
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


def getB_(k=-1):
    '''
    getB的副本，作为其他函数调用
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


# @decoratorFunc
def getA():
    """ 从关联矩阵求邻接矩阵A ，按列扫描，注意观察i，j的先后顺序"""
    B = getB_()
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


# @decoratorFunc
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


@decoratorFunc
def getA3():
    """
    直接读取D1，D2作为二维数组的下标，给对应的位置赋值为1
    """
    text.insert('insert', '邻接矩阵为:')
    A3 = np.zeros((m, m), dtype='int')
    for i in range(n):
        A3[D1[i]-1][D2[i]-1] = 1  # 减1是为了转化为电脑的存储方式
    return A3


@decoratorFunc
def getD():
    '''依据邻接矩阵求道路矩阵'''
    A = getA2()
    text.insert('insert', '道路矩阵为:')
    Ai = A  # 这里用掉了一次m-1
    D = A
    for _ in range(m-2):
        Ai = np.dot(Ai, A)
        D = Ai + D
    return D


def getD_():
    '''getB的副本，供求可达矩阵使用'''
    A = getA2()
    Ai = A  # 这里用掉了一次m-1
    D = A
    for _ in range(m-2):
        Ai = np.dot(Ai, A)
        D = Ai + D
    return D


@decoratorFunc
def getP():
    '''将道路矩阵转化为可达矩阵'''
    text.insert('insert', '可达矩阵为:')
    P = getD_()
    for i in range(m):
        for j in range(m):
            if P[i][j] != 0:
                P[i][j] = 1
            else:
                P[i][j] = 0
    return P


@decoratorFunc
def getSt():
    """
    求当前图的生成树数目
    """
    text.insert('insert', '生成树数目为:')
    B1 = getB_(1)
    St = np.linalg.det(np.dot(B1, np.transpose(B1)))
    return int(St)


def setT():
    """
    修改树枝
    """
    system('notepad GVE_T.txt')


def getT():
    """从文件中读取树枝"""
    Ts = np.loadtxt('GVE_T.txt', dtype='int', delimiter=' ',
                    comments='#', encoding='utf8')
    return Ts


def getOrderB():
    """
    将基本关联矩阵按照余树弦在前，树枝在后的顺序排列
    """
    B = getB_(1)
    # T = [1, 3, 4, 6, 8]
    Ts = getT()
    if len(Ts) == 0:
        setT()
        Ts = getT()
    B11 = np.zeros((m-1, n-m+1), dtype='int')
    B12 = np.zeros((m-1, m-1), dtype='int')
    i11 = 0
    i12 = 0
    Ec = []
    for i in range(n):
        if i+1 not in Ts:
            Ec.append(i+1)
            B11[:, i11] = B[:, i]
            i11 += 1
        else:
            B12[:, i12] = B[:, i]
            i12 += 1
    E = np.hstack((Ec, Ts))
    return E, np.hstack((B11, B12)), B11, B12


@decoratorFunc
def getOrderB_():
    """
    将基本关联矩阵按照余树弦在前，树枝在后的顺序排列
    """
    text.insert('insert', '基本关联矩阵B1为')
    B = getB_(1)
    # T = [1, 3, 4, 6, 8]
    Ts = getT()
    # if len(Ts) == 0:
    #     setT()
    #     Ts = getT()
    B11 = np.zeros((m-1, n-m+1), dtype='int')
    B12 = np.zeros((m-1, m-1), dtype='int')
    i11 = 0
    i12 = 0
    Ec = []
    for i in range(n):
        if i+1 not in Ts:
            Ec.append(i+1)
            B11[:, i11] = B[:, i]
            i11 += 1
        else:
            B12[:, i12] = B[:, i]
            i12 += 1
    E = np.hstack((Ec, Ts))
    return np.hstack((B11, B12))


@decoratorFunc
def getCf():
    """
    求取独立回路矩阵
    """
    text.insert('insert', '独立回路矩阵为:')
    _, _, B11, B12 = getOrderB()
    C12 = np.dot(-B11.T, np.linalg.inv(B12).T).astype(int)  # 书上求C12的公式
    I = np.eye(n-m+1, dtype='int')
    Cf = np.hstack((I, C12))
    return Cf


def getCf_():
    """"
    getCf的副本，供求取独立割集使用
    """
    _, _, B11, B12 = getOrderB()
    C12 = np.dot(-B11.T, np.linalg.inv(B12).T).astype(int)
    # C12.dtype = 'int'
    # I = np.eye(n-m+1, dtype='int')
    # print("当前弧的排列顺序为：\n"+str(E))
    return C12


@decoratorFunc
def getSf():
    """
    求取独立割集矩阵
    """
    text.insert('insert', '独立割集矩阵为:')
    C12 = getCf_()
    I = np.eye(m-1, dtype='int')
    S11 = -C12.T
    Sf = np.hstack((S11, I))
    return Sf


def getHelp():
    mnt = ' {:^6}  |  {:^11}'
    print('\n输入命令执行对应的操作（不区分大小写）：')
    print(mnt.format('A', '查看邻接矩阵'))
    print(mnt.format('B', '查看关联矩阵'))
    print(mnt.format('D', '查看道路矩阵'))
    print(mnt.format('P', '查看可达矩阵'))
    print(mnt.format('St', '查看生成树数目'))
    print(mnt.format('T', '查看当前树枝'))
    print(mnt.format('Cf', '独立回路矩阵'))
    print(mnt.format('Sf', '独立割集矩阵'))
    print(mnt.format('SetT', '修改树枝'))
    print(mnt.format('Q', '退出程序'))


# 以下为界面布局的代码
E_info1 = tk.Label(win_up0, text='E：'+str(E),
                   height='1', font='Times 14 bold')
E_info1.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, pady=5)
G_info1 = tk.Label(win_up0, text='D1：'+str(D1),
                   height='1', font='Times 14 bold')
G_info1.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, pady=5)
G_info2 = tk.Label(win_up0, text='D2: '+str(D2),
                   height='1', font='Times 14 bold')
G_info2.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, pady=5)
T_info = tk.Label(win_up0, text='当前树枝：'+str(getT()),
                  height='1', font='Times 14 bold')
draw_button = tk.Button(
    win_up1, text='绘制\n图形',
    font='Times 16 normal', command=drawG, height=3, width=15)
tk.Label(win_up1, text='                    ', width=16).pack(side='left')
T_info.pack(side=tk.LEFT, expand=tk.NO, fill=tk.X, pady=5)
draw_button.pack(side='right', expand='yes', padx=3, pady=3)

A_button = tk.Button(win_down_left2, text='邻接矩阵', width=12,
                     height=2, command=getA3)
A_button.pack(side='top', pady=8)
B_button = tk.Button(win_down_left2, text='基本关联矩阵', width=12,
                     height=2, command=getOrderB_)
B_button.pack(side='top', pady=8)
D_button = tk.Button(win_down_left2, text='道路矩阵', width=12,
                     height=2, command=getD)
D_button.pack(side='top', pady=8)
P_button = tk.Button(win_down_left2, text='可达矩阵', width=12,
                     height=2, command=getP)
P_button.pack(side='top', pady=8)
Cf_button = tk.Button(win_down_left2, text='独立回路', width=12,
                      height=2, command=getCf)
Cf_button.pack(side='top', pady=8)

Sf_button = tk.Button(win_down_left1, text='独立割集', width=12,
                      height=2, command=getSf)
Sf_button.pack(side='top', pady=8, padx=3)
T_button = tk.Button(win_down_left1, text='生成树数目', width=12,
                     height=2, command=getSt)
T_button.pack(side='top', pady=8, padx=3)
help_button = tk.Button(win_down_left1, text='修改数据', width=12,
                        height=2, command=modifyG)
help_button.pack(side='top', pady=8, padx=3)
setT_button = tk.Button(win_down_left1, text='设置树枝', width=12,
                        height=2, command=setT)
setT_button.pack(side='top', pady=8, padx=3)
Q_button = tk.Button(win_down_left1, text='退出程序', width=12,
                     height=2, command=win.destroy)
Q_button.pack(side='top', pady=8, padx=3)


def about():
    msg.showinfo('关于', '代码已在Github开源，欢迎fork和star\n项目地址：')


def helps():
    msg.showinfo('帮助', '此程序基于矿井通风网络图论，代码包含一些图论相关知识的基本代码实现\n内容包括：\n邻接\
矩阵\n基本关联矩阵\n基本回路矩阵\n独立割集矩阵\n生成树数目\n道路矩阵\n以及简单的绘图等\n\n\
注意事项：\n使用前请确保文件数据文件存在，数据文件包括两个:\nGVE_G.txt      储存图的空间结构\nGVE_T.txt       储存树枝\n输入时确保文件末尾不含有空格\n点击按钮即可开始使用。')


menubar = tk.Menu(win)
about_menu = tk.Menu(menubar, tearoff=0)
menubar.add_command(label='帮助', command=helps)
menubar.add_command(label='关于', command=about)
# about_menu.add_command()
# about_menu.add_cascade(label='关于',command=getA)

if len(Ts) != p:
    text.insert(
        'insert', "当前树枝数目为{}不正确\n期待的树枝数目为{}\n若要求取独立回路矩阵和独立\
割集矩阵\n请先点击设置树枝按钮修改\n其他操作不影响".format(len(Ts), p))
else:

    text.insert(
        tk.INSERT, '相关数据文件已读取完毕，欢迎使用。\n\n小组成员：\n周慧\n尹邦高\n杨秀光\n汤抑鑫\n朱进鹏\n')

win.config(menu=menubar)
# 显示程序窗口
win.mainloop()
