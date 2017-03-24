import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import sparse,  io, linalg
import pickle
from tqdm import tqdm_notebook as tqdm
import networkx as nx
import snap
import os

def plot_log_log(records):
    fig, ax = plt.subplots()
    # Remove zero values
    records = records[np.array([idx for idx in range(records.shape[0]) if records[idx, 0]>0 and records[idx,1] > 0]),:]
    #median_x, median_y = zip(*sorted((xVal, np.median([yVal for a, yVal in zip(records[:,0], records[:,1]) if xVal==a])) for xVal in set(records[:,0])))
    plt.scatter([np.log(r) for r in records[:,0]], [np.log(r) for r in  records[:,1]],s=0.5)
    # Todo add median code
    
def plot(records):
    fig, ax = plt.subplots()
    # Remove zero values
    records = records[np.array([idx for idx in range(records.shape[0]) if records[idx, 0]>0 and records[idx,1] > 0]),:]
    #median_x, median_y = zip(*sorted((xVal, np.median([yVal for a, yVal in zip(records[:,0], records[:,1]) if xVal==a])) for xVal in set(records[:,0])))
    plt.scatter([(r) for r in records[:,0]], [(r) for r in  records[:,1]],s=0.5)
    # Todo add median code
    
    
def plot_log_log_summary(records,xlabel,ylabel,B=20,summary=np.median,discrete=False):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Selecting only positive values
    records = records[np.array([idx for idx in range(records.shape[0]) if records[idx, 0]>0 and records[idx,1] > 0]),:]
    xs = [np.log(record[0]) for record in records]
    ys = [np.log(record[1]) for record in records]
    plt.scatter([(x) for x in xs], [(y) for y in  ys],s=0.5)
    if discrete==False:
        x_max = max(xs) - min(xs)
        # bin into default 20 values
        binned_xs = [min(xs)+(x_max*i)/float(B) for i in range(B+2)]
        median_ys = []
        err = []
        for i in range(B+1):
            # find median y values for x values in [binned_xs[i], binned_xs[i+1])
            current_ys = [ys[j] for j in range(len(xs)) if binned_xs[i] <= xs[j] and xs[j] < binned_xs[i+1]]
            current_y = summary(current_ys)
            median_ys += [current_y]
            #err += [[median_ys-np.std(current_ys)*0.5,median_ys+np.std(current_ys)*0.5]]
            err += [np.std(current_ys)]
        plt.errorbar((np.asarray(binned_xs[:-1])+np.asarray(binned_xs[1:]))/2.0, median_ys, marker='o', yerr=err,color='r')
    else:
        binned_xs = np.unique(xs)
        median_ys = []
        err = []
        for x in binned_xs:
            current_ys = [ys[j] for j in range(len(xs)) if xs[j] == x]
            current_y = summary(current_ys)
            median_ys += [current_y]
            #err += [[median_ys-np.std(current_ys)*0.5,median_ys+np.std(current_ys)*0.5]]
            err += [np.std(current_ys)]
        plt.errorbar(binned_xs, median_ys, marker='o', yerr=err,color='r')


    
def plot_summary(records,B=20,summary=np.median, discrete=False):
    fig, ax = plt.subplots()
    #records = records[np.array([idx for idx in range(records.shape[0]) if records[idx, 0]>0 and records[idx,1] > 0]),:]


    xs = [(record[0]) for record in records]
    ys = [(record[1]) for record in records]
    plt.scatter([(x) for x in xs], [(y) for y in  ys],s=0.5)


    if discrete==False:
        x_max = max(xs) - min(xs)
        # bin into default 20 values
        binned_xs = [min(xs)+(x_max*i)/float(B) for i in range(B+1)]
        median_ys = []
        err = []
        for i in range(B):
            # find median y values for x values in [binned_xs[i], binned_xs[i+1])
            current_ys = [ys[j] for j in range(len(xs)) if binned_xs[i] <= xs[j] and xs[j] < binned_xs[i+1]]
            current_y = summary(current_ys)
            median_ys += [current_y]
            #err += [[median_ys-np.std(current_ys)*0.5,median_ys+np.std(current_ys)*0.5]]
            err += [np.std(current_ys)]
        plt.errorbar(binned_xs[1:], median_ys, marker='o', yerr=err,color='r')
    else:
        binned_xs = np.unique(xs)
        median_ys = []
        err = []
        for x in binned_xs:
            current_ys = [ys[j] for j in range(len(xs)) if xs[j] == x]
            current_y = summary(current_ys)
            median_ys += [current_y]
            #err += [[median_ys-np.std(current_ys)*0.5,median_ys+np.std(current_ys)*0.5]]
            err += [np.std(current_ys)]
        plt.errorbar(binned_xs, median_ys, marker='o', yerr=err,color='r')




def read_all_graphs(dir):
    G=nx.Graph()
    for filename in os.listdir(dir):
        temp_G = nx.read_adjlist(dir+filename,nodetype=int)
        for e in tqdm(temp_G.edges_iter()):
            if G.has_edge(e[0],e[1]):
                G[e[0]][e[1]]['weight']+=1
            else:
                G.add_edge(e[0],e[1],weight=1)
    G[e[0]][e[1]]['weight_inv']=1/float(G[e[0]][e[1]]['weight'])
    G[e[0]][e[1]]['weight_inv_exp']=np.exp(-float(G[e[0]][e[1]]['weight'])/3.0)
    return G

def bad_sim_score_1(G):
    pwl = []
    for e in tqdm(G.edges()):
        wt = G[e[0]][e[1]]['weight']#/float(G.degree(e[0],weight='weight')+G.degree(e[1],weight='weight'))
        nbr1 = set(G.neighbors(e[0]))
        nbr2 = set(G.neighbors(e[1]))
        common_nbrs= set.intersection(nbr1,nbr2)
        all_nbrs = set.union(nbr1,nbr2)
        #sim = sum([1/np.log(len(oregon_G.neighbors(v))) for v in common_nbrs])/sum(
        #    [1/np.log(len(oregon_G.neighbors(v))) for v in all_nbrs])
        sim = len(set.intersection(nbr1,nbr2))/float(len(set.union(nbr1,nbr2)))
        #sim = len(set.intersection(nbr1,nbr2))
        pwl += [(wt, sim)]
    return np.asarray(pwl)

def sim_score_1(G):
    pwl = []
    for e in tqdm(G.edges()):
        wt = G[e[0]][e[1]]['weight']/float(G.degree(e[0],weight='weight')+G.degree(e[1],weight='weight'))
        nbr1 = set(G.neighbors(e[0]))
        nbr2 = set(G.neighbors(e[1]))
        common_nbrs= set.intersection(nbr1,nbr2)
        all_nbrs = set.union(nbr1,nbr2)
        #sim = sum([1/np.log(len(oregon_G.neighbors(v))) for v in common_nbrs])/sum(
        #    [1/np.log(len(oregon_G.neighbors(v))) for v in all_nbrs])
        sim = len(set.intersection(nbr1,nbr2))/float(len(set.union(nbr1,nbr2)))
        #sim = len(set.intersection(nbr1,nbr2))
        pwl += [(wt, sim)]
    return np.asarray(pwl)


def sim_score_2(G):
    pwl = []
    for e in tqdm(G.edges()):
        wt = G[e[0]][e[1]]['weight']/float(G.degree(e[0],weight='weight')+G.degree(e[1],weight='weight'))
        nbr1 = set(G.neighbors(e[0]))
        nbr2 = set(G.neighbors(e[1]))
        common_nbrs= set.intersection(nbr1,nbr2)
        all_nbrs = set.union(nbr1,nbr2)
        sim = sum([1/np.log(len(G.neighbors(v))+1) for v in common_nbrs])/sum([1/np.log(len(G.neighbors(v))+1) for v in all_nbrs])
        #sim = len(set.intersection(nbr1,nbr2))/float(len(set.union(nbr1,nbr2)))
        #sim = len(set.intersection(nbr1,nbr2))
        pwl += [(wt, sim)]
    return np.asarray(pwl)


def self_similarity(G, distances,size=1000, weight='weight_inv'):
    pwl = []
    for node in tqdm(np.random.choice(G.nodes(),size=size)):
        for d in distances:
            pwl += [(d,len(nx.single_source_dijkstra_path_length(G,node,cutoff=d,weight=weight)))]
    pwl=np.asarray(pwl)
    return pwl
