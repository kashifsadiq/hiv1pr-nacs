# HIV-1 Protease-substrates Near Attack Conformation Analysis

#Author: S. Kashif Sadiq

#Correspondence: kashif.sadiq@embl.de, Affiliation: 1. Heidelberg Institute for Theoretical Studies, HITS gGmbH 2. European Moelcular Biology Laboratory

#This module contains core functions for molecular dynamics (MD) simulation analyses of HIV-1 protease subtrate near attack conformations for the manuscripts:


#S.K. Sadiq‡ (2020) Catalysts, Fine-tuning of sequence-specificity by near attack conformations in enzyme-catalyzed peptide hydrolysis, 10 (6) 684

#Sadiq, S.K. ‡ and Coveney P.V. (2015). J Chem Theor Comput. Computing the role of near attack conformations in an enzyme-catalyzed nucleophilic bimolecular reaction. 11 (1), pp 316–324


########################################################################################################################################


import sys
import glob
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#from lmfit import minimize, Parameters, Parameter, report_fit
#import numdifftools
from scipy.special import lambertw
import math
import os
from distutils.version import LooseVersion
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import itertools


#####################################################

#Functions to load data files

#read text file of rows
def read_file(fname):
    """
    reads a file containing a rows of text
    """
#    a = []
    with open(fname) as f:
        a=f.readlines()
    return a


def read_int_matrix(fname):
    """
    reads a file containing a matrix of integer numbers
    """
    a = []
    with open(fname) as f:
        for line in f:
            row = line.rstrip().split()
            a.append(row)
    foo = np.array(a)
    bar = foo.astype(np.int)
    return bar

#Read in matrix of floats from file
def read_float_matrix(fname):
    """
    reads a file containing a matrix of floating point numbers
    """
    a = []
    with open(fname) as f:
        for line in f:
            row = line.rstrip().split()
            a.append(row)
    foo = np.array(a)
    bar = foo.astype(np.float)
    return bar


def load_dataset_from_simlist(data_dir, syslist, nsims):
    
    for j in range(len(syslist)):
        #List of all simulations in the corresponding subdirectory
        datlist=np.sort(np.array([int(x[2:-4]) for x in os.listdir(data_dir+syslist[j])]))
        #print(datlist)
        
        for i in range(nsims):
    
            d=read_float_matrix(data_dir+syslist[j]+'/1-'+str(datlist[i])+ '.dat')
            d=d[:,3:]
    
            if i==0:
                data = d
            else:
                data = np.vstack((data,d))
    
        if j==0:
            alldata = data
        else:
            alldata = np.dstack((alldata,data))

        print("Loaded:"+str(syslist[j]))

    return alldata

def load_dataset_from_range(data_dir, syslist, nsims, nunits):

    for k in range(len(syslist)):
        for j in range(nsims):
            for i in range(nunits):
            
                d=read_float_matrix(data_dir+syslist[k]+'/'+str(j+1)+'-'+str(i+1)+'.dat')
                d=d[:,3:]
            
            
                if i==0 and j==0:
                    data = d
                else:
                    data = np.vstack((data,d))
        if k==0:
            apodata= data
        else:
            apodata = np.dstack((apodata,data))

        print("Loaded:"+str(syslist[k]))
    
    return apodata


#####################################################

#Distribution plotting functions

def plot_distribution_single_panel(plt, axes, Xdata,junction_list,sys,colno, xpoint,xlab=r'$d_{n} (\AA)$',ylab=r'$d_{n} (\AA)$'):

    X_plot = np.linspace(0, 10, 1000)[:, np.newaxis]
    X=Xdata[:,colno,sys][:, np.newaxis]
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.75).fit(X)
    log_dens = kde.score_samples(X_plot)
        
    
    axes.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    axes.text(0.5, 0.65, junction_list[sys],fontsize=20)
    axes.set_xticks([x for x in range(0,12,2)])
    axes.set_yticks([y/10 for y in range(0,8,2)])        
    axes.tick_params(axis='x', labelsize=30)
    axes.tick_params(axis='y', labelsize=30)
    axes.set_xlim(0, 9)
    axes.set_ylim(0, 0.8)
    
        
    yline=[y/10 for y in range(0,12)]
    axes.plot(xpoint, yline,ls='--',color='k')
    
    axes.set_xlabel(xlab,fontsize=30)
    axes.set_ylabel(ylab,fontsize=30)
    
    return plt, log_dens


def plot_distribution_multi_panel(plt, axes, Xdata,junction_list,sys,colno,xpoint,i,j):
    
    X_plot = np.linspace(0, 10, 1000)[:, np.newaxis]
    X=Xdata[:,colno,sys][:, np.newaxis]
    kde = KernelDensity(kernel='epanechnikov', bandwidth=0.75).fit(X)
    log_dens = kde.score_samples(X_plot)

    axes[i, j].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    axes[i, j].text(0.5, 0.65, junction_list[sys],fontsize=20)
    axes[i, j].set_xticks([x for x in range(0,12,2)])
    axes[i, j].set_yticks([y/10 for y in range(0,8,2)])        
    axes[i, j].tick_params(axis='x', labelsize=30)
    axes[i, j].tick_params(axis='y', labelsize=30)
    axes[i, j].set_xlim(0, 9)
    axes[i, j].set_ylim(0, 0.8)

    yline=[y/10 for y in range(0,12)]
    axes[i, j].plot(xpoint, yline,ls='--',color='k')

    return plt, log_dens



#####################################################


def linear_func(x, a, b):

    return a *x + b

def linear_fit(xdata, ydata):
    popt, pcov = curve_fit(linear_func, xdata, ydata)
    y_fit=linear_func(xdata, *popt)
    # residual sum of squares
    ss_res = np.sum((ydata - y_fit) ** 2)
    # regression sum of squares
    ss_reg = np.sum((y_fit - np.mean(ydata)) ** 2) 
    # total sum of squares
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    # r-squared
    #r2 = 1 - (ss_res / ss_tot)
    r2 = (ss_reg / ss_tot)
    
    return y_fit, popt, r2





#####################################################
### NAC and data processing functions


#Read in matrix of floats from file
def subsets(numsets,datasize):
    """
    Makes a list of equally sized lists of elements within a data set for use in averaging of functions 
    """
    subsize=int(datasize/numsets)
    range_list=[]
    for i in range(numsets):
        set_no=i
        range_list.append([x for x in range(set_no*subsize,(set_no+1)*subsize)])
        
    
    return range_list


def bootstrapping(numsets,datasize):
    """
    Makes a list of equally sized lists of elements within a data set for use in averaging of functions 
    chosen from a random selection of the whole data set
    """
    subsize=int(datasize/numsets)
    range_list=[]
    for i in range(numsets):
        range_list.append(np.random.randint(datasize, size=subsize).tolist())

    return range_list
    

def NAC_calc(data,nwb_cutoff,BDdist,BDanglow,BDanghigh,kBT):
    
    """
    Calculates N,count_nwb, count_nac, rho_nwb, rho_nac, G_nwb, G_nac data for given data set  
    """
    offset=0.0000000000001
    
    N=np.shape(data)[0]
    count_nwb=np.sum(data[:,1]<=nwb_cutoff)    
    if count_nwb==N:
        rho_nwb=1/offset
    else:
        rho_nwb=count_nwb/(N-count_nwb)

    #count_nac=np.sum(np.logical_and(data[:,2]<=BDdist, data[:,3]>=BDanglow, data[:,3]<=BDanghigh))
    count_nac=np.sum(np.logical_and(data[:,1]<=nwb_cutoff,np.logical_and(data[:,2]<=BDdist, np.logical_and(data[:,3]>=BDanglow,data[:,3]<=BDanghigh))))
    rho_nac=count_nac/count_nwb
    if rho_nac==0:
        rho_nac=rho_nac+offset
    
    G_nwb = -kBT*math.log(rho_nwb)
    G_nac = -kBT*math.log(rho_nac)

    
    return N,count_nwb, count_nac, rho_nwb, rho_nac, G_nwb, G_nac



def total_Gnac(alldata,nwb_cut,BDdist,BDanglow,BDanghigh,kBT):
    """
    Calculates N,count_nwb, count_nac, rho_nwb, rho_nac, G_nwb, G_nac, from total data set for each complex system
    """
    mat=[]
    for j in range(np.shape(alldata)[2]):    
        data = alldata[:,:,j]
        N,count_nwb, count_nac, rho_nwb, rho_nac, G_nwb, G_nac = NAC_calc(data,nwb_cut[j],BDdist,BDanglow,BDanghigh,kBT)
        mat.append([count_nwb, count_nac, rho_nwb, rho_nac, G_nwb, G_nac])
    mat=np.array(mat)
    
    return mat



def plot_Gvary_witherrors(plt,G,min_dn,max_dn,xlim,ylim,apo=False):
    """
    PLots GNAC as a function of the d_n threshold set for partioning bound and unbound water molecules
    """
    plt.rc('text', usetex=True)
    plt.figure(figsize=(12,9))
    color_list=['k','r','b','g','m','y','c','k','r']
    marker_list=['o','o','o','o','o','s','s','s','s']
    junction_list=['MA-CA','CA-SP1','SP1-NC','NC-SP2','SP2-p6','TFR-PR','PR-RT','RT-RH','RH-IN']
    
    dn=[x/10 for x in range(min_dn,max_dn)]
    
    if apo==False:
        
        for i in range(np.shape(G)[2]):
            if i!=6:
                m = G[:,0, i]
                s = G[:,1, i]
                plt.plot(dn, m, ls='-', label=junction_list[i],color=color_list[i],marker=marker_list[i], ms=10, mec=color_list[i], mfc=color_list[i])
                plt.fill_between(dn, m-s, m+s, color=color_list[i],alpha=0.2)
    else:
        for i in range(np.shape(G)[2]):
            
            m = G[:,0, i]
            s = G[:,1, i]
            plt.plot(dn, m, ls='-', label=junction_list[i],color=color_list[i],marker=marker_list[i], ms=10, mec=color_list[i], mfc=color_list[i])
            plt.fill_between(dn, m-s, m+s, color=color_list[i],alpha=0.2)
    
    plt.legend(loc='best',fontsize = 20)
    plt.xlabel(r'$d^{p}_{n} (\AA)$',fontsize=30)
    plt.ylabel(r'$\Delta G^{\ddagger}_{NAC} (kcal/mol)$',fontsize=30)
    plt.xticks([x/10 for x in range(30,70,5)], fontsize=30)    
    plt.yticks([y/10 for y in range(15,70,5)], fontsize=30)
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
     
    return

def plot_Gvary(plt,G,min_dn,max_dn,xlim,ylim,apo=False):
    """
    PLots GNAC as a function of the d_n threshold set for partioning bound and unbound water molecules
    Using total data set
    """
    plt.rc('text', usetex=True)
    plt.figure(figsize=(12,9))
    color_list=['k','r','b','g','m','y','c','k','r']
    marker_list=['o','o','o','o','o','s','s','s','s']
    junction_list=['MA-CA','CA-SP1','SP1-NC','NC-SP2','SP2-p6','TFR-PR','PR-RT','RT-RH','RH-IN']
    
    dn=[x/10 for x in range(min_dn,max_dn)]
    
    if apo==False:
        
        for i in range(np.shape(G)[0]):
            if i!=6:
                m = G[i,:]
                plt.plot(dn, m, ls='-', label=junction_list[i], color=color_list[i],marker=marker_list[i], ms=10, mec=color_list[i], mfc=color_list[i])
    else:
        for i in range(np.shape(G)[0]):
            m = G[i,:]
            plt.plot(dn, m, ls='-', label=junction_list[i], color=color_list[i],marker=marker_list[i], ms=10, mec=color_list[i], mfc=color_list[i])
    
    plt.legend(loc='best',fontsize = 20)
    plt.xlabel(r'$d^{p}_{n} (\AA)$',fontsize=30)
    plt.ylabel(r'$\Delta G^{\ddagger}_{NAC} (kcal/mol)$',fontsize=30)
    plt.xticks([x/10 for x in range(30,70,5)], fontsize=30)    
    plt.yticks([y/10 for y in range(15,70,5)], fontsize=30)
    plt.xlim(xlim[0],xlim[1])
    plt.ylim(ylim[0],ylim[1])
     
    return


def calc_Gvary_witherrors(alldata,range_list,numsets,min_dn,max_dn,BDdist,BDanglow,BDanghigh,kBT):
    """
    Calculates GNAC as a function of the d_n threshold set for partioning bound and unbound water molecules
    Using subsets of data and creating mean and standard deviations
    """

    for k in range(np.shape(alldata)[2]):
        G=[]
        data = alldata[:,:,k]
        for j in range(min_dn,max_dn):    
            mat=[]       
            for i in range(numsets):
                N,count_nwb, count_nac, rho_nwb, rho_nac, G_nwb, G_nac = NAC_calc(data[range_list[i]],j/10,BDdist,BDanglow,BDanghigh,kBT)
                mat.append([count_nwb, count_nac, rho_nwb, rho_nac, G_nwb, G_nac])    

            mat=np.array(mat)        
            G.append([np.mean(mat[:,-1]), np.std(mat[:,-1])])
        
        G=np.array(G)
            
        if k == 0:
            G_nwbvary = G
        else:
            G_nwbvary = np.dstack((G_nwbvary,G))
    
    return G_nwbvary
    
def calc_Gvary(alldata,min_dn,max_dn,BDdist,BDanglow,BDanghigh,kBT):
    """
    Calculates GNAC as a function of the d_n threshold set for partioning bound and unbound water molecules
    Using total data set - so no mean or std
    """
    
    for k in range(np.shape(alldata)[2]):
        G=[]
        data = alldata[:,:,k]
        for j in range(min_dn,max_dn):                   
            N,count_nwb, count_nac, rho_nwb, rho_nac, G_nwb, G_nac = NAC_calc(data,j/10,BDdist,BDanglow,BDanghigh,kBT)           
            G.append(G_nac)
        G=np.array(G)
            
        if k == 0:
            G_nwbvary = G
        else:
            G_nwbvary = np.vstack((G_nwbvary,G))
    
    return G_nwbvary


def confs(data,BDdist,BDanglow,BDanghigh,nwb_cut,sys):
    """
    Returns Logical True,False in elements of array of size (1,data) that satisfy corresponding nac and nwb definitions
    Based on Burgi-Dunitz distance and angle criteria and near water binding threshold
    """
    #Near water binding definition
    nwb=data[:,1]<=nwb_cut[sys]
    
    #NAC conformation definition
    bd_d=data[:,2]<=BDdist
    bd_ang=np.logical_and(data[:,3]>=BDanglow,data[:,3]<=BDanghigh)
    nac=np.logical_and(bd_d,bd_ang)
    
    
    return nwb, nac

def hbond_analysis(data,hb_dist,hb_ang):
    """
    Returns hbond Logical True,False in elements of array of size (1,data) that satisfy corresponding hbond definitions
    Based on hbond distance and angle criteria
    """
    # columns in data files are as followS:
    # NWI $NWD $ATTACK $ATTACKA 
    # $H1d $H1a (D25:O2 - D25:H - p1:O)
    #$H21d $H21a $H22d $H22a (D25:O2 - D25:H - D25':O1, D25:O2 - D25:H - D25':O2)
    #$H31d $H31a $H32d $H32a (p1':N - p1':H - D25':O1, p1':N - p1':H - D25':O2)
    # $H411d $H411a $H412d $H412a (WAT:O - WAT:H1 - D25':O1, WAT:O - WAT:H2 - D25':O1)
    #$H421d $H421a $H422d $H422a (WAT:O - WAT:H1 - D25':O2, WAT:O - WAT:H2 - D25':O2)
    # $H511d $H511a $H512d $H512a (WAT:O - WAT:H1 - D25:O1, WAT:O - WAT:H2 - D25:O1)
    #$H521d $H521a $H522d $H522a (WAT:O - WAT:H1 - D25:O2, WAT:O - WAT:H2 - D25:O2)
    #In this terminology the monoprotonated aspartyl is on D25:O2
        
    
    #Hbond definitions in data using terminology in original NAC paper
    #(D25:O2 - D25:H - p1:O)
    hb1=np.logical_and(data[:,4]<=hb_dist,data[:,5]>=hb_ang)

    #$H21d $H21a $H22d $H22a (D25:O2 - D25:H - D25':O1, D25:O2 - D25:H - D25':O2)
    hb21=np.logical_and(data[:,6]<=hb_dist,data[:,7]>=hb_ang)
    hb22=np.logical_and(data[:,8]<=hb_dist,data[:,9]>=hb_ang)
    hb2=np.logical_or(hb21,hb22)

    # $H411d $H411a $H412d $H412a (WAT:O - WAT:H1 - D25':O1, WAT:O - WAT:H2 - D25':O1)
    #$H421d $H421a $H422d $H422a (WAT:O - WAT:H1 - D25':O2, WAT:O - WAT:H2 - D25':O2)
    hb311=np.logical_and(data[:,14]<=hb_dist,data[:,15]>=hb_ang)
    hb312=np.logical_and(data[:,16]<=hb_dist,data[:,17]>=hb_ang)
    hb31=np.logical_or(hb311,hb312)
    hb321=np.logical_and(data[:,18]<=hb_dist,data[:,19]>=hb_ang)
    hb322=np.logical_and(data[:,20]<=hb_dist,data[:,21]>=hb_ang)
    hb32=np.logical_or(hb321,hb322)
    hb3=np.logical_or(hb31,hb32)

    # $H511d $H511a $H512d $H512a (WAT:O - WAT:H1 - D25:O1, WAT:O - WAT:H2 - D25:O1)
    hb41=np.logical_and(data[:,22]<=hb_dist,data[:,23]>=hb_ang)
    hb42=np.logical_and(data[:,24]<=hb_dist,data[:,25]>=hb_ang)
    hb4=np.logical_or(hb41,hb42)

    #$H521d $H521a $H522d $H522a (WAT:O - WAT:H1 - D25:O2, WAT:O - WAT:H2 - D25:O2)
    #This is actually a chemically independent hbond from H511 etc
    hb51=np.logical_and(data[:,26]<=hb_dist,data[:,27]>=hb_ang)
    hb52=np.logical_and(data[:,28]<=hb_dist,data[:,29]>=hb_ang)
    hb5=np.logical_or(hb51,hb52)

    #$H31d $H31a $H32d $H32a (p1':N - p1':H - D25':O1, p1':N - p1':H - D25':O2)
    hb61=np.logical_and(data[:,10]<=hb_dist,data[:,11]>=hb_ang)
    hb62=np.logical_and(data[:,12]<=hb_dist,data[:,13]>=hb_ang)
    hb6=np.logical_or(hb61,hb62)

    #Statistics of count numbers using sums of individual hbond definitions versus when combined using logical OR
    stat=[]
    stat.append([np.sum(hb1)])
    stat.append([np.sum(hb21), np.sum(hb22), np.sum(hb21)+np.sum(hb22), np.sum(hb2)])
    stat.append([np.sum(hb311),np.sum(hb312),np.sum(hb321),np.sum(hb322), np.sum(hb311)+np.sum(hb312)+np.sum(hb321)+np.sum(hb322),np.sum(hb3)])
    stat.append([np.sum(hb41), np.sum(hb42), np.sum(hb41)+np.sum(hb42), np.sum(hb4)])
    stat.append([np.sum(hb51), np.sum(hb52), np.sum(hb51)+np.sum(hb52), np.sum(hb5)])
    stat.append([np.sum(hb61), np.sum(hb62), np.sum(hb61)+np.sum(hb62), np.sum(hb6)])
        
    return hb1,hb2,hb3,hb4,hb5,hb6,stat

def hb_states(hb):
    """
    Returns list of Logical True,False arrays that satisfy hbond combinations definitions 
    of all combinations of individual hbonds listed in hb
    """
    no_hb = [np.invert(x) for x in hb]
    all_hb = [[no_hb[x], hb[x]] for x in range(len(hb))]        
    lst = list(itertools.product([0, 1], repeat=len(hb)))    

    hb_states=[]
    for i in range(len(lst)):
        x=np.ones((1, np.shape(hb[0])[0]), dtype=bool)[0]
        for j in range(len(hb)):
            x=np.logical_and(x,all_hb[j][lst[i][j]])
        
        hb_states.append(x)
    
    return hb_states, lst


def kBT_func(T):
    
    h_planck=6.62607015*10**-34 #Js
    kB_joules = 1.380649*10**-23 #JK^-1
    kB_over_h= kB_joules/h_planck
    N_avo=6.02214076*10**23
    joules_to_kcal=1/4184
    kB= kB_joules*N_avo*joules_to_kcal

    kBT=kB*T
    kBT_over_h = kB_over_h*T
    
    return kBT, kBT_over_h


