"""
Finding the worker computation time for the proposed optimal scheme
Having two random matrices A and B
We have n workers and s = n - kA * kB stragglers.
Storage fraction gammaA = 1/kA and gammaB = 1/kB.
Matrices A and B are partitioned into LCM(n,kA) and kB block columns.
ellAu and ellAc are number of uncoded and coded blocks for A
"""

from __future__ import division
import numpy as np
import time
from scipy.sparse import csr_matrix
from mpi4py import MPI
import sys
import warnings
from scipy.sparse import rand,vstack

if not sys.warnoptions:
    warnings.simplefilter("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    n = 24;
    gammaA = 1/4;
    gammaB = 1/5;
    kA = int(1/gammaA)
    kB = int(1/gammaB)
    sm = n - kA*kB

    r = 1500 ;
    t = 1200 ;
    w = 1350 ;
    A = rand(r, t, density=0.01, format="csr")
    B = rand(w, t, density=0.01, format="csr")
   
    DeltaA = 24
    ellA = int(DeltaA*gammaA);
    DeltaB = kB
    ellB = 1;
    ellAu = int(DeltaA/n*kB);
    ellAc = ellA - ellAu  

    sub_ellA = int(r/DeltaA);
    As = {};
    for j in range(0,DeltaA):
        As[j] = A[j*sub_ellA:(j+1)*sub_ellA,:];

    sub_ellB = int(w/DeltaB);
    Bs = {};
    for j in range(0,DeltaB):
        Bs[j] = B[j*sub_ellB:(j+1)*sub_ellB,:];

    CMA = np.zeros((n,ellA))
    for i in range(0,n):
        kk = int(i*DeltaA/n)
        for j in range(0,ellAu):
            CMA[i,j] = np.mod(np.arange((kk+j),(kk+j+1)),DeltaA)
        for j in range(ellAu,ellA):
            CMA[i,j] = np.mod(CMA[i,j-1]+1,ellA)
   
    res_finAs = {}
    res_finAd = {}        
    res_finB = {}        
    Av = {};
    for j in range(0,DeltaA):
        Av[j] = csr_matrix.reshape(As[j],(1,sub_ellA*t))

    Ac = {}
    for i in range(0,ellA):
        mm = i
        Ac[i] = Av[i];    
        for j in range(1,kA):
            mm = mm + ellA
            Ac[i] = vstack([Ac[i],Av[mm]]) 

    Bv = {};
    for j in range(0,DeltaB):
        Bv[j] = csr_matrix.reshape(Bs[j],(1,sub_ellB*t))

    den = int(np.floor(DeltaB/(int(1+kB/sm)))+1)
    Bc = {}
    for i in range(0,n):
        mm = np.mod(i,DeltaB)
        Bc[i] = Bv[mm];    
        for j in range(1,den):
            Bc[i] = vstack([Bc[i],Bv[np.mod(mm+j,DeltaB)]])  
    
    for k in range(0,n):
        resA = {};
        resB = {};
        for i in range(0, ellAu):
            resA[i] = As[CMA[k,i]].todense()
        for i in range(0, ellAc):
            vecA = np.random.rand(1,kA)
            coded_submatA = vecA * Ac[CMA[k,ellAu+i]]
            resA[i+ellAu] = coded_submatA.reshape((sub_ellA, t))
        vecB = np.random.rand(1,den)
        coded_submatB = vecB * Bc[k]
        resB = coded_submatB.reshape((sub_ellB, t))        
        res_finAs[k] = np.vstack((resA[i]) for i in range (0,ellAu))
        res_finAs[k] = csr_matrix(res_finAs[k])
        res_finAd[k] = np.vstack((resA[i]) for i in range (ellAu,ellA))
        res_finB[k] = csr_matrix(resB)
        res_finB[k] = csr_matrix.transpose(res_finB[k])
        
    
    for k in range (0,n):
        comm.send(res_finAs[k], dest=k+1)
        comm.send(res_finAd[k], dest=k+1)
        comm.send(res_finB[k], dest=k+1)

    computation_time = np.zeros(n,dtype = float); 
    for i in range (0,n):
        computation_time[i] = comm.recv(source=i+1);
   
    for i in range (0,n):
        print("Computation time for processor %s is %s" %(i,computation_time[i]))


    comm.Abort()

else:
    smatAs = comm.recv(source=0);
    smatAd = comm.recv(source=0);
    smatB = comm.recv(source=0);
    Ad = csr_matrix(smatAd)

    start_time = time.time()
    result1 = smatAs * smatB
    result2 = Ad * smatB
    end_time = time.time()
    comm.send(end_time - start_time, dest=0)

