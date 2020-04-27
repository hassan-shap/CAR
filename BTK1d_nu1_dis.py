import numpy as np
import scipy as sp
from math import pi, tanh
from cmath import sqrt
from scipy.linalg import block_diag
import time


D1=0.4
D2=0.3
k0=1.2

s0=np.array([[1,0],[0,1]])
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])

# T matrix with random disorder

Nrep=2000
Nimp=30
Uimp=1.
L=10/D1
Z=0.4
Esw= np.linspace(-1.1,1.1,100)

Ham=  -k0* np.kron(sz,s0) +D1*np.kron(sy,sy) #+ G* np.kron(sz,sx)
vk= np.kron(s0,sz)+ D2*np.kron(sx,s0)
Hd= np.linalg.inv(vk)
    
out_dir='Dis1d_data_files/'
f1='nu1_D2_%.2f_Z_%.2f_Ni_%d_U_%.2f_L_%d' % (D2,Z,Nimp,Uimp,int(L*D1))
print(f1)
fname=out_dir+f1

t_timer=time.time()


Revecs_l= np.array([[0,1,0,0],[0,0,0,1]]).T
Tevecs_l= np.array([[1,0,0,0],[0,0,1,0]]).T
evecs_l= np.concatenate((Tevecs_l,Revecs_l),axis=1)

Tl=evecs_l
Tr=np.linalg.inv(evecs_l)
Tz=[[1-1j*Z,0,-1j*Z,0],\
    [0,1+1j*Z,0,-1j*Z],\
    [1j*Z,0,1+1j*Z,0],\
        [0,1j*Z,0,1-1j*Z]]

t0=time.time()
np.random.seed()
x_imp_mat=np.random.rand(Nimp,Nrep)*L
Ur_mat= np.random.rand(Nimp+1,Nrep)-0.5
Ree=np.zeros(len(Esw))
Reh=np.zeros(len(Esw))

for i_r in range(1000,Nrep):

    print(' ',i_r,end=' \r')
    x_imp= np.sort(x_imp_mat[:,i_r])
    x_imp=np.concatenate(([0],x_imp))
    
    for i_E in range(len(Esw)):
        E=Esw[i_E]
        Ht= np.dot(Hd,E*np.eye(4)-  Ham)
        evs, evecs = np.linalg.eig(Ht)

        evecs_inv=np.linalg.inv(evecs)
        Jmat = np.reshape(np.diag(np.dot(evecs_inv,np.dot(vk,evecs))),[4,])
        evecs_0= np.dot(evecs,np.diag(1/Jmat**0.5))

        def Asc(x):
            return np.dot(evecs,np.dot(np.diag(np.exp(1j*evs*x)),evecs_inv))

        Tmat=np.dot(Tl,Tz)
        for i_n in range(1,Nimp+1):
            Ur=Ur_mat[i_n,i_r]
            Ts=sp.linalg.expm(1j*Uimp*Ur*np.dot(Hd,np.kron(sz,sx)))
#             Ts=sp.linalg.expm(1j*Uimp*Ur*np.dot(Hd,np.kron(sz,s0)))
            Tmat=np.dot(Ts,np.dot(Asc(x_imp[i_n]-x_imp[i_n-1]),Tmat))
        Tmat=np.dot(Asc(L-x_imp[Nimp]),Tmat)
        Tt=np.dot(Tr,Tmat)
        TLL=Tt[np.ix_([2,3],[2,3])]
        TLR=Tt[np.ix_([2,3],[0,1])]
        x= -sp.linalg.solve(TLL,TLR)
        Ree[i_E] = np.abs(x[0,0])**2
        Reh[i_E] = np.abs(x[1,0])**2

    np.savez(fname+'_%d.npz' % (i_r) , E_list=Esw, Ree=Ree , Reh=Reh)

elapsed = time.time() - t_timer
print("Finished, elapsed time = %.0f " % (elapsed)+ "sec")

