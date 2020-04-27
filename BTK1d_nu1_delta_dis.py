import numpy as np
import scipy as sp
from math import pi, tanh
from cmath import sqrt
from scipy.linalg import block_diag
import time


D1=0.4
D2=0.
k0=1.2

s0=np.array([[1,0],[0,1]])
sx=np.array([[0,1],[1,0]])
sy=np.array([[0,-1j],[1j,0]])
sz=np.array([[1,0],[0,-1]])

# T matrix with random disorder

Nrep=1000
Nimp=80
L=40/D1
dl=L/Nimp
Z=0.4
Esw= np.linspace(-4.1,4.1,100)
    
out_dir='Dis1d_data_files/'
f1='nu1_delta_D12_%.2f_%.2f_Z_%.2f_Ni_%d_L_%d' % (D1,D2,Z,Nimp,int(L*D1))
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

Ree=np.zeros(len(Esw))
Reh=np.zeros(len(Esw))

vk= np.kron(s0,sz)+ D2*np.kron(sx,s0)
Hd= np.linalg.inv(vk)

t0=time.time()
np.random.seed()
D1s=D1*(np.random.rand(Nimp,Nrep)-0.5)

for i_r in range(Nrep):
    print(' ',i_r,end=' \r')
    
    for i_E in range(len(Esw)):
        E=Esw[i_E]
        
        Tmat=np.dot(Tl,Tz)
        for i_n in range(Nimp):
            Ham=  -k0* np.kron(sz,s0) +D1s[i_n,i_r]*np.kron(sy,sy)
            Tmat=np.dot(sp.linalg.expm(1j*np.dot(Hd,E*np.eye(4)-  Ham)*dl),Tmat)
            
        Tt=np.dot(Tr,Tmat)
        TLL=Tt[np.ix_([2,3],[2,3])]
        TLR=Tt[np.ix_([2,3],[0,1])]
        x= -sp.linalg.solve(TLL,TLR)
        Ree[i_E] = np.abs(x[0,0])**2
        Reh[i_E] = np.abs(x[1,0])**2

    np.savez(fname+'_%d.npz' % (i_r) , E_list=Esw, Ree=Ree , Reh=Reh)

elapsed = time.time() - t_timer
print("Finished, elapsed time = %.0f " % (elapsed)+ "sec")


