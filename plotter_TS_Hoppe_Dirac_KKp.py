#!/tmp/yes/bin python3

import numpy as np
from math import pi, sqrt, tanh
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import minimize
import time

hbar=1.05e-34
elec=1.6e-19
meV=elec*1e-3

vF=1e6
# B=10 # Tesla
# hwc=vF*sqrt(2*hbar*elec*B)
# lB=np.sqrt(hbar/(elec*B))
# m_sc=3*hwc #1e2*meV
# mu_sc=8*hwc #1e2*meV
# D1=0.5*hwc #10*meV
# D2=0.6*hwc #10*meV
# nu=0.3#sqrt(2)+0.1
# mu_n=nu*hwc #10*meV
# m_n=0.1*hwc

# lRx= 1*hwc
# lRy= 0.*hwc
# lso= 0.*hwc
# gs=0.0*hwc
# gn=0.3*hwc
# new parameters

B=6 # Tesla
hwc=vF*sqrt(2*hbar*elec*B)
lB=np.sqrt(hbar/(elec*B))
m_sc=3*hwc #1e2*meV
mu_sc=8*hwc #1e2*meV
# D1=0.02*hwc #10*meV
# D2=0.02*hwc #10*meV
# nu=0.0055#sqrt(2)+0.1
# mu_n=nu*hwc #10*meV
# m_n=0.001*hwc

# lRx= 0.01*hwc
# print(lRx/hwc)
# lRy= 0.*hwc
# lso= 0.*hwc
# gs=0.0*hwc
# gn=0.003*hwc
D1=0.3*hwc 
D2=0.2*hwc 
m_n=0.06*hwc

lRx= 0.2*hwc
lRy= 0.*hwc
lso= 0.*hwc
gs=0.0*hwc
gn=0.2*hwc
nu=0.45
mu_n=nu*hwc 

Ls=6*lB
Lx=8*lB

ky_sw=np.linspace(0,2,200)*1/lB
# ky_sw=np.linspace(0.5,1.5,201)/lB

# Ls=6*lB
# Lx=8*lB
Nx=140
kx=2*pi*np.arange(Nx)/(2*Lx+Ls)-pi*Nx/(2*Lx+Ls)
[k1,k2]=2*pi*np.mgrid[range(Nx),range(Nx)]/(2*Lx+Ls)-pi*Nx/(2*Lx+Ls)


def y2(k):
    with np.errstate(divide='ignore', invalid='ignore'):
        c= np.exp(1j*k*Ls/2)*(-2j+np.exp(1j*k*Lx)*(2j+ 2*k*Lx- 1j* k**2 * Lx**2))/ (k**3) /(2*Lx+Ls)
        c[ ~ np.isfinite( c )] = Lx**3 /(3*(2*Lx+Ls))
    return c

def y1(k):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.exp(1j*k*Ls/2)*(-1+np.exp(1j*k*Lx)*(1- 1j* k * Lx))/ (k**2) /(2*Lx+Ls)
        c[ ~ np.isfinite( c )] = Lx**2 /(2*(2*Lx+Ls))
    return c

def y0(k):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = -1j*np.exp(1j*k*Ls/2)*(-1+np.exp(1j*k*Lx) )/(k*(2*Lx+Ls))
        c[ ~ np.isfinite( c )] = Lx /(2*Lx+Ls)
    return c

def y0s(k):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 2*np.sin(k*Ls/2)/(k*(2*Lx+Ls))
        c[ ~ np.isfinite( c )] = Ls /(2*Lx+Ls)
    return c

sigma0=np.array([[1,0],[0,1]])
sigma1=np.array([[0,1],[1,0]])
sigma2=np.array([[0,-1j],[1j,0]])
sigma3=np.array([[1,0],[0,-1]])
s00=np.kron(sigma0,sigma0)
s30=np.kron(sigma3,sigma0)
s01=np.kron(sigma0,sigma1)
s02=np.kron(sigma0,sigma2)


Hx= hbar*vF*np.kron(s30, np.kron(np.diag(kx) , sigma1))
HeB= 2j*vF*elec*B* np.kron(np.imag(y1(k1-k2)), sigma2)
HeBT= 2j*vF*elec*B* np.kron(np.imag(y1(k2-k1)), sigma2)

Hxm= np.kron( m_sc*y0s(k1-k2)+2*m_n*np.real(y0(k1-k2)) , sigma3)
HxmT= np.kron( m_sc*y0s(k2-k1)+2*m_n*np.real(y0(k2-k1)) , sigma3)
Hlx= np.kron(np.kron(sigma3,sigma2),np.kron( y0s(k1-k2) , lRx*sigma1)) #1
HlxT= np.kron(np.kron(sigma3,sigma2),np.kron( y0s(k2-k1) , lRx*sigma1)) #1
Hly= -np.kron(np.kron(sigma0,sigma1),np.kron( y0s(k1-k2) , lRy*sigma2)) #2
HlyT= -np.kron(np.kron(sigma0,sigma1),np.kron( y0s(k2-k1) , lRy*sigma2)) #2
Hlz= np.kron(np.kron(sigma3,sigma3),np.kron( y0s(k1-k2) , lso*sigma3)) # 333
HlzT= np.kron(np.kron(sigma3,sigma3),np.kron( y0s(k2-k1) , lso*sigma3))
Hl=Hlx+Hly+Hlz
HlT=HlxT+HlyT+HlzT
# Hlx=hbar*lam*np.kron(s02, np.kron( (k1+k2)/2*y0s(k1-k2),sigma0) )
# HlxT=-hbar*lam*np.kron(s02, np.kron((k1+k2)/2*y0s(k2-k1),sigma0) )

Hef= -np.kron( 2*mu_n*np.real(y0(k1-k2)) + mu_sc*y0s(k1-k2), sigma0 )
HefT= -np.kron( 2*mu_n*np.real(y0(k2-k1)) + mu_sc*y0s(k2-k1), sigma0 )

Hd= np.kron(y0s(k1-k2),sigma0)
# basis ud,ud
# dmat1=np.kron(np.array([[0.,1j],[0.,0.]]),sigma1)
# ddmat1=np.kron(dmat1,sigma2) 
dmat2=np.kron(np.array([[0.,1j],[0.,0.]]),D1*sigma1+D2*sigma0)
ddmat2=np.kron(dmat2,sigma2)
Hd_mat=np.kron(ddmat2,Hd)
Hd_mat=Hd_mat+np.transpose(Hd_mat).conj()

s03=np.kron(sigma0,sigma3)
Hz_up= -(gs*y0s(k1-k2)+2*gn*np.real(y0(k1-k2)))
Hz= np.kron(s03, np.kron(Hz_up,sigma0) )
Hz_upT= -(gs*y0s(k2-k1)+2*gn*np.real(y0(k2-k1)))
HzT= np.kron(s03, np.kron(Hz_upT,sigma0) )

t_timer=time.time()

# if lRx>0:
#     fname='TS_Hoppe_Lxs_%d_%d_Nx_%d_nu_%.2f.pdf' % (Lx/lB,Ls/lB,Nx,nu)
# else:
#     fname='TS_Hoppe_no_SO_Lxs_%d_%d_Nx_%d_nu_%.2f.pdf' % (Lx/lB,Ls/lB,Nx,nu)

# print(fname)

# assuming lRx>0
# f1= 'bands_nu_%.3f_Lxs_%d_%d_Nx_%d.npz' % (nu,Lx/lB,Ls/lB,Nx)
f1= 'bands_nu_%.3f_Lxs_%d_%d_l_%.2f_Nx_%d.npz' % (nu,Lx/lB,Ls/lB,lRx/hwc,Nx)
print(f1)


En=np.zeros((16*Nx,len(ky_sw)))
for i_y in range(len(ky_sw)):
    ky=ky_sw[i_y]
    Hy= hbar*vF*ky*np.kron(np.eye(Nx), sigma2)


    H1 = Hx + np.kron(s00, Hy+ HeB + Hxm + Hef)+ Hz + Hl
    H1T = -Hx  + np.kron(s00,Hy+ (HeBT + HxmT + HefT).conj())+ (HzT+HlT).conj()

    H_t=block_diag(H1,-H1T)+Hd_mat
    En[:,i_y] =np.linalg.eigvalsh(H_t)


elapsed = time.time() - t_timer
print("Finished, elapsed time = %.0f " % (elapsed)+ "sec")


out_dir = 'LL_bands_new/' 
fname = out_dir+f1
# np.savez(fname, kps=ky_sw , En=En, evecs=Vn[:,8*Nx-1:8*Nx+1,:])
np.savez(fname, kps=ky_sw , En=En)



# #### plot for notes

# plt.figure(figsize=(5,4))
# plt.plot(ky_sw*lB,En.T/hwc,'b')

# fsize=16
# plt.ylabel(r"$\varepsilon/\varepsilon_0$",fontsize = fsize)
# plt.xlabel(r"$k_y \ell_B$",fontsize = fsize)
# plt.ylim(-0.5,0.5)
# # plt.xlim(-1,4)
# # plt.xticks(np.arange(-4,4.1,2))
# # plt.yticks(np.arange(-2,2.1,1))
# # plt.legend(loc='upper right')
# plt.tight_layout()
# plt.savefig('figs/'+fname)
