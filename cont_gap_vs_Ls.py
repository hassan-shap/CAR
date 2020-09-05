#!/tmp/yes/bin python3

import numpy as np
import scipy as sp
from math import pi, sqrt
from scipy.linalg import block_diag
from scipy.optimize import minimize
import time

def y1(k,Lx,Ls):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.exp(1j*k*Ls/2)*(-1+np.exp(1j*k*Lx)*(1- 1j* k * Lx))/ (k**2) /(2*Lx+Ls)
        c[ ~ np.isfinite( c )] = Lx**2 /(2*(2*Lx+Ls))
    return c

def y0(k,Lx,Ls):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = -1j*np.exp(1j*k*Ls/2)*(-1+np.exp(1j*k*Lx) )/(k*(2*Lx+Ls))
        c[ ~ np.isfinite( c )] = Lx /(2*Lx+Ls)
    return c

def y0s(k,Lx,Ls):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 2*np.sin(k*Ls/2)/(k*(2*Lx+Ls))
        c[ ~ np.isfinite( c )] = Ls /(2*Lx+Ls)
    return c

hbar=1.05e-34
elec=1.6e-19
meV=elec*1e-3

vF=1e6
B=10 # Tesla
hwc=vF*sqrt(2*hbar*elec*B)
lB=np.sqrt(hbar/(elec*B))


sigma0=np.array([[1,0],[0,1]])
sigma1=np.array([[0,1],[1,0]])
sigma2=np.array([[0,-1j],[1j,0]])
sigma3=np.array([[1,0],[0,-1]])
s00=np.kron(sigma0,sigma0)
s30=np.kron(sigma3,sigma0)
s01=np.kron(sigma0,sigma1)
s02=np.kron(sigma0,sigma2)
s03=np.kron(sigma0,sigma3)


def system_E_vs_k(H_0,ky_sw):

    Nx= int(H_0.shape[0]/16)

    En=np.zeros((16*Nx,len(ky_sw)))
    for i_y in range(len(ky_sw)):
        ky=ky_sw[i_y]
        Hy= hbar*vF*ky*np.kron(s00,np.kron(np.eye(Nx), sigma2))

        H_t=block_diag(Hy,-Hy)+H_0*hbar*vF/lB
        En[:,i_y] =np.linalg.eigvalsh(H_t)

    return En


def Hamiltonian_builder(Nx,Lx,Ls,params):

    m_n=params['m_n']
    mu_sc=params['mu_sc']
    m_sc=params['m_sc']
    
    D1=params['D1']
    D2=params['D2']

    lRx= params['lRx']
    lRy= params['lRy']
    lso= params['lso']
    gs= params['gs']
    gn= params['gn']

    kx=2*pi*np.arange(Nx)/(2*Lx+Ls)-pi*Nx/(2*Lx+Ls)
    [k1,k2]=2*pi*np.mgrid[range(Nx),range(Nx)]/(2*Lx+Ls)-pi*Nx/(2*Lx+Ls)


    Hx= hbar*vF*np.kron(s30, np.kron(np.diag(kx) , sigma1))
    HeB= 2j*vF*elec*B* np.kron(np.imag(y1(k1-k2,Lx,Ls)), sigma2)
    HeBT= 2j*vF*elec*B* np.kron(np.imag(y1(k2-k1,Lx,Ls)), sigma2)

    Hxm= np.kron( m_sc*y0s(k1-k2,Lx,Ls)+2*m_n*np.real(y0(k1-k2,Lx,Ls)) , sigma3)
    HxmT= np.kron( m_sc*y0s(k2-k1,Lx,Ls)+2*m_n*np.real(y0(k2-k1,Lx,Ls)) , sigma3)
    Hlx= np.kron(np.kron(sigma3,sigma2),np.kron( y0s(k1-k2,Lx,Ls) , lRx*sigma1)) #1
    HlxT= np.kron(np.kron(sigma3,sigma2),np.kron( y0s(k2-k1,Lx,Ls) , lRx*sigma1)) #1
    Hly= -np.kron(np.kron(sigma0,sigma1),np.kron( y0s(k1-k2,Lx,Ls) , lRy*sigma2)) #2
    HlyT= -np.kron(np.kron(sigma0,sigma1),np.kron( y0s(k2-k1,Lx,Ls) , lRy*sigma2)) #2
    Hlz= np.kron(np.kron(sigma3,sigma3),np.kron( y0s(k1-k2,Lx,Ls) , lso*sigma3)) # 333
    HlzT= np.kron(np.kron(sigma3,sigma3),np.kron( y0s(k2-k1,Lx,Ls) , lso*sigma3))
    Hl=Hlx+Hly+Hlz
    HlT=HlxT+HlyT+HlzT

    Hef= -np.kron( mu_sc*y0s(k1-k2,Lx,Ls), sigma0 )
    HefT= -np.kron( mu_sc*y0s(k2-k1,Lx,Ls), sigma0 )

    Hd= np.kron(y0s(k1-k2,Lx,Ls),sigma0)
    # basis ud,ud
    # dmat1=np.kron(np.array([[0.,1j],[0.,0.]]),sigma1)
    # ddmat1=np.kron(dmat1,sigma2) 
    dmat2=np.kron(np.array([[0.,1j],[0.,0.]]),D1*sigma1+D2*sigma0)
    ddmat2=np.kron(dmat2,sigma2)
    Hd_mat=np.kron(ddmat2,Hd)
    Hd_mat=Hd_mat+np.transpose(Hd_mat).conj()

    s03=np.kron(sigma0,sigma3)
    Hz_up= -(gs*y0s(k1-k2,Lx,Ls)+2*gn*np.real(y0(k1-k2,Lx,Ls)))
    Hz= np.kron(s03, np.kron(Hz_up,sigma0) )
    Hz_upT= -(gs*y0s(k2-k1,Lx,Ls)+2*gn*np.real(y0(k2-k1,Lx,Ls)))
    HzT= np.kron(s03, np.kron(Hz_upT,sigma0) )

    H1 = Hx + np.kron(s00, HeB + Hxm + Hef)+ Hz + Hl
    H1T = -Hx  + np.kron(s00, (HeBT + HxmT + HefT).conj())+ (HzT+HlT).conj()

    H_t=(block_diag(H1,-H1T)+Hd_mat)/(hbar*vF)*lB
    
    Hef= -np.kron( 2*hwc*np.real(y0(k1-k2,Lx,Ls)) , sigma0 )
    HefT= -np.kron( 2*hwc*np.real(y0(k2-k1,Lx,Ls)), sigma0 )

    H1 = np.kron(s00,Hef)
    H1T = np.kron(s00, HefT.conj())
    H_mu = block_diag(H1,-H1T)/(hbar*vF)*lB

    return H_t, H_mu
    
def sample_modes(H_t,lead_sample,E):
    
    Nx= int(H_t.shape[0]/16)
    kmat=np.kron(sigma3, np.kron(np.eye(4*Nx),sigma2))

    ky_max = 4
    ky_min = 1e-5

    if lead_sample=='sample':
        En=E/(hbar*vF)*lB
        evs, evecs = np.linalg.eig( np.dot(kmat,En*np.eye(16*Nx)- H_t ) )
        i_evan= np.where( np.abs(np.imag(evs)) > ky_min )[0]
        ie_T= np.where( np.imag(evs[i_evan]) > 0)[0]
        i_e= i_evan[ie_T]

        i_prop1= np.where( np.abs(np.imag(evs)) < ky_min )[0]
        i_prop2= np.where( np.abs(np.real(evs[i_prop1])) > ky_min )[0]
        Jmat = np.reshape(np.diag(np.dot(np.transpose(evecs).conj(),np.dot(kmat,evecs))),[16*Nx,])

        ip_T= np.where( np.real(Jmat[i_prop1[i_prop2]]) > 0)[0]
        i_p= i_prop1[i_prop2[ip_T]]
        i_pos=np.concatenate((i_p,i_e))
        evecs= np.dot(evecs,np.diag(1/Jmat**0.5))
        Tevecs = evecs[:,i_pos]
        return Tevecs, evs
    else:
        En=E/(hbar*vF)*lB
        evs, evecs = np.linalg.eig( np.dot(kmat,En*np.eye(16*Nx)- H_t ) )
        Jmat = np.reshape(np.diag(np.dot(np.transpose(evecs).conj(),np.dot(kmat,evecs))),[16*Nx,])

        i_prop2= np.where( np.real(evs) > ky_min )[0]
        i_prop3= np.where( np.real(evs[i_prop2]) < ky_max )[0]            
        i_tot= i_prop2[i_prop3]
        evs_p=evs[i_tot]
        i_s= np.argsort(np.abs(evs_p))

        return evs_p[i_s[::2]] #, Jmat[i_tot]


def main():

    # plot k points at minima vs nu
    Lx=8*lB
    Nx=100

#     D1=0.5*hwc #10*meV
#     D2=0.6*hwc #10*meV
#     m_n=0.1*hwc

#     lRx= 1.*hwc
#     lRy= 0.*hwc
#     lso= 0.*hwc
#     gs=0.0*hwc
#     gn=0.3*hwc

#     nu=0.55
    D1=0.2*hwc 
    D2=0.2*hwc 
    m_n=0.001*hwc

    lRx= 1*hwc
    lRy= 0.*hwc
    lso= 0.*hwc
    gs=0.0*hwc
    gn=0.003*hwc
    nu=0.003
    
    m_sc=3*hwc 
    mu_sc=8*hwc
    params=dict(m_n=m_n, mu_sc=mu_sc, m_sc=m_sc, D1=D1, D2=D2,\
                lRx=lRx, lRy=lRy, lso=lso, gs=gs, gn=gn)

    out_dir='cont_data_files/'
#     f1='Eg_fine2_paw_vs_Ls_Nx_%d_Lxs_%d_nu_%.2f_mn_%.2f_ms_%.2f_mus_%.2f_D12_%.2f_%.2f_lxys_%.2f_%.2f_%.2f_gsn_%.2f_%.2f.npz' %\
#           (Nx,Lx/lB,nu,m_n/hwc,m_sc/hwc,mu_sc/hwc,\
#            D1/hwc,D2/hwc,lRx/hwc,lRy/hwc,lso/hwc,gs/hwc,gn/hwc)
    f1='Eg_fine2_paw_vs_Ls_Nx_%d_Lxs_%d_nu_%.3f_mn_%.3f_ms_%.2f_mus_%.2f_D12_%.2f_%.2f_lxys_%.2f_%.2f_%.2f_gsn_%.2f_%.3f.npz' %\
          (Nx,Lx/lB,nu,m_n/hwc,m_sc/hwc,mu_sc/hwc,\
           D1/hwc,D2/hwc,lRx/hwc,lRy/hwc,lso/hwc,gs/hwc,gn/hwc)



    print(f1)
    fname=out_dir+f1
    
    t_timer=time.time()

    E_sample= 0*hwc
    Ls_sw=np.linspace(6,10,50)

    kps=np.zeros((4,len(Ls_sw)))
    Egs=np.zeros((4,len(Ls_sw)))
    for i_n in range(len(Ls_sw)):
        print(i_n,end='\r')
        Ls= Ls_sw[i_n]*lB
        
        H_0, H_mu = Hamiltonian_builder(Nx,Lx,Ls,params)
        H_t= H_0+nu*H_mu

        ks = sample_modes(H_t,'evals',E_sample)
        kps[:,i_n] = np.real(ks[0:4])
        Egs[:,i_n]= system_E_vs_k(H_t,kps[:,i_n]/lB)[8*Nx,:]

        
    np.savez(fname, Ls_list=Ls_sw, kps=kps , Egs=Egs)

    elapsed = time.time() - t_timer
    print("Finished, elapsed time = %.0f " % (elapsed)+ "sec")

# Call the main function if the script gets executed (as opposed to imported).
if __name__ == '__main__':
    main()