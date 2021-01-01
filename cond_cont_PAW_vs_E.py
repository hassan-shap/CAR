#!/tmp/yes/bin python3

import numpy as np
import scipy as sp
from math import pi, sqrt
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.optimize import minimize
import time

def y2(k,Lx,Ls):
    with np.errstate(divide='ignore', invalid='ignore'):
        c= np.exp(1j*k*Ls/2)*(-2j+np.exp(1j*k*Lx)*(2j+ 2*k*Lx- 1j* k**2 * Lx**2))/ (k**3) /(2*Lx+Ls)
        c[ ~ np.isfinite( c )] = Lx**3 /(3*(2*Lx+Ls))
    return c

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

# fundamental constants
hbar=1.05e-34
elec=1.6e-19
meV=elec*1e-3

vF=1e6
B=6 # Tesla
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

def sample_modes(Nx,Lx,Ls,lead_sample,E,params):

    nu=params['nu']
    mu_n=nu*hwc 
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
    # Hlx=hbar*lam*np.kron(s02, np.kron( (k1+k2)/2*y0s(k1-k2),sigma0) )
    # HlxT=-hbar*lam*np.kron(s02, np.kron((k1+k2)/2*y0s(k2-k1),sigma0) )

    Hef= -np.kron( 2*mu_n*np.real(y0(k1-k2,Lx,Ls)) + mu_sc*y0s(k1-k2,Lx,Ls), sigma0 )
    HefT= -np.kron( 2*mu_n*np.real(y0(k2-k1,Lx,Ls)) + mu_sc*y0s(k2-k1,Lx,Ls), sigma0 )

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


#     Hy= hbar*vF*ky*np.kron(np.eye(Nx), sigma2)
    H1 = Hx + np.kron(s00, HeB + Hxm + Hef)+ Hz + Hl
    H1T = -Hx  + np.kron(s00, (HeBT + HxmT + HefT).conj())+ (HzT+HlT).conj()

    H_t=(block_diag(H1,-H1T)+Hd_mat)/(hbar*vF)*lB
  
    kmat=np.kron(sigma3, np.kron(np.eye(4*Nx),sigma2))

    ky_max = 4
    ky_min = 1e-5

    if lead_sample=='lead': 
#         En=(E+1e-3*hwc)/(hbar*vF)*lB
        En=E/(hbar*vF)*lB
        evs, evecs = np.linalg.eig( np.dot(kmat,En*np.eye(16*Nx)- H_t ) )
        i_evan= np.where( np.abs(np.imag(evs)) > ky_min )[0]
        ie_T= np.where( np.imag(evs[i_evan]) < 0)[0]
        i_e= i_evan[ie_T]

        i_prop1= np.where( np.abs(np.imag(evs)) < ky_min )[0]
        i_prop2= np.where( np.abs(np.real(evs[i_prop1])) > ky_min )[0]
        i_prop3= np.where( np.abs(np.real(evs[i_prop1[i_prop2]])) < ky_max )[0]            
        Jmat = np.reshape(np.diag(np.dot(np.transpose(evecs).conj(),np.dot(kmat,evecs))),[16*Nx,])
        evecs= np.dot(evecs,np.diag(1/Jmat**0.5))

        ip_T= np.where( np.real(Jmat[i_prop1[i_prop2[i_prop3]]]) > 0)[0]
        i_p= i_prop1[i_prop2[i_prop3[ip_T]]]
        ptcl= np.sum(np.abs(evecs[:8*Nx,i_p])**2,axis=0)
        hole= np.sum(np.abs(evecs[8*Nx:,i_p])**2,axis=0)
        hz= ptcl-hole
        i_ppt= np.where(ptcl-hole > 0)[0]
#             i_phl= np.where(ptcl-hole < 0)[0]
        Tevecs = evecs[:,i_p[i_ppt]]

        in_T= np.where( np.real(Jmat[i_prop1[i_prop2]]) < 0)[0]
        i_n= i_prop1[i_prop2[in_T]]
        ptcl= np.sum(np.abs(evecs[:8*Nx,i_n])**2,axis=0)
        hole= np.sum(np.abs(evecs[8*Nx:,i_n])**2,axis=0)
        hz= ptcl-hole
        i_npt= np.where(ptcl-hole > 0)[0]
        i_nhl= np.where(ptcl-hole < 0)[0]
#             num_refl=len(i_n)/2
#             assert num_refl-int(num_refl)~=0, "Number of reflecting modes is %d" % (num_refl)
#             num_refl= int(num_refl)
        num_refl=[len(i_npt),len(i_n)]
        i_neg=np.concatenate((i_n[i_npt],i_n[i_nhl],i_e))
        Revecs = evecs[:,i_neg]

        return num_refl,Revecs,Tevecs

    elif lead_sample=='sample':
        En=E/(hbar*vF)*lB
        evs, evecs = np.linalg.eig( np.dot(kmat,En*np.eye(16*Nx)- H_t ) )
        i_evan= np.where( np.abs(np.imag(evs)) > ky_min )[0]
        ie_T= np.where( np.imag(evs[i_evan]) > 0)[0]
        i_e= i_evan[ie_T]

        i_prop1= np.where( np.abs(np.imag(evs)) < ky_min )[0]
        i_prop2= np.where( np.abs(np.real(evs[i_prop1])) > ky_min )[0]
        Jmat = np.reshape(np.diag(np.dot(np.transpose(evecs).conj(),np.dot(kmat,evecs))),[16*Nx,])
        evecs= np.dot(evecs,np.diag(1/Jmat**0.5))

        ip_T= np.where( np.real(Jmat[i_prop1[i_prop2]]) > 0)[0]
        i_p= i_prop1[i_prop2[ip_T]]
        i_pos=np.concatenate((i_p,i_e))
        Tevecs = evecs[:,i_pos]
        return Tevecs
    else:
        En=E/(hbar*vF)*lB
        evs, evecs = np.linalg.eig( np.dot(kmat,En*np.eye(16*Nx)- H_t ) )
        i_prop1= np.where( np.abs(np.imag(evs)) < ky_min )[0]
        i_prop2= np.where( np.abs(np.real(evs[i_prop1])) > ky_min )[0]
        i_prop3= np.where( np.abs(np.real(evs[i_prop1[i_prop2]])) < ky_max )[0]            
        Jmat = np.reshape(np.diag(np.dot(np.transpose(evecs).conj(),np.dot(kmat,evecs))),[16*Nx,])
        ip_T= np.where( np.real(Jmat[i_prop1[i_prop2[i_prop3]]]) > 0)[0]
        i_p= i_prop1[i_prop2[i_prop3[ip_T]]]
        in_T= np.where( np.real(Jmat[i_prop1[i_prop2[i_prop3]]]) < 0)[0]
        i_n= i_prop1[i_prop2[i_prop3[in_T]]]
        return evs[i_p],evs[i_n]

def plane_waves_T(Nx,Nd,Ld,Lx,Ls):
    kx=np.reshape(2*pi*np.arange(Nx)/(2*Lx+Ls)-pi*Nx/(2*Lx+Ls),[Nx,1])
    kxp=np.reshape(2*pi*np.arange(Nd)/Ld-pi*Nd/Ld,[Nd,1])
    evec=np.zeros((16*Nx,4*Nd), dtype=np.complex64)
    v_p=np.kron(np.array([[1],[0]]),np.eye(4))
    v_pos=np.array([[1],[1j]])/sqrt(2)
    for n in range(0,Nd):
        with np.errstate(divide='ignore', invalid='ignore'):
            c= (np.exp(1j*(kxp[n]-kx)*Ls/2)-np.exp(1j*(kxp[n]-kx)*(Ld+Ls/2)) )/(-1j*(kxp[n]-kx))
            c[ ~ np.isfinite( c )] = Ld          
        evec[:,4*n:4*n+4]= np.kron(v_p,np.kron(c.conj(),v_pos))/np.linalg.norm(c)
    return evec

def y0out(k,Ld,Lx,Ls):
    with np.errstate(divide='ignore', invalid='ignore'):
        c= (np.exp(1j*k*Ls/2)-np.exp(1j*k*(Ld+Ls/2)) )/(-1j*k)/(2*Lx+Ls)
        c[ ~ np.isfinite( c )] = Ld/(2*Lx+Ls)
    return c



def main():

    Ls=6*lB
    Lx=8*lB
    Nx=200

    D1=0.3*hwc #10*meV
    D2=0.2*hwc #10*meV
    m_n=0.06*hwc

    lRx= 0.2*hwc
    lRy= 0.*hwc
    lso= 0.*hwc
    gs=0.0*hwc
    gn=0.1*hwc

    nu=0.3
    m_sc=3*hwc 
    mu_sc=8*hwc
    params=dict(nu=nu, m_n=m_n, mu_sc=mu_sc, m_sc=m_sc, D1=D1, D2=D2,\
                lRx=lRx, lRy=lRy, lso=lso, gs=gs, gn=gn)

    Esw=np.linspace(-1,1,200)*0.04*hwc

    Ree=np.zeros(len(Esw))
    Reh=np.zeros(len(Esw))

    v_p=np.array([[1],[0]])
    v_h=np.array([[0],[1]])
    v_pos=np.array([[1],[1j]])/sqrt(2)
    v_neg=np.array([[1],[-1j]])/sqrt(2)
    Revecs_l_p=np.kron(v_p,np.kron(np.eye(4*Nx),v_neg))
    Revecs_l_h=np.kron(v_h,np.kron(np.eye(4*Nx),v_pos))
    Revecs_l= np.concatenate((Revecs_l_p,Revecs_l_h),axis=1)
    [k1,k2]=2*pi*np.mgrid[range(Nx),range(Nx)]/(2*Lx+Ls)-pi*Nx/(2*Lx+Ls)

    Nd= int(Nx/4)
    Ld=Lx
#     Tevecs_l= plane_waves_T(Nx,Nd,Ld,Lx,Ls)
    Xmat=np.kron(s00,y0out(k2-k1,Ld,Lx,Ls))
    Tevecs_l=np.kron(v_p,np.kron(np.eye(4*Nx),v_pos))

    out_dir='cont_data_files/'
    f1='cond_paw_vs_E_Nx_%d_Lxs_%d_%d_nu_%.2f_mn_%.2f_ms_%.2f_mus_%.2f_D12_%.2f_%.2f_lxys_%.2f_%.2f_%.2f_gsn_%.2f_%.2f.npz' %\
          (Nx,Lx/lB,Ls/lB,nu,m_n/hwc,m_sc/hwc,mu_sc/hwc,\
           D1/hwc,D2/hwc,lRx/hwc,lRy/hwc,lso/hwc,gs/hwc,gn/hwc)

    print(f1)
    fname=out_dir+f1

    t_timer=time.time()

    for i_E in range(len(Esw)):
        print(i_E,end='\r')

        Tevecs= sample_modes(Nx,Lx,Ls,'sample',Esw[i_E],params)

        Psi_t=np.concatenate((-Revecs_l,Tevecs),axis=1)
        x=sp.linalg.solve(Psi_t,Tevecs_l)

#         Ree[i_E]=np.real(np.trace(np.dot(np.matrix(x[:4*Nx,:]).H,x[:4*Nx,:])))
#         Reh[i_E]=np.real(np.trace(np.dot(np.matrix(x[4*Nx:8*Nx,:]).H,x[4*Nx:8*Nx,:])))
        Ree[i_E]=np.real( np.trace( np.dot( x[:4*Nx,:], np.dot(np.dot(Xmat.conj(),np.matrix(x[:4*Nx,:]).H ) , Xmat ) ) ))
        Reh[i_E]=np.real( np.trace( np.dot( x[4*Nx:8*Nx,:], np.dot(np.dot(Xmat.conj(),np.matrix(x[4*Nx:8*Nx,:]).H ) , Xmat ) ) ))


    np.savez(fname, E_list=Esw, Ree=Ree , Reh=Reh)

    elapsed = time.time() - t_timer
    print("Finished, elapsed time = %.0f " % (elapsed)+ "sec")

# Call the main function if the script gets executed (as opposed to imported).
if __name__ == '__main__':
    main()


