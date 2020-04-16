#!/tmp/yes/bin python3

import kwant
import numpy as np
from math import pi, sqrt, tanh
from cmath import exp
import time
import tinyarray
from kwant.digest import uniform    # a (deterministic) pseudorandom number generator
import sys

# armchair edges

t=1.0
t_sc=0.5


tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])

def make_system(Delta=0.2, salt=13, U0=0.0, gn=0.0, gs=0.0, lam=0.0,
                W=200, L=200, Lsc=20, L_lead=20, t_j=0.1, mu=0.6, mu_sc=2,mu_lead=0.6, phi=0):

            
    def qh_slab(pos):
        (x, y) = pos 
        return (0 <= x < W) and (Lsc/2 <= abs(y) < L+Lsc/2)

    def hopping_ab(site_i, site_j, phi):
        xi, yi = site_i.pos
        xj, yj = site_j.pos
#         # modulated hopping in x direction
        H1s=tinyarray.array([[-t*exp(-1j * pi* phi * (xi - xj) * (yi + yj-Lsc*np.sign(yi + yj))),0],\
                            [0,t*exp(1j * pi* phi * (xi - xj) * (yi + yj-Lsc*np.sign(yi + yj)))]])
        H1=np.kron(H1s,np.eye(2))
        return H1

    def onsite(site, mu, gn, U0, salt):
        return  (U0 * (uniform(repr(site), repr(salt)) - 0.5)- mu)* np.kron(tau_z,np.eye(2))-gn* np.kron(tau_z,tau_z)

    
    # Define the graphene lattice
    sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
    lat = kwant.lattice.general([(sqrt(3), 0), (0, 1)],
                                [(0, 0), (1/sqrt(3), 0), (3/sqrt(3)/2, 1/2),(5/sqrt(3)/2, 1/2)],norbs=4)
    subA1, subB1,subA2, subB2 = lat.sublattices
    syst = kwant.Builder()
    syst[lat.shape(qh_slab, (0,int(Lsc/2)))] = onsite
    syst[lat.shape(qh_slab, (0,-int(Lsc/2)))] = onsite
    syst[lat.neighbors()] = hopping_ab

    def hopping_jn(site_i, site_j, t_j):
        return -t_j*np.kron(tau_z,np.eye(2))

    # sc part
    def onsite_sc(site, mu_sc, gs, U0, Delta, salt):
        return  (U0 * (uniform(repr(site), repr(salt)) - 0.5)- mu_sc + 4 * t_sc)* np.kron(tau_z,np.eye(2)) + Delta * np.kron(tau_y,tau_y)- gs* np.kron(tau_z,tau_z)


    def hopping_sc(site1, site2,lam):
        h1=-t_sc* np.kron(tau_z,np.eye(2))
        xi, yi = site1.pos
        xj, yj = site2.pos
        if np.abs(xj-xi)>0:
            h2=1j*lam* np.kron(tau_z,tau_y)
        else:
            h2=1j*lam* np.kron(np.eye(2),tau_x)
        return h1+h2
    
    a0=3/(4*sqrt(3))
    b0=0.5
    primitive_vectors = [(a0, 0), (0, b0)]
    lat_sc = kwant.lattice.Monatomic(primitive_vectors, offset=(a0/6,b0/2),norbs=4)

    Wsq=int(W/a0)
    Lsq=int(Lsc/b0)-1
    syst[(lat_sc(x,y) for x in range(Wsq) for y in range(-int(Lsq/2),int(Lsq/2)))] = onsite_sc
    syst[kwant.builder.HoppingKind((0,1),lat_sc,lat_sc)] = hopping_sc
    syst[kwant.builder.HoppingKind((1,0),lat_sc,lat_sc)] = hopping_sc
    syst[((lat_sc(4*i,int(Lsq/2)-1), subA1(i,int(Lsc/2))) for i in range(int(Wsq/4)))] = hopping_jn
    syst[((lat_sc(4*i+1,int(Lsq/2)-1), subB1(i,int(Lsc/2))) for i in range(int(Wsq/4)))] = hopping_jn
    syst[((lat_sc(4*i+2,int(Lsq/2)-1), subA2(i,int(Lsc/2))) for i in range(int(Wsq/4)))] = hopping_jn
    syst[((lat_sc(4*i+3,int(Lsq/2)-1), subB2(i,int(Lsc/2))) for i in range(int(Wsq/4)))] = hopping_jn
    syst[((lat_sc(4*i,-int(Lsq/2)), subA1(i,-int(Lsc/2))) for i in range(int(Wsq/4)))] = hopping_jn
    syst[((lat_sc(4*i+1,-int(Lsq/2)), subB1(i,-int(Lsc/2))) for i in range(int(Wsq/4)))] = hopping_jn
    syst[((lat_sc(4*i+2,-int(Lsq/2)), subA2(i,-int(Lsc/2)-1)) for i in range(int(Wsq/4)))] = hopping_jn
    syst[((lat_sc(4*i+3,-int(Lsq/2)), subB2(i,-int(Lsc/2)-1)) for i in range(int(Wsq/4)))] = hopping_jn


    def onsite_lead(site, mu, gn):
        return  - mu* np.kron(tau_z,np.eye(2))-gn* np.kron(tau_z,tau_z)

    def onsite_sc_lead(site, mu_sc, gs, Delta):
        return  (- mu_sc + 4 * t_sc)* np.kron(tau_z,np.eye(2))\
                + Delta * np.kron(tau_y,tau_y)- gs* np.kron(tau_z,tau_z)

    
    sym_right = kwant.TranslationalSymmetry(lat.vec((1, 0)))
    right_lead = kwant.Builder(sym_right, particle_hole=np.kron(tau_x,np.eye(2)))
    right_lead[lat.shape(qh_slab, (0,int(Lsc/2)))] = onsite_lead
    right_lead[lat.shape(qh_slab, (0,-int(Lsc/2)))] = onsite_lead
    right_lead[lat.neighbors()] = hopping_ab
    
    right_lead[(lat_sc(x,y) for x in range(4) for y in range(-int(Lsq/2),int(Lsq/2)))] = onsite_sc_lead
    right_lead[kwant.builder.HoppingKind((0,1),lat_sc,lat_sc)] = hopping_sc
    right_lead[kwant.builder.HoppingKind((1,0),lat_sc,lat_sc)] = hopping_sc
    
    right_lead[(lat_sc(0,int(Lsq/2)-1), subA1(0,int(Lsc/2)))] = hopping_jn
    right_lead[(lat_sc(1,int(Lsq/2)-1), subB1(0,int(Lsc/2)))] = hopping_jn
    right_lead[(lat_sc(2,int(Lsq/2)-1), subA2(0,int(Lsc/2)))] = hopping_jn
    right_lead[(lat_sc(3,int(Lsq/2)-1), subB2(0,int(Lsc/2)))] = hopping_jn
    right_lead[(lat_sc(0,-int(Lsq/2)), subA1(0,-int(Lsc/2)))] = hopping_jn
    right_lead[(lat_sc(1,-int(Lsq/2)), subB1(0,-int(Lsc/2)))] = hopping_jn
    right_lead[(lat_sc(2,-int(Lsq/2)), subA2(0,-int(Lsc/2)-1))] = hopping_jn
    right_lead[(lat_sc(3,-int(Lsq/2)), subB2(0,-int(Lsc/2)-1))] = hopping_jn

    
    def lead_slab(pos):
        (x, y) = pos 
        return (0 <= x < W) and (L_lead <= abs(y) < L+Lsc/2)
    
    c1 = np.diag([-2, -1, 1, 2])
    sym_left = kwant.TranslationalSymmetry(lat.vec((1, 0)))
    l1_lead = kwant.Builder(sym_left, conservation_law=c1, particle_hole=np.kron(tau_x,np.eye(2)))
#     l1_lead[lat.shape(qh_slab, (0,int(Lsc/2)))] = onsite_lead
    l1_lead[lat.shape(lead_slab, (0,L_lead))] = onsite_lead
    l2_lead = kwant.Builder(sym_left, conservation_law=c1, particle_hole=np.kron(tau_x,np.eye(2)))
#     l2_lead[lat.shape(qh_slab, (0,-int(Lsc/2)))] = onsite_lead
    l2_lead[lat.shape(lead_slab, (0,-L_lead))] = onsite_lead
    l1_lead[lat.neighbors()] = hopping_ab
    l2_lead[lat.neighbors()] = hopping_ab


    syst.attach_lead(l1_lead.reversed())
    syst.attach_lead(l2_lead.reversed())
    syst.attach_lead(right_lead)
    
    return syst

def compute_conductance(syst, energies, params):
    # Compute conductance
    Pe = np.zeros((4,len(energies)))
    for i_e in range(len(energies)):
        print(i_e, end='\r')
        energy=energies[i_e]
        smatrix = kwant.smatrix(syst, energy=energy,params=params)
        for i in range(4):
            Pe[i,i_e]=smatrix.transmission((1, i), (0, 0))
    return Pe


def compute_conductance(syst, energies, params):
    # Compute conductance
    Pu = np.zeros((4,len(energies)))
    Pd = np.zeros((4,len(energies)))
    for i_e in range(len(energies)):
#         print(i_e)
        energy=energies[i_e]
        smatrix = kwant.smatrix(syst, energy=energy,params=params)
        for i in range(4):
            Pu[i,i_e]=smatrix.transmission((1, i), (0, 0))
            Pd[i,i_e]=smatrix.transmission((1, i), (0, 1))

#         wfs = kwant.wave_function(syst, energy=energy, params=params)
#         if i_e==0:
#             # construct the wf array to be saved
#             print( "Dimension of scattering wf = "+ str(wfs(0).shape) ) 
#             wf_arr=np.zeros((wfs(0).shape[0],wfs(0).shape[1],len(energies)),dtype=np.complex64)
        
    return Pu, Pd
    

def main():
    W=40
    L=100
    Lsc=10
    Llead=50
    syst = make_system(W=W, L=L, Lsc=Lsc, L_lead=Llead)
    # Finalize the system.
    syst = syst.finalized()

    # parameters
    mu=0.32515152 #  for nu=4 # 0.214 # for nu=2  
    lam=0.5
    mu_sc= 0.18+t - 2*lam
    Delta=0.03
    t_j=1.0
    phi=0.0095
    gs=0.
    gn=0.04
    U0=.0
    
    # Compute and plot the conductance
    Nrep=1
    NE=51
    E_list=np.linspace(-1, 1,NE)*Delta/4
    t0=time.time()
        
    out_dir='kw_data_files_dis/'
    f1= 'twosided_U_%.2f_phi_%.4f_mu_%.2f_mus_%.2f_D_%.2f_tj_%.2f_W_%d_L_%d_Ls_%d_Llead_%d' % (U0,phi,mu,mu_sc,Delta,t_j,W,L,Lsc,Llead)
    print(f1)
    f1=out_dir+f1
    for i_r in range(0,Nrep):
        print(i_r)
        salt=t0+i_r
        params=dict(t_j=t_j, gs=gs, gn=gn, lam=lam, Delta=Delta, U0=U0, salt=salt, mu=mu ,mu_sc=mu_sc, phi=phi)

        t_timer=time.time()
        # Compute and plot the conductance
        Pu,Pd = compute_conductance(syst, energies=[E for E in E_list], params=params )
        elapsed = time.time() - t_timer
        print("Finished, elapsed time = %.0f " % (elapsed)+ "sec")

        fname=f1+ '_r_%d.npz' % (i_r)
        np.savez(fname, E_list=E_list, Pu=Pu , Pd=Pd)
    
# Call the main function if the script gets executed (as opposed to imported).
if __name__ == '__main__':
    main()
