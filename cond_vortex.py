#!/tmp/yes/bin python3

import kwant
import numpy as np
from cmath import exp
from math import pi
import time
import tinyarray
from kwant.digest import uniform    # a (deterministic) pseudorandom number generator

t=1.0
t_sc=1.0

tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])

def make_system(a=1, Delta=0.2, salt=13, U0=0.0, vden=0.0,
                W=200, L=200, Wsc=80, Lsc=20, L_top=30, t_j=0.1, mu=0.6, mu_sc=2, phi=0):


    def hopping(site_i, site_j, phi):
        xi, yi = site_i.pos
        xj, yj = site_j.pos
        # modulated hopping in y direction
#         H1=tinyarray.array([[-t*exp(1j * pi* phi * (xi + xj) * (yi - yj)),0],\
#                             [0,t*exp(-1j * pi* phi * (xi + xj) * (yi - yj))]])
#         # modulated hopping in x direction
        H1=tinyarray.array([[-t*exp(-1j * pi* phi * (xi - xj) * (yi + yj)),0],\
                            [0,t*exp(1j * pi* phi * (xi - xj) * (yi + yj))]])
        return H1
    
    def onsite(site, mu, U0, salt):
        return  (U0 * (uniform(repr(site), repr(salt)) - 0.5)- mu + 4 * t)* tau_z
#         return  (4 * t - mu) * tau_z

    def onsite_sc(site, mu_sc, U0, Delta, salt):
        return  (U0 * (uniform(repr(site), repr(salt)) - 0.5)- mu_sc + 4 * t_sc)* tau_z+ Delta * tau_x
#         return  (4 * t_sc - mu_sc) * tau_z + Delta * tau_x

    
    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()
    syst[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    syst[(lat(x, y) for x in range(int(L/2-Lsc/2),int(L/2+Lsc/2) ) for y in range(Wsc))] = onsite_sc
    syst[lat.neighbors()] = hopping
    
    # Modify only those hopings in SC
    def hopping_sc(site1, site2):
        return -t_sc*tau_z
    
    def inside_sc(hop):
        x, y = hop[0].tag
        return ( (int(L/2-Lsc/2)<=x<int(L/2+Lsc/2) )  and (-1<y<Wsc) )

    def hops_inside_sc(syst):
        for hop in kwant.builder.HoppingKind((1, 0), lat, lat)(syst):
            if inside_sc(hop):
                yield hop
        for hop in kwant.builder.HoppingKind((0, 1), lat, lat)(syst):
            if inside_sc(hop):
                yield hop
    syst[hops_inside_sc] = hopping_sc

    # Modify only those hopings on SC-QH bdy
    def hopping_jn(site_i, site_j, t_j, phi):
        return -t_j*tau_z

    def crosses_ybdy(hop):
        x, y = hop[0].tag
        if y==Wsc and (int(L/2-Lsc/2) <=x< int(L/2+Lsc/2)):
            return True
        else:
            return False

    def crosses_xbdy(hop):
        x, y = hop[0].tag
        if x==int(L/2-Lsc/2) and y<Wsc :
            return True
        elif x==int(L/2+Lsc/2) and y<Wsc :
            return True
        else:
            return False

    def hops_across_xbdy(syst):
        for hop in kwant.builder.HoppingKind((1, 0), lat, lat)(syst):
            if crosses_xbdy(hop):
                yield hop
    def hops_across_ybdy(syst):                
        for hop in kwant.builder.HoppingKind((0, 1), lat, lat)(syst):
            if crosses_ybdy(hop):
                yield hop

    syst[hops_across_xbdy] = hopping_jn
    syst[hops_across_ybdy] = hopping_jn
    
    
    
    # vortices
    Nv=int(Wsc*vden)
    np.random.seed(int(time.time()))
    vx=np.random.randint(low=int(L/2-Lsc/2), high=int(L/2+Lsc/2)-1, size=Nv)+0.5
    ys=np.arange(0,Wsc-1)
    np.random.shuffle(ys)
    vy=np.sort(ys[:Nv])+0.5
    
    # vortex inside SC
    def crosses_sc_vortex(hop):
        x2, y2 = hop[0].tag
        x1, y1 = hop[1].tag
        y0=(y2+y1)/2.0
        i0=np.argwhere(vy==y0)
        if len(i0)!=0 and x1>=int(L/2-Lsc/2):
            if x1< vx[i0[0][0]]:
                return True
            else:
                return False                
        else:
            return False
        
    def hops_sc_vortex(syst):
        for hop in kwant.builder.HoppingKind((0, 1), lat, lat)(syst):
            if crosses_sc_vortex(hop):
                yield hop
                
    def neg_hopping_sc(site1, site2):
        return t_sc*tau_z
    syst[hops_sc_vortex] = neg_hopping_sc
   
    # vortex outside SC
    def crosses_qh_vortex(hop):
        x2, y2 = hop[0].tag
        x1, y1 = hop[1].tag
        y0=(y2+y1)/2.0
        i0=np.argwhere(vy==y0)
        if len(i0)!=0 and x1<int(L/2-Lsc/2):
                return True
        else:
            return False
        
    def hops_qh_vortex(syst):
        for hop in kwant.builder.HoppingKind((0, 1), lat, lat)(syst):
            if crosses_qh_vortex(hop):
                yield hop
                
    def neg_hopping(site1, site2):
        return t*tau_z
    syst[hops_qh_vortex] = neg_hopping


    def onsite_lead(site, mu):
        return  (- mu + 4 * t)* tau_z

    def hopping_lead(site1, site2):
        return -t*tau_z

    sym_left = kwant.TranslationalSymmetry((-1, 0))
    left_lead = kwant.Builder(sym_left, conservation_law=-tau_z, particle_hole=tau_y)
    left_lead[(lat(0, y) for y in range(0,W))] = onsite_lead
    left_lead[lat.neighbors()] = hopping
    syst.attach_lead(left_lead)
    syst.attach_lead(left_lead.reversed())

    sym_top = kwant.TranslationalSymmetry((0, 1))
    top_lead = kwant.Builder(sym_top, conservation_law=-tau_z, particle_hole=tau_y)
    top_lead[(lat(x, 0) for x in range(int(L/2-L/4),int(L/2+L/4))) ] = onsite_lead
    top_lead[lat.neighbors()] = hopping_lead
    syst.attach_lead(top_lead)

    sym_bottom = kwant.TranslationalSymmetry((0, -1))
    bottom_lead = kwant.Builder(sym_bottom, particle_hole=tau_y)
    bottom_lead[(lat(x, 0) for x in range(int(L/2-Lsc/2),int(L/2+Lsc/2))) ] = onsite_sc
    bottom_lead[lat.neighbors()] = hopping_sc
    syst.attach_lead(bottom_lead)

    return syst


def compute_conductance(syst, energies, params):
    # Compute conductance
    Pe = np.zeros(len(energies))
    Ph = np.zeros(len(energies))
    for i_e in range(len(energies)):
#         print(i_e)
        energy=energies[i_e]
        smatrix = kwant.smatrix(syst, energy=energy,params=params)
        Pe[i_e]=smatrix.transmission((1, 0), (0, 0))                     
        Ph[i_e]=smatrix.transmission((1, 1), (0, 0)) 
        wfs = kwant.wave_function(syst, energy=energy, params=params)
        if i_e==0:
            # construct the wf array to be saved
            print( "Dimension of scattering wf = "+ str(wfs(0).shape) ) 
            wf_arr=np.zeros((wfs(0).shape[0],wfs(0).shape[1],len(energies)),dtype=np.complex64)
        
        wf_arr[:,:,i_e]= wfs(0)
    return Pe, Ph, wf_arr
    
def main():
    Wsc=120
    W=Wsc+40
    Lsc=10
    L=Lsc+40

    # parameters
    mu=0.6
    mu_sc=1.5
    phi=0.05
    U0=0.0 # disorder strength
    salt=13
    Delta=0.06
    t_j=0.6
    
    # Compute and plot the conductance
    NE=50
    E_list=np.linspace(0, 1.2,NE)*Delta
    
    vden=0.5
    Nrep=100
    for i_r in range(48,Nrep):
        print(i_r)
        t_timer=time.time()
        syst = make_system(vden=vden, W=W, L=L, Wsc=Wsc, Lsc=Lsc)
        syst = syst.finalized()

        Pe, Ph, wf_arr= compute_conductance(syst, energies=[E for E in E_list], params=dict(t_j=t_j,Delta=Delta,mu=mu,mu_sc=mu_sc,phi=phi,U0=U0,salt=salt) )
        elapsed = time.time() - t_timer
        print("Finished, elapsed time = %.0f " % (elapsed)+ "sec")
    
        out_dir='kw_data_files_vortex/'
        fname=out_dir+ 'vd_%.2f_phi_%.2f_mu_%.2f_mus_%.2f_D_%.2f_tj_%.2f_W_%d_L_%d_Ws_%d_Ls_%d_r_%d.npz' % (vden,phi,mu,mu_sc,Delta,t_j,W,L,Wsc,Lsc,i_r)
        np.savez(fname, E_list=E_list, Pe=Pe, Ph=Ph, wf_arr=wf_arr)
    
# Call the main function if the script gets executed (as opposed to imported).
if __name__ == '__main__':
    main()

