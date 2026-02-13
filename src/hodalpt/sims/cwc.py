'''

module for the CosmicSignals bias model 


author(s):
    * Francesco Sinigaglia 
    * ChangHoon Hahn: minor modifications 

'''
import numpy as np
import numba as nb
from numba import njit, prange, int64
from numba.typed import List


@njit(parallel=True, fastmath=True, cache=True)
def real_to_redshift_space_local_box(delta, tweb, posx, posy, posz, vx, vy, vz,
                                     ngrid, lbox, bv, bb, betarsd, gamma, redshift, omega_m):
    # Real to redshift space (local)

    H0 = 100.
    omega_l = 1. - omega_m 

    lcell = lbox/ ngrid

    posznew = np.zeros(len(posz))

    ascale = 1./(1.+redshift)

    # Parallelize the outer loop                                                                                                                                                                            
    for ii in prange(len(posx)):

        # Initialize positions at the centre of the cell                                                                                                                                                    
        xtmp = posx[ii]
        ytmp = posy[ii]
        ztmp = posz[ii]

        indx = int(xtmp/lcell)
        indy = int(ytmp/lcell)
        indz = int(ztmp/lcell)

        ind3d = indz+ngrid*(indy+ngrid*indx)

        # Compute redshift           
        HH = H0 * np.sqrt(omega_m*(1.+redshift)**3 + omega_l)

        sigma = bb*(1. + delta[indx,indy,indz])**betarsd

        vzrand = np.random.normal(0,sigma)
        vzrand = np.sign(vzrand) * abs(vzrand) ** gamma

        vztmp = trilininterp(xtmp, ytmp, ztmp, vz, lbox, ngrid)

        vztmp += vzrand

        # Go from cartesian to sky coordinates
        ztmp = ztmp + bv * vztmp #/ (ascale * HH) # BOX velocities are already in Mpc/h

        # Impose boundary conditions
        if ztmp<0:
            ztmp += lbox
        elif ztmp>lbox:
            ztmp -= lbox

        posznew[ii] = ztmp

    return posx, posy, posznew


@njit(fastmath=True, cache=True)
def negative_binomial(n, p):
    if n>0:
        if p > 0. and p < 1.:
            gfunc = np.random.gamma(n, (1. - p) / p)
            Y = np.random.poisson(gfunc)

    else:
        Y = 0

    return Y


@njit(parallel=False, cache=True, fastmath=True)
def trilininterp(xx, yy, zz, arrin, lbox, ngrid):
    lcell = lbox/ngrid

    indxc = int(xx/lcell)
    indyc = int(yy/lcell)
    indzc = int(zz/lcell)

    wxc = xx/lcell - indxc
    wyc = yy/lcell - indyc
    wzc = zz/lcell - indzc

    if wxc <=0.5:
        indxl = indxc - 1
        if indxl<0:
            indxl += ngrid
        wxc += 0.5
        wxl = 1 - wxc
    elif wxc >0.5:
        indxl = indxc + 1
        if indxl>=ngrid:
            indxl -= ngrid
        wxl = wxc - 0.5
        wxc = 1 - wxl

    if wyc <=0.5:
        indyl = indyc - 1
        if indyl<0:
            indyl += ngrid
        wyc += 0.5
        wyl = 1 - wyc
    elif wyc >0.5:
        indyl = indyc + 1
        if indyl>=ngrid:
            indyl -= ngrid
        wyl = wyc - 0.5
        wyc = 1 - wyl

    if wzc <=0.5:
        indzl = indzc - 1
        if indzl<0:
            indzl += ngrid
        wzc += 0.5
        wzl = 1 - wzc
    elif wzc >0.5:
        indzl = indzc + 1
        if indzl>=0:
            indzl -= ngrid
        wzl = wzc - 0.5
        wzc = 1 - wzl

    wtot = wxc*wyc*wzc + wxl*wyc*wzc + wxc*wyl*wzc + wxc*wyc*wzl + wxl*wyl*wzc + wxl*wyc*wzl + wxc*wyl*wzl + wxl*wyl*wzl

    out = 0.

    out += arrin[indxc,indyc,indzc] * wxc*wyc*wzc
    out += arrin[indxl,indyc,indzc] * wxl*wyc*wzc
    out += arrin[indxc,indyl,indzc] * wxc*wyl*wzc
    out += arrin[indxc,indyc,indzl] * wxc*wyc*wzl
    out += arrin[indxl,indyl,indzc] * wxl*wyl*wzc
    out += arrin[indxc,indyl,indzl] * wxc*wyl*wzl
    out += arrin[indxl,indyc,indzl] * wxl*wyc*wzl
    out += arrin[indxl,indyl,indzl] * wxl*wyl*wzl

    return out


@njit(parallel=True, cache=True, fastmath=True)
def biasmodel_local_box(ngrid, lbox, delta, nmean, alpha, beta, dth, rhoeps, eps, rhoepsprime, epsprime):
    # Compute bias model
    lcell = lbox/ ngrid

    # Allocate tracer field (may be replaced with delta if too memory consuming)
    ncounts = np.zeros((ngrid,ngrid,ngrid))

    # FIRST LOOP: deterministic bias
    # Parallelize the outer loop
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                # Sample number counts
                if delta[ii,jj,kk]<dth:
                    ncounts[ii,jj,kk] = 0.
                else:
                    ncounts[ii,jj,kk] = (1.+delta[ii,jj,kk])**alpha * np.exp(-((1 + delta[ii,jj,kk])/rhoeps)**eps) * np.exp(-((1 + delta[ii,jj,kk])/rhoepsprime)**epsprime)


    # SECOND LOOP: stochastic bias - we need to compute the right normalization beforehand
    denstot = np.sum(ncounts) / lbox**3
    
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                ncounts[ii,jj,kk] = nmean / denstot *  ncounts[ii,jj,kk]
                pnegbin = 1 - ncounts[ii,jj,kk]/(ncounts[ii,jj,kk] + beta)

                ncounts[ii,jj,kk] = negative_binomial(beta, pnegbin)
    
    return ncounts


@njit(parallel=True, cache=True, fastmath=True)
def biasmodel_nonlocal_box(ngrid, lbox, delta, tweb, dweb, nmean_arr, alpha_arr, beta_arr, dth_arr, rhoeps_arr, eps_arr):
    ''' compute bias model with non-local terms. more documentation to come. 


    see https://arxiv.org/pdf/2403.19337 for details 

    https://github.com/francescosinigaglia/CosmicSignal4SimBIG/blob/622cbebf48d863c77f11c556734212dadbf7108a/boxes/make_galaxy_catalog.py#L822
    minus redshift dependence
    '''
    lcell = lbox/ ngrid

    # Allocate tracer field (may be replaced with delta if too memory consuming)
    ncounts = np.zeros((ngrid,ngrid,ngrid))
    
    denstot_arr = np.zeros((4,4))

    # FIRST LOOP: deterministic bias
    # Parallelize the outer loop
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                indtweb = int(tweb[ii,jj,kk])-1
                inddweb = int(dweb[ii,jj,kk])-1

                nmean   = nmean_arr[indtweb, inddweb]
                alpha   = alpha_arr[indtweb, inddweb] 
                beta    = beta_arr[indtweb, inddweb]
                dth     = dth_arr[indtweb, inddweb] # could potentially remove 
                rhoeps  = rhoeps_arr[indtweb, inddweb]
                eps     = eps_arr[indtweb, inddweb]
                #nmean *= 0.1 # why? 
                
                if delta[ii,jj,kk] < dth:
                    ncounts[ii,jj,kk] = 0.
                else:
                    ncounts[ii,jj,kk] = (1. + delta[ii,jj,kk])**alpha * np.exp(-((1 + delta[ii,jj,kk])/rhoeps)**eps)
                    denstot_arr[indtweb,inddweb] += ncounts[ii,jj,kk]
    
    # SECOND LOOP: stochastic bias - we need to compute the right normalization beforehand
    denstot /= lbox**3

    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                indtweb = int(tweb[ii,jj,kk])-1
                inddweb = int(dweb[ii,jj,kk])-1

                denstot = denstot_arr[indtweb,inddweb]
                nmean = nmean_arr[indtweb,inddweb]

                ncounts[ii,jj,kk] = nmean / denstot * ncounts[ii,jj,kk]
                pnegbin = 1 - ncounts[ii,jj,kk]/(ncounts[ii,jj,kk] + beta)

                ncounts[ii,jj,kk] = negative_binomial(beta, pnegbin)


    return ncounts


@njit(parallel=False, cache=True, fastmath=True)
def prepare_indices_array(posx, posy, posz, ngrid, lbox):

    lcell = lbox / ngrid

    #posxarr = np.zeros(ngrid**3)
    #posyarr = np.zeros(ngrid**3)
    #poszarr = np.zeros(ngrid**3)

    posxarr = List()#np.empty(0)
    posyarr = List()
    poszarr = List()

    for ii in range(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                #ept = List([0.])
                posxarr.append(List([0.]))
                posyarr.append(List([0.]))
                poszarr.append(List([0.]))
            
    for ii in range(len(posx)):

        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]

        indx = int(xx / lcell)
        indy = int(yy / lcell)
        indz = int(zz / lcell)

        ind3d = int(indz + ngrid*(indy + ngrid*indx))

        posxarr[ind3d].append(xx)
        posyarr[ind3d].append(yy)
        poszarr[ind3d].append(zz)
    
    for ii in range(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                ind3dnew = int(kk + ngrid*(jj + ngrid*ii))
                posxarr[ind3dnew] = posxarr[ind3dnew][1:]
                posyarr[ind3dnew] = posyarr[ind3dnew][1:]
                poszarr[ind3dnew] = poszarr[ind3dnew][1:]

    return posxarr, posyarr, poszarr    

@njit(parallel=True, cache=True, fastmath=False)
def sample_galaxies(lbox, ngrid, posxprep, posyprep, poszprep, ncounts):
    lcell = lbox / ngrid 

    ncounts_sum = int(np.sum(ncounts))

    # Allocate arrays for final positions
    xtot = np.zeros(ncounts_sum)
    ytot = np.zeros(ncounts_sum)
    ztot = np.zeros(ncounts_sum)

    # Prepare indstart array
    indstart_arr = np.zeros((ngrid,ngrid,ngrid))

    indtmp = 0

    print('-->Find starting indices ...')
    for ii in range(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                indstart_arr[ii,jj,kk] = indtmp
                indtmp += ncounts[ii,jj,kk]

    print('-->Start loop over cells of the mesh ...')
    
    # Now loop over the cells of the mesh to perform HOD
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                #print(kk + ngrid*(jj + ngrid*ii))
                if ncounts[ii,jj,kk]>0:

                    ind3d = kk + ngrid*(jj + ngrid*ii)
                    indstart = int(indstart_arr[ii,jj,kk])

                    xtmp = np.asarray(posxprep[ind3d])
                    ytmp = np.asarray(posyprep[ind3d])
                    ztmp = np.asarray(poszprep[ind3d])

                    num_dm_part = len(xtmp)
                    num_gal_cell = int(ncounts[ii,jj,kk])

                    if num_gal_cell == num_dm_part:
                        # Fill with all the DM particles
                        xtot[indstart:indstart+num_gal_cell] = xtmp
                        ytot[indstart:indstart+num_gal_cell] = ytmp
                        ztot[indstart:indstart+num_gal_cell] = ztmp

                    elif num_gal_cell<num_dm_part:
 
                        # Fill with a random subsample of the DM particles
                        inddummy = np.arange(int(num_gal_cell))
                        np.random.shuffle(inddummy)

                        xtot[indstart:indstart+num_gal_cell] = xtmp[inddummy]
                        ytot[indstart:indstart+num_gal_cell] = ytmp[inddummy]
                        ztot[indstart:indstart+num_gal_cell] = ztmp[inddummy]

                    elif num_gal_cell>num_dm_part:
                        
                        # Fill part of the array with all the DM particles ...
                        if num_dm_part>0:
                            xtot[indstart:indstart+num_dm_part] = xtmp
                            ytot[indstart:indstart+num_dm_part] = ytmp
                            ztot[indstart:indstart+num_dm_part] = ztmp

                        # ... and the rest of the arrays with random positions within the cell
                        xinf = ii * lcell
                        xsup = (ii+1) * lcell 
                        yinf = jj * lcell
                        ysup = (jj+1) * lcell 
                        zinf = kk * lcell
                        zsup = (kk+1) * lcell 

                        xtot[indstart+num_dm_part:indstart+num_gal_cell] = np.random.uniform(xinf,xsup, size=(num_gal_cell-num_dm_part))
                        ytot[indstart+num_dm_part:indstart+num_gal_cell] = np.random.uniform(yinf,ysup, size=(num_gal_cell-num_dm_part))
                        ztot[indstart+num_dm_part:indstart+num_gal_cell] = np.random.uniform(zinf,zsup, size=(num_gal_cell-num_dm_part))

    return xtot, ytot, ztot
