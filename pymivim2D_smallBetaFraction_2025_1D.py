# -*- coding: utf-8 -*-
"""
Created on Wed May 25 12:57:27 2022

@author: WarnkenN

Added a branch on gitlab

v2.00
    Based on last 0D version
    To Do
        go back to being able to do 2D simulations.
        Making 2D-0D switch-able

v1.91_0D
    based on v1.90_0D.py
    - tidy up. Remove all that is not correct.


 Dev since start of v1.90_0D
   - conc field is really molar fraction! Design decision, makes some calculations easier
       molar fractions can be transformed easilty into concentrations
   - phi no longer calculated from molar fractions, instead calculate change of phi
       from number of moles crossing the interface.

v1.90_0D
    based on myPF_myPF_v1.75a_ternaryNonIdealParticle_growth_dampenDG_sameCompos.py
    - reduce calculations to a single point
    - produce table output of relevant data, phase fractions, chemical potentials, compositions
    - Allow skipping diffusion solver
    - Introduce 0D switch
    - Eventually branch of a new version
    


v1.75a:
    - I AM NOT SURE ABOUT THE SIGN OF corrFac (line ~1083). Look into the model. Changing this does not seem to influence the results much, which is worrying.
    - scale dG_therm, limit driving force, testing stability
    - increase cushion factor (?)
    - visualise driving forces
        -- added to image
    - a number of small bug fixes and mistakes in equations corrected.
        --> do a diff with v1.75 for details!!! Vm missing here and there, dx*dx missing etc.
            helps balancing driving forces for more stable results.
            Need to check if more of these are needed!!!
    - produce some documentation/animations
        - Shrinking
        - Shrinking with thermodynamic driving force
        - Less curvature effect
        - does the particle adopt a round shape and remain stable?
        - .... more of this...
        - does the interface remain stable?

v1.75:
    ToDo:
        - activate thermodynamic driving force
        - test, test, test
        - cross effects
        - tests for realistic thermodynamics
        - demo: particle dissolution - identical and different compositions
        - setting concA not to balance... is this the right thing to do? (line 731 ff)
        - write tables: phase fraction, total composition
        - include total composition into VTK output
        - split VTK into different files for different types of output


v1.7:
    - implemented simple anisotropy function in Sigma and Mob-interface


v1.6:
2023 02 26, end of day:
    - dissolution works, did many small corrections and tweaks
    - no crashing when phase dissapears
    - diffusion solver seems to work, diffusion confined to phases
    - now realistic initialisation of domain, phase is conc=0 where phase is not present
    - update of order parameter based on number of moles in phase, as written in paper draft.
        effectively creates a conserved order parameter.
    ---> could be the point of attack to introduce spinodal decomposition into the model!
    - after dissolution of particle with identical composition as matrix, no trace of particle left
        matrix is homogeneous. As it should be.
    - had some more ideas for demonstration... cannot remember now...
        - dissolution of particle at different composition.
        - dito with diffusion
    NEXT:
    - introduce thermodynamic driving force
    - think about interesting test/demonstration examples.
        - cross diffusion


2023 02 22, end of day...

- added weighting function to diffusion solver, 
    to restrict diffusion to points in space where phase is present
- somehow code does not run stable now, even when the weights are disables
- probably produced a bug when implemented Larsson thermodynamics
    -- did not test this properly after implementation.
    -- go back to just-interface version and find the bug... :-(
    -- also, can I test the weighting in the just-interface version?
    


2023 02 22
Continue working on this one.
Some previous versions were merged.
Next addressing 

    *diffusion solver
        - precent solute diffusing into non-existing phases

    *pf solver
        - get the thermodynamic transformation into the model



Simple implementation of my python model, using python,
objectives:
    
    - test the model
    - have a simple implementation
    - provide some basic results for the initial paper
    - up to ternary composition
    
A proper implementation using c++/fortran to follow...


!! ToDo
05 Feb 2023: Understand and comment current status
    - diffusion solver: Do a ternary-cross diffusion test
        -- I think the diffusion solver was complete...
    - Phase field




!! Status
=========
09 Feb 2023

- diffusion solver working
- conc-field initialised, homogeneous
- pf solver solves just the curvature term and redistribution
- removed most unncessary print statements
- pf interface mobility increased relative to atomistic mobility
- write a vtk-file writer

-- Phase field solver

-- IO 




"""

import numpy as np
import matplotlib.pyplot as plt



#
# some functions

# Post processing
#==================
def plotImage(time,rows):

    if(rows==2):
        fig, [(ax1,ax2),(ax3,ax4)] = plt.subplots(nrows=rows,ncols=2)
    else:
        fig, (ax1,ax2) = plt.subplots(nrows=rows,ncols=2)
        
    # conc-plot
#    ax1.set_title('c(C-beta), mu(c-beta)(l)')
#    ax1.set_title('c(B-beta), B-alph')
    ax1.set_title('dG_c, dGsum')
    ax1.set_aspect('equal')
    ax1.set_xticklabels([])  # suppress tick labels and ticks
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
#    pcm1=ax1.pcolormesh(conc,vmax=1.0,vmin=0.0,cmap='inferno')
#    pcm1=ax1.pcolormesh(conc[1:dimX+1,1:dimZ+1],vmax=1.0,vmin=0.0)

##    pcm1=ax1.pcolormesh(conc[1:dimX+1,1:dimZ+1,pBeta,elC])  # Plot concentration of elC in Beta phase

    #pcm1=ax1.pcolormesh(concTotal[1:dimX+1,1:dimZ+1,elB])  # Plot concentration of elC in Beta phase
    pcm1=ax1.pcolormesh(dGfield[1:dimX+1,1:dimZ+1,DGcurv])  # Plot concentration of elC in Beta phase

#    pcm1=ax1.pcolormesh(cliquid[1:dimX+1,1:dimZ+1])
    fig.colorbar(pcm1, ax=ax1)

    # phi-plot
#std    ax2.set_title('phi, c_B,beta (l)')
#    ax2.set_title('phi, d phi')
#    ax2.set_title('phi, C-alph')
    ax2.set_title('phi, mu(beta,B)')
    ax2.set_xticklabels([])  # suppress tick labels and ticks
    ax2.set_xticks([])
    ax2.set_yticklabels([])
    ax2.set_yticks([])
#    ax2.set_title('phi, d phi')
#    ax2.set_title('aux, c_B beta(l)')
    ax2.set_aspect('equal')
    pcm2=ax2.pcolormesh(phi[1:dimX+1,1:dimZ+1],vmax=1.0,vmin=0.0,cmap='inferno')
#    pcm2=ax2.pcolormesh( aux[1:dimX+1,1:dimZ+1,pBeta,elA])  # aux field, for debug
#    pcm2=ax2.pcolormesh(conc[1:dimX+1,1:dimZ+1,pAlpha,elA],vmax=1.0,vmin=0.0,cmap='inferno')  # DEBUG 23/08/2022! plot A-field
#    pcm2=ax2.pcolormesh(csolid[1:dimX+1,1:dimZ+1])
    fig.colorbar(pcm2, ax=ax2)

    if(rows==2):
        # conc Alpha-plot
        #ax3.set_title('mu C, beta')
        ax3.set_aspect('equal')
        ax3.set_xticklabels([])  # suppress tick labels and ticks
        ax3.set_xticks([])
        ax3.set_yticklabels([])
        ax3.set_yticks([])
#        pcm3=ax3.pcolormesh(conc[1:dimX+1,1:dimZ+1,pAlpha,elB])
        #pcm3=ax3.pcolormesh(dGfield[1:dimX+1,1:dimZ+1,DGB])
        pcm3=ax3.pcolormesh(dGfield[1:dimX+1,1:dimZ+1,DGsum])
        #pcm3=ax3.pcolormesh(mu[1:dimX+1,1:dimZ+1,pAlpha,elB])
        fig.colorbar(pcm3, ax=ax3)
        
        # conc Beta-plot
        #ax4.set_title('Beta B')
        ax4.set_aspect('equal')
        ax4.set_xticklabels([])  # suppress tick labels and ticks
        ax4.set_xticks([])
        ax4.set_yticklabels([])
        ax4.set_yticks([])
        ##pcm4=ax4.pcolormesh(conc[1:dimX+1,1:dimZ+1,pAlpha,elC])
        #pcm4=ax4.pcolormesh(mu[1:dimX+1,1:dimZ+1,pBeta,elB])
#        pcm4=ax4.pcolormesh(conc[1:dimX+1,1:dimZ+1,pAlpha,elC])
        #pcm4=ax4.pcolormesh(dGfield[1:dimX+1,1:dimZ+1,DGsum])
        pcm4=ax4.pcolormesh(mu[1:dimX+1,1:dimZ+1,pBeta,elB])
        #pcm4=ax4.pcolormesh( aux[1:dimX+1,1:dimZ+1,pBeta,elA])  # aux field, for debug
        fig.colorbar(pcm4, ax=ax4)
    plt.show()

#!!!
def plotGraph():
    rows = 2
    fig, [(ax1,ax2),(ax3,ax4)] = plt.subplots(nrows=rows,ncols=2)
    ax1.set_title('phi')
    ax1.plot(dist, phi[:,1])
    ax1.set_ylim(ymin=-0.1, ymax=1.1)
    ax2.set_title('conc B, alph')
    ax2.plot(dist, conc[:,1, pAlpha, elB])
    ax3.set_title('conc C, alph')
    ax3.plot(dist, conc[:,1, pAlpha, elC])
    ax4.set_title('conc B, beta')
    ax4.plot(dist, conc[:,1, pBeta, elB])
    plt.show()


def writeVTK(ti):
    print("write vtk at t=",time)
    # create file name from time-step and open file
    #   using 7 digits for numbering
    filename = baseName+f'_{ti:07d}.vtk'
    file = open(filename,"w")

    # write header
    file.write("# vtk DataFile Version 2.0\n")
    file.write("Output from mivim-python on (insert date function)\n")
    file.write("ASCII\n")
    file.write("DATASET STRUCTURED_POINTS\n")
    file.write("DIMENSIONS "+str(dimX)+" "+str(dimY)+" "+str(dimZ)+"\n")
    file.write("SPACING "+str(dx)+" "+str(dx)+" "+str(dx)+"\n")
    file.write("ORIGIN 0 0 0\n")
    dimXYZ = dimX * dimY * dimZ
    file.write("POINT_DATA  "+str(dimXYZ)  +"\n")
    
    #
    # write data...

    # phi
#    file.write("SCALARS phi float 1\n")
    file.write("SCALARS phi double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(phi[x,z]) +" ")
           pp = phi[xw,zw]
#           if pp < phiMin :
#               pp = 0.0
           file.write("{:.3e}".format(pp) +" ")
        file.write("\n")
    file.write("\n")
    file.flush()


    # Alpha phase compositions A, B and C
    # c_A in Alpha
    ph = pAlpha
    el = elA
    file.write("SCALARS conc(alpha,A) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ")
           file.write("{:.3e}".format(conc[xw,zw,ph,el]) +" ")
        file.write("\n")
    file.write("\n")
    # c_B in Alpha
    ph = pAlpha
    el = elB
    file.write("SCALARS conc(alpha,B) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ")
           file.write("{:.3e}".format(conc[xw,zw,ph,el]) +" ")
        file.write("\n")
    file.write("\n")
    # c_C in Alpha
    el = elC
    file.write("SCALARS conc(alpha,C) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ")
           file.write("{:.3e}".format(conc[xw,zw,ph,el]) +" ")
        file.write("\n")
    file.write("\n")
    file.flush()
    
    # Beta phase compositions A, B and C
    # c_A in Beta
    ph = pBeta
    el = elA
    file.write("SCALARS conc(beta,A) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ")
           file.write("{:.3e}".format(conc[xw,zw,ph,el]) +" ")
        file.write("\n")
    file.write("\n")
    # c_B in Beta
    ph = pBeta
    el = elB
    file.write("SCALARS conc(beta,B) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ")
           file.write("{:.3e}".format(conc[xw,zw,ph,el]) +" ")
        file.write("\n")
    file.write("\n")
    # c_C in Beta
    el = elC
    file.write("SCALARS conc(beta,C) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ") 
           file.write("{:.3e}".format(conc[xw,zw,ph,el]) +" ")
        file.write("\n")
    file.write("\n")
    
    # Total compositions A, B and C
    # c_A
    el = elA
    file.write("SCALARS concT(A) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ")
           file.write("{:.3e}".format(concTotal[xw,zw,el]) +" ")
        file.write("\n")
    file.write("\n")
    # c_B in Alpha
    ph = pAlpha
    el = elB
    file.write("SCALARS concT(B) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ")
           file.write("{:.3e}".format(concTotal[xw,zw,el]) +" ")
        file.write("\n")
    file.write("\n")
    # c_C in Alpha
    el = elC
    file.write("SCALARS concT(C) double 1\n")
    file.write("LOOKUP_TABLE default\n")
    for xw in range(1,dimX+1):
        for zw in range(1,dimZ+1):
           #file.write(str(conc[xw,zw,ph,el]) +" ")
           file.write("{:.3e}".format(concTotal[xw,zw,el]) +" ")
        file.write("\n")
    file.write("\n")
    file.flush()
    # close file
    file.flush()
    file.close()    





# define thermochemical data
maxNoElements = 3
maxNoPhases   = 2
elA = 0
elB = 1
elC = 2
pAlpha = 0
pBeta  = 1
mu0   = np.zeros((maxNoPhases, maxNoElements)               ,dtype=np.float64)
Lbin  = np.zeros((maxNoPhases, maxNoElements, maxNoElements),dtype=np.float64)
Ltern = np.zeros((maxNoPhases)                              ,dtype=np.float64)

mu0[pAlpha,elA] = 1.0
mu0[pAlpha,elB] = 1.0
mu0[pAlpha,elC] = 1.0

mu0[pBeta,elA] = 1.0
mu0[pBeta,elB] = 1.0
mu0[pBeta,elC] = 1.0

# Add regular solution parameters!






# thermodynamic data from Larsson, Strandlund, Hillert 2006
#   for calculations at 1300K
phase = pAlpha
mu0[phase,elA] = 50.0E3
mu0[phase,elB] = 70.0E3
mu0[phase,elC] = 60.0E3

Lbin[phase,elA,elB] = -10E3
Lbin[phase,elA,elC] = -40E3
Lbin[phase,elB,elC] = -15E3
Ltern[phase]        = 0.0
Ltern[phase]        = -10E3


phase = pBeta
mu0[phase,elA] = 70.0E3
mu0[phase,elB] = 50.0E3
mu0[phase,elC] = 30.0E3

Lbin[phase,elA,elB] = -20E3
Lbin[phase,elA,elC] = -50E3
#Lbin[phase,elB,elC] = 0.0E3
Lbin[phase,elB,elC] = +15E3
Ltern[phase]        = 0.0
Ltern[phase]        =  -5E3


# DEBUG! Remove all element interactions!
"""
phase = pAlpha
Lbin[phase,elA,elB] = 0.0
Lbin[phase,elA,elC] = 0.0
Lbin[phase,elB,elC] = 0.0
Ltern[phase]        = 0.0

phase = pBeta
Lbin[phase,elA,elB] = 0.0
Lbin[phase,elA,elC] = 0.0
Lbin[phase,elB,elC] = 0.0
Ltern[phase]        = 0.0
"""

# Chemical potential of elements in a given phase
# ===============================================
def chemPot(cA,cB,cC,phase):
#    muA = mu0[phase,elA] + R*temp*np.log(cA) + cB*(cB+cC)*Lbin[phase,elA,elB] + cC*(cB+cC)*Lbin[phase,elA,elC]+cB*cC*Lbin[phase,elB,elC]   
#    muA = mu0[phase,elB] + R*temp*np.log(cB) + cC*(cB+cC)*Lbin[phase,elA,elC] + cC*(cB+cC)*Lbin[phase,elA,elC]+cB*cC*Lbin[phase,elB,elC]   

    if (cA < 0 or cB < 0 or  cC<0) and DiffDebug > 0:
        print("x,z,cA,cB,cC",x,z,cA,cB,cC)
        
    
    # need to think about what the correct default value for mu should be,
    #  value returned if an element is not present... (2023 02 26)
    muA = 0.0  
    if(cA>0):
        muA = mu0[phase,elA] + (  R*temp*np.log(cA)  
                                + cB*(cB+cC)*Lbin[phase,elA,elB]
                                + cC*(cB+cC)*Lbin[phase,elA,elC]
                                - cB*cC     *Lbin[phase,elB,elC] 
                                + (1.0-2.0*cA)*cB*cC*Ltern[phase]
                                )
    muB = 0.0
    if(cB>0):
        muB = mu0[phase,elB] + (  R*temp*np.log(cB)
                                + cA*(cA+cC)*Lbin[phase,elA,elB] 
                                - cA* cC    *Lbin[phase,elA,elC]
                                + cC*(cA+cC)*Lbin[phase,elB,elC] 
                                + (1.0-2.0*cB)*cA*cC*Ltern[phase]
                                )
    muC = 0.0
    if(cC>0):
        muC = mu0[phase,elC] + (  R*temp*np.log(cC)
                                - cA*cB     *Lbin[phase,elA,elB] 
                                + cA*(cA+cB)*Lbin[phase,elA,elC]
                                + cB*(cA+cB)*Lbin[phase,elB,elC] 
                                + (1.0-2.0*cC)*cA*cB*Ltern[phase]
                                )

#    print("chemPot ",cA,  cB, cC, phase, cA+cB+cC)
#    print("        ",muA,muB,muC, phase)
#    print()

    return muA, muB, muC





def reInitAux():
    aux[:,:,:,:]=0.0
    #aux=0.0

def updateConcTotal():
    for el in range(0,noElem):
        concTotal[:,:,el] = conc[:,:,pAlpha,el] * (1.0-phi[:,:]) + conc[:,:,pBeta,el] * phi[:,:]
    
def calcFracTotal():
    frac = np.sum(phi[1:dimX+1,1:dimX+2])/(dimX*dimZ)
    return frac



def creatResTabLine():
    """
    Function returns states of the system to populate the resTable output structure.
    2023 04 03: 0D calculations only!

    Returns
    -------
    None.

    """
    # inti
    _x = 1
    _z = 1
    resTabLine[:,:] = 0.0
    
    # populate resTabLine
    resTabLine[0,iTime] = time
    resTabLine[0,itemp] = temp
    
    # fraction
    fra = calcFracTotal()
    resTabLine[0,ifrac] = fra
    
    # total compositions
    updateConcTotal()
    resTabLine[0,icAtot] = concTotal[_x,_z,elA]
    resTabLine[0,icBtot] = concTotal[_x,_z,elB]
    resTabLine[0,icCtot] = concTotal[_x,_z,elC]

    # phase compositions and chemical potentials
    #pp = phi[_x,_z]
    # alpha
    _ph = pAlpha
    _cB = conc[_x,_z,_ph,elB]
    _cC = conc[_x,_z,_ph,elC]
    _cA = 1.0 - _cB - _cC
    resTabLine[0,icAalph] = _cA
    resTabLine[0,icBalph] = _cB
    resTabLine[0,icCalph] = _cC

    muA, muB, muC = chemPot(_cA,_cB,_cC,_ph)
    resTabLine[0,imuAalph] = muA
    resTabLine[0,imuBalph] = muB
    resTabLine[0,imuCalph] = muC
    

    # beta
    _ph = pBeta
    _cB = conc[_x,_z,_ph,elB]
    _cC = conc[_x,_z,_ph,elC]
    _cA = 1.0 - _cB - _cC
    resTabLine[0,icAbeta] = _cA
    resTabLine[0,icBbeta] = _cB
    resTabLine[0,icCbeta] = _cC

    muA, muB, muC = chemPot(_cA,_cB,_cC,_ph)
    resTabLine[0,imuAbeta] = muA
    resTabLine[0,imuBbeta] = muB
    resTabLine[0,imuCbeta] = muC
    # done...


    #cross interface fluxes and netFlux
    resTabLine[0,ijA] = jIntFlux[elA]
    resTabLine[0,ijB] = jIntFlux[elB]
    resTabLine[0,ijC] = jIntFlux[elC]
    resTabLine[0,inetFlx] = jIntFlux[elA] + jIntFlux[elB] + jIntFlux[elC]
    
    



def writeResTab():
    filename = baseName+'_tab.dat'
    file = open(filename,"w")

    file.write("# Tabulated results, miVim python\n")
    cols='# '+'  '.join(resTabEntries)
    file.write(cols)
    file.write("\n")
    lines = len(resTable[:,0])
    for i in range(0,lines):
        #file.write('  '.join(resTable[i,:]))
        #file.write("\n")
#        np.save(file, resTable[i,:])
        resTable[i,:].tofile(file, sep='  ', format='%s')
        file.write("\n")
    # close file
    file.flush()
    file.close()    


def anisoSigma(pP,pW,pE,pS,pN,dx):
    """ Calculated anisotropy function a
        from given phi values of central and neighbour grid points.
    """
    aNX, aNZ = normalVectorUnit(pP,pW,pE,pS,pN,dx)
    
    
    nx2 = aNX * aNX
    nx4 = nx2 * nx2
    
    nz2 = aNZ * aNZ
    nz4 = nz2 * nz2
    
    aSigma = 1.0 + deltaSig*(4*(nx4 + nz4)-3.0)
    aMobil = 1.0 + deltaMob*(4*(nx4 + nz4)-3.0)
    
    return aSigma, aMobil


def normalVectorUnit(pP,pW,pE,pS,pN,dx):
    """ Calculated the normalised unit vector
        from given phi values of central and neighbour grid points.
    """

    #twoDX = 2.0 * dx   # not needed, cancels out during normalisation
    nx = (pE - pW)
    nz = (pN - pS)
    
    magnitude = np.sqrt((nx*nx + nz*nz))
    
    # there is no normal vector in the bulk, thus magnitude = 0. 
    #  need to catch this before it leads to nan error.
    #  not elegant but functional.
    if magnitude == 0:
        magnitude = 1.0
    else:
        nx /= magnitude
        nz /= magnitude
    
    return nx,nz

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# =======================================================================================
# =======================================================================================
#                                     Main Code
# =======================================================================================
# =======================================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Some switches
pointCalc = 1     # switch 0D calculation

#
# Some constants
R = 8.314    # Universal gas constant  [J/molK]

#Grid dimension, max 2D, numerical parameters
dimX = 100
dimY = 1
dimZ = 100
dimX = 100
dimZ = 100
dimX = 50
dimZ = 50
dimX = 600 # 300
dimZ = 1

#dimX = 1
#dimY = 1
#dimZ = 1

dx   = 0.01   # umeter [um]
DX2  = dx*dx  
dt = 0.50E-1  # time-step in [s]
dt = 0.60E-1  # time-step in [s]
dt = 1.0E-0  # time-step in [s]
dt = 1.0* 1.00E3  # time-step in [s]
#dt = 0.10E-1  # time-step in [s]
eps = 1E-10    # accuracy, values between this and zero are regarded as zero.
#dt = 0.30E-1  # time-step in [s]
noTimeSteps = 1000   # number of timestep in the simulation
noTimeSteps = 5000   # number of timestep in the simulation
noTimeSteps = 15000   # number of timestep in the simulation
#noTimeSteps = 500    # number of timestep in the simulation
outStep = 50      # plot result every n-step
#outStep = 50      # plot result every n-step

# for debug
noTimeSteps = 60000    # number of timestep in the simulation
outStep     = 100      # plot result every n-step


noTimeSteps = 400000    # number of timestep in the simulation
outStep     = 200      # plot result every n-step
outStep     = 20      # plot result every n-step



#noTimeSteps = 5    # number of timestep in the simulation
#outStep     = 1      # plot result every n-step


baseName = "ResultsGrowth/ternParticleGrowth_DG"
baseName = "Results0D/ternParticleGrowth_DG"
baseName = "resultsCALPHAD/3Particles/comp2025_1D"
outVTK = 1  # write vtk files
outPYT = 0  # write python plot files
rowsToPlot = 2    # either 1 or 2
outGRA = 1  # produce graphs, using python plots

# Creating table for results.
# Table is kind of self organising.
# labels of columns can be changed, as long as the identifiers are used to index the table.
resTabEntries = [
                 "Time"  , "temp"  , "frac"  , "curv"   ,
                 "cAalph", "cBalph", "cCalph", "muAalph", "muBalph", "muCalph",
                 "cAbeta", "cBbeta", "cCbeta", "muAbeta", "muBbeta", "muCbeta",
                 "cAtot" , "cBtot" , "cCtot" ,
                 "jA"    , "jB"    , "jC", "netFlx"
                 ]

noColumns = len(resTabEntries)
resTabLine = np.zeros((1,noColumns))
resTable   = resTabLine
# create identifiers to place data into the correct columns. Identifiers start with i
# see srcColumnEdit.txt for edited source
iTime      = resTabEntries.index("Time")
itemp      = resTabEntries.index("temp")
ifrac      = resTabEntries.index("frac")
icurv      = resTabEntries.index("curv")
icAalph    = resTabEntries.index("cAalph")
icBalph    = resTabEntries.index("cBalph")
icCalph    = resTabEntries.index("cCalph")
imuAalph   = resTabEntries.index("muAalph")
imuBalph   = resTabEntries.index("muBalph")
imuCalph   = resTabEntries.index("muCalph")
icAbeta    = resTabEntries.index("cAbeta")
icBbeta    = resTabEntries.index("cBbeta")
icCbeta    = resTabEntries.index("cCbeta")
imuAbeta   = resTabEntries.index("muAbeta")
imuBbeta   = resTabEntries.index("muBbeta")
imuCbeta   = resTabEntries.index("muCbeta")
icAtot     = resTabEntries.index("cAtot")
icBtot     = resTabEntries.index("cBtot")
icCtot     = resTabEntries.index("cCtot")
ijA        = resTabEntries.index("jA")
ijB        = resTabEntries.index("jB")
ijC        = resTabEntries.index("jC")
inetFlx    = resTabEntries.index("netFlx")



# indices of elements, used to reference arrays, 0 is always main element
#    indices of elements and phase have been defined above already.
noElem = 3
noPhas = 2


#Material and process properties
temp = 1000.0    # Temperature [K]
temp = 1300.0    # Temperature [K], as in Larsson et al.2006

Mob = np.zeros((noPhas,noElem),dtype=np.float64)


"""
# Initial guess
Mob[pAlpha,elA] = 1.0E-09     # atomistic mobility of element A
Mob[pAlpha,elB] = 1.0E-09     # atomistic mobility of element B
Mob[pAlpha,elC] = 1.0E-09     # atomistic mobility of element C

Mob[pBeta,elA] = 1.0E-09     # atomistic mobility of element A
Mob[pBeta,elB] = 1.0E-09     # atomistic mobility of element B
Mob[pBeta,elC] = 1.0E-09     # atomistic mobility of element C
"""

mobFacAlpha = 1.0   # easy way of modifying mobility in Alpha
Mob[pAlpha,elA] = mobFacAlpha * 5.0E-14     # atomistic mobility of element A
Mob[pAlpha,elB] = mobFacAlpha * 5.0E-14     # atomistic mobility of element B
Mob[pAlpha,elC] = mobFacAlpha * 5.0E-14     # atomistic mobility of element C

mobFacBeta = 1.0   # easy way of modifying mobility in Alpha
Mob[pBeta,elA] = mobFacBeta * 5.0E-14     # atomistic mobility of element A
Mob[pBeta,elB] = mobFacBeta * 5.0E-14     # atomistic mobility of element B
Mob[pBeta,elC] = mobFacBeta * 5.0E-14     # atomistic mobility of element C


# concentration of elements A, B and C
#      concentration of mol%?
concCInit_alph  = 0.00001E-2    
concCInit_alph  = 10.0E-2    
concBInit_alph  = 10.0E-2
concAInit_alph  = 100.0E-2 - concBInit_alph - concCInit_alph

concCInit_alph  = 0.8110 #+0.07    
concBInit_alph  = 0.0911 #+0.015
concAInit_alph  = 1.0 - concBInit_alph - concCInit_alph

concBInit_beta  = 0.435
concCInit_beta  = 0.2096
concAInit_beta  = 1.0 - concBInit_beta - concCInit_beta



"""
concBInit_alph  = 50.0E-2
concCInit_alph  = 0.0E-2    
concAInit_alph  = 100.0E-2 - concBInit_alph - concCInit_alph

#concBInit_beta  = concBInit_alph
#concCInit_beta  = concCInit_alph
concBInit_beta  = 50.0E-2
concCInit_beta  = 0.0E-2
concAInit_beta  = 100.0E-2 - concBInit_beta - concCInit_beta
"""


"""
concBInit_alph  = 0.10E-2
concCInit_alph  = 0.10E-2    
concAInit_alph  = 100.0E-2 - concBInit_alph - concCInit_alph

#concBInit_beta  = concBInit_alph
#concCInit_beta  = concCInit_alph
concBInit_beta  = 0.10E-2
concCInit_beta  = 0.10E-2
concAInit_beta  = 100.0E-2 - concBInit_beta - concCInit_beta
"""



concBInit_alph  = 20.0E-2
concCInit_alph  = 20.0E-2    
concAInit_alph  = 100.0E-2 - concBInit_alph - concCInit_alph

#concBInit_beta  = concBInit_alph
#concCInit_beta  = concCInit_alph
concBInit_beta  = 30.0E-2
concCInit_beta  = 30.0E-2
concAInit_beta  = 100.0E-2 - concBInit_beta - concCInit_beta

concBInit_beta  = 15.0E-2
concCInit_beta  = 20.0E-2
concAInit_beta  = 100.0E-2 - concBInit_beta - concCInit_beta


# Init beta phase at the same composition as alpha phase.
concBInit_alph  = 10.0E-2
concCInit_alph  = 40.0E-2    
concBInit_alph  = 20.0E-2
concCInit_alph  = 20.0E-2    

concBInit_alph  = 20.0E-2
concCInit_alph  = 25.0E-2    

# smaller Beta fraction
concBInit_alph  = 15.0E-2
concCInit_alph  = 18.0E-2    

#concBInit_alph  = 20.0E-2
#concCInit_alph  = 20.0E-2    

# Set beta to same composition as alpha
#concBInit_beta  = concBInit_alph
#concCInit_beta  = concCInit_alph


# 2025 -Tie-line
TieLineB2025= [2.61275E-01, 9.99172E-02]
TieLineC2025= [3.28467E-01, 1.21837E-01]
concBInit_alph = 9.99172E-02
concCInit_alph = 1.21837E-01

concBInit_beta = 2.61275E-01
concCInit_beta = 3.28467E-01

"""
concBInit_beta = 9.99172E-02
concCInit_beta = 1.21837E-01
concBInit_alph = 2.61275E-01
concCInit_alph = 3.28467E-01
"""

concBInit_alph = 0.10
concCInit_alph = concBInit_alph

concBInit_alph = 0.099917+0.0170   #+0.015, not stable
concCInit_alph = 0.121837+0.0170   #+0.015, not stable


#concBInit_beta = 0.29
#concCInit_beta = concBInit_beta
concBInit_beta = 2.61275E-01
concCInit_beta = 3.28467E-01

scaleF = 0.75
concBInit_alph  = 20.0E-2 * scaleF
concCInit_alph  = 25.0E-2 * scaleF 

# Set beta to same composition as alpha
concBInit_beta  = concBInit_alph
concCInit_beta  = concCInit_alph

# Init phase compositions according to eq tie line
concBInit_beta = TieLineB2025[0]
concCInit_beta = TieLineC2025[0]

concBInit_alph = TieLineB2025[1]
concCInit_alph = TieLineC2025[1]



concAInit_alph  = 100.0E-2 - concBInit_alph - concCInit_alph
concAInit_beta  = 100.0E-2 - concBInit_beta - concCInit_beta



"""
# Init beta phase at the same composition as alpha phase.
concBInit_alph  = 0.044848518324384065
concCInit_alph  = 0.010091502647872615
concAInit_alph  = 100.0E-2 - concBInit_alph - concCInit_alph
concBInit_beta  = concBInit_alph
concCInit_beta  = concCInit_alph
concAInit_beta  = 100.0E-2 - concBInit_beta - concCInit_beta
#Final tie-line
#ElB, beta, alpha: [0.27297604736891995, 0.044848518324384065]
#ElC, beta, alpha: [0.31362483650766226, 0.010091502647872615]
"""


Vm     = 10.0    # Molar volume [m**3/mole]
deltaZ = dx/10.0 # physical interface thickness


# Phase field parameters
eta   = 10.00*dx    # interface thickness [um]
#eta   = 3.00*dx    # interface thickness [um]
#eta   = 1.00*dx    # interface thickness [um]
#eta   = 0.50*dx    # interface thickness [um]
#eta   = 0.010*dx    # interface thickness [um]
sigma0   = 0.15E1      # interfacial energy [J/m**2]
sigma0   = 0.40E-1      # interfacial energy [J/m**2]
sigma0   = 0.10E-1      # interfacial energy [J/m**2]
#sigma0   = 150.0E-3      # interfacial energy [J/m**2]
#sigma0   = 1.5E-2      # interfacial energy [J/m**2]
deltaSig = 0.0 * 1.0/15.001 # Anisotropy strength of interfacial energy, max. 1/15
deltaMob = 0.0 * 1.0/15.001 # Anisotropy strength of interfacial mobility, max. 1/15
phiMin = 1E-4     # phi-min, below regard phi as 0, above 1-phiMin as 1

pMob = np.zeros((noElem),dtype=np.float64)
pMobInit = 1.0E-6
pMobInit = Mob[pAlpha,elA] * 10.0E4 # Set interface mobility to mob(A) in alpha
#pMobInit = 0.5E-6
pMob[elA] = pMobInit     # phase field mobility of element A
pMob[elB] = pMobInit     # phase field mobility of element B
pMob[elC] = pMobInit     # phase field mobility of element C


justCurvaturePF = 0   # =1 : no chemical driving force
einSchwing = 200
#einSchwing = 2
#einSchwing = 0

# Create the mesh
# x,y,z,number of phases,number of elements
#grid = np.zeros((dimX,dimZ),dtype=float)
conc      = np.zeros(((dimX+2),(dimZ+2), noPhas ,noElem),dtype=np.float64)
aux       = np.zeros(((dimX+2),(dimZ+2), noPhas ,noElem),dtype=np.float64)
phi       = np.zeros((dimX+2,dimZ+2),dtype=np.float64)
mu        = np.zeros(((dimX+2),(dimZ+2), noPhas ,noElem),dtype=np.float64)
concTotal = np.zeros(((dimX+2),(dimZ+2), noElem),dtype=np.float64)
# create fields to store driving forces for visualisation
noDG = 8
dGfield   = np.zeros(((dimX+2),(dimZ+2), noDG),dtype=np.float64)
# DG:
    # 0  : curvature
DGcurv = 0
    # 1-3: dmuA, dmuB, dmuC
DGA = 1
DGB = 2
DGC = 3
    # 4  : dmu_sum
DGsum= 4
    # 5-6: flux_c, flux_el
AREA= 7
    # local interfacial area

# Set parts of the phi field to test some functionality
#phi[21:51,21:47]=0.01
#NW
#####phi[42:48,42:48]=1.0
#phi[21:51,21:51]=1.0
#phi[33:46,33:46]=1.0
#phi[29:38,29:38]=1.0

"""
phi[1,1]=0.0005
phi[1,1]=0.01

phi[1,1]=0.5
phi[1,1]=0.80
phi[1,1]=8.2951E-01
phi[1,1]=8.2951E-03
"""
#phi[1,1]=9.30E-01
#phi[1,1]=0.63
#phi[1,1]=0.50

#eta = 0.125*dx  # carefule here!!!!!!!!!!!
#eta = 1.00*dx  # carefule here!!!!!!!!!!!
"""#phi[22:30,22:30]=1.0
phi[33:37,19:25]=1.0; phi[36:40,19:25]=1.0
#phi[2:5,40:45]=1.0
phi[10:15,10:17]=1.0; phi[18:23,33:40]=1.0
"""
rad=50; 
Xmin = int(dimX/2)-rad+1
Xmax = int(dimX/2)+rad+1
#phi[Xmin:Xmax,Xmin:Xmax]=1.0
#phi[Xmin:Xmax,:]=1.0
phi[Xmin:Xmax,:]=1.0

# set parts of conc field to test diffusion solver
testDiffPhase = pBeta
testDiffPhase = pAlpha
#conc[3:7,3:7, testDiffPhase, elB] = 0.30

#conc[10:30,10:46, testDiffPhase, elB] = 0.30
#conc[25:40,35:66, testDiffPhase, elC] = 0.05
#conc[:,:,:,elA] = 1.0 - conc[:,:,:,elB] - conc[:,:,:,elC]


#
# Initiate composition field, to match order parameter
# init grids, composition
#"""
# I don't like this. I think it is wrong. It make the calculation unpredictable
conc[:,:,pAlpha,elA] = concAInit_alph*(1.0-phi[:])
conc[:,:,pAlpha,elB] = concBInit_alph*(1.0-phi[:])
conc[:,:,pAlpha,elC] = concCInit_alph*(1.0-phi[:])

conc[:,:,pBeta,elA] = concAInit_beta * phi[:]
conc[:,:,pBeta,elB] = concBInit_beta * phi[:]
conc[:,:,pBeta,elC] = concCInit_beta * phi[:]

"""
conc[:,:,pAlpha,elA] = concAInit_alph
conc[:,:,pAlpha,elB] = concBInit_alph
conc[:,:,pAlpha,elC] = concCInit_alph

conc[:,:,pBeta,elA] = concAInit_beta
conc[:,:,pBeta,elB] = concBInit_beta
conc[:,:,pBeta,elC] = concCInit_beta
"""


# DiffDebug:
    # set parameters: pre-selected line along y, phase
    # Plots show: Composition of B, comp/chemp. of A, B, C
DiffDebug = 0   # 1=True, 0=False
diffDebugYline = 5
diffDebugPhase = testDiffPhase
diffDebugConc  = 1 # plot concentrtion (1=True), or chemical potentials (0=False)
diffSwitch = 1  # switch off diffusion: 0 -> no diffusion
diffSkip   = 0  # switch to skip diffusion block altogether. Some field are therefore not updated!
dist = np.zeros(dimX+2)
# create x-pos vector for plotting
for x in range(0,dimX+2):
    dist[x]=x*dx
jIntFlux = np.zeros(noElem)



# quick test of initialisation
# storing initial condition
time = 0.0
if DiffDebug==0:
    creatResTabLine()
#    updateConcTotal()
#    fra = calcFracTotal()
    fra = resTabLine[0,ifrac]
    print("Phase frac = ", fra)
 #   resTabLine[0,iTime] = time
 #   resTabLine[0,ifrac] = fra
    resTab = resTabLine.copy()
    if outPYT != 0 :
        plotImage(time,rowsToPlot)
    if outVTK != 0 :
        writeVTK(0)
    if outGRA != 0:
        plotGraph()
        

#
# Start time-loop
print("Start simulation");
for t in range(1,noTimeSteps+1):
    time=t*dt
    #print (t, time, "s")   # commeted out to save time...


    # Solve diffusion field, long range
    # =================================
    
    # Status of diffusion solver:
        # 05 Feb 2023: 
        # I think the diffusion solver works. 
        #Test the solver, interdiffusion of a ternary system, without phase field solver. (Do be done!)

    # update chemical potential fields for all phases
    for x in range(1,dimX+1):
        for z in range(1,dimZ+1):
            for phas in range(0,noPhas):
                cpB = conc[x,z,phas,elB]
                # This needs looking at ... 16/03/2023
                if cpB < eps :
                    cpB = 0.0
                    conc[x,z,phas,elB] = cpB
                cpC = conc[x,z,phas,elC]
                if cpC < eps :
                    cpC = 0.0
                    conc[x,z,phas,elC] = cpC
                cpA = 1.0 - cpB - cpC
                
                upA, upB, upC = chemPot(cpA,cpB,cpC,phas)

                mu[x,z,phas,elA] = upA
                mu[x,z,phas,elB] = upB
                mu[x,z,phas,elC] = upC

    #
    # update boundaries
    # Isolating boundaries, concentration
    #   python indexing is strange. Upper index is the upper limit! not the highest index.
    #   
    conc[0,:,:,:]      = conc[1,:,:,:]
    conc[dimX+1,:,:,:] = conc[dimX,:,:,:]
    conc[:,0,:,:]      = conc[:,1,:,:]
    conc[:,dimZ+1,:,:] = conc[:,dimZ,:,:]

    # Isolating boundaries, chemical potential
    mu[0,:,:,:]      = mu[1,:,:,:]
    mu[dimX+1,:,:,:] = mu[dimX,:,:,:]
    mu[:,0,:,:]      = mu[:,1,:,:]
    mu[:,dimZ+1,:,:] = mu[:,dimZ,:,:]

    # Isolating boundaries, phase field
    phi[0,:]      = phi[1,:]
    phi[dimX+1,:] = phi[dimX,:]
    phi[:,0]      = phi[:,1]
    phi[:,dimZ+1] = phi[:,dimZ]

    """
    #
    # period boundary conditions
    conc[0,:,:,:]      = conc[dimX,:,:,:]
    conc[dimX+1,:,:,:] = conc[1,:,:,:]
    conc[:,0,:,:]      = conc[:,dimZ,:,:]
    conc[:,dimZ+1,:,:] = conc[:,1,:,:]
    # period boundaries, chemical potential
    mu[0,:,:,:]      = mu[dimX,:,:,:]
    mu[dimX+1,:,:,:] = mu[1,:,:,:]
    mu[:,0,:,:]      = mu[:,dimZ,:,:]
    mu[:,dimZ+1,:,:] = mu[:,1,:,:]
    # period boundaries, phase field
    phi[0,:]      = phi[dimX,:]
    phi[dimX+1,:] = phi[1,:]
    phi[:,0]      = phi[:,dimZ]
    phi[:,dimZ+1] = phi[:,1]
    """


#09 Jul 2022, continue below, using the diffusion solver developed recently.

    #!!!
    # solve concentration diffusion
    # loop over space
       # python loops are to <dimX! Hence dimX+1 is upper limit.
    #print("Diffusion solver - start ...")   # saving time...
#    aux[:,:,:,:] = 0.0
    if diffSwitch == 1 :
        for x in range(1,dimX+1):
            for z in range(1,dimZ+1):
                # calc weights/Heaviside function
                hsE = x+z
    
                # loop over phases
                #   phi: fraction of precipitate phase, phase-index=1
                ppXZ     = phi[x,z]
                for phas in range(0,noPhas):
                    
                    #   getPhiA, getPhiB: helpers to convert phi into the
                    #                     the fraction of the correct phase
                    getPhiA = 1-phas
                    getPhiB =(-1)**(phas+1)
                    
                    # if phase ph not present, cycle
                    pp = getPhiA + getPhiB*ppXZ
                    #pp = ppXZ
                    #pp = phi[x,z]
                        
    #                if pp <= phiMin or pp >= 1.0-phiMin:
    #                    continue
    
                    # make sure fluxes are only counted, 
                    #   if the neighbouring cell contains the present phase
                    #    look at phi in neighbours, define a weight for the
                    #     diffusion flux: 
                    #       wt=0: no flux
                    wtW, wtE = getPhiA + getPhiB*phi[x-1,z], getPhiA + getPhiB*phi[x+1,z]
                    wtS, wtN = getPhiA + getPhiB*phi[x,z-1], getPhiA + getPhiB*phi[x,z+1]
    
                    # set correct weights, from phase fractions.
    
                    if wtW <= phiMin:
                        wtW = 0.0
                    else :
                        wtW = 1.0
    
                    if wtE <= phiMin:
                        wtE = 0.0
                    else :
                        wtE = 1.0
                        
                    if wtS <= phiMin:
                        wtS = 0.0
                    else :
                        wtS = 1.0
    
                    if wtN <= phiMin:
                        wtN = 0.0
                    else :
                        wtN = 1.0
    
                    # lines below remove restriction of diffusion to diffusion within
                    #  phases. Used for development and debugging and testing.
                    #wtE = 1.0
                    #wtW = 1.0
    
                    #wtS = 1.0
                    #wtN = 1.0
                    
                    # loop over components
                    # element 0 is the base element
                    for el in range(1,noElem):
                        # get local mole-fractions c and/or chemical potentials u
                        cp = conc[x  ,z  ,phas,el]
                        cw = conc[x-1,z  ,phas,el]
                        ce = conc[x+1,z  ,phas,el]
                        cs = conc[x  ,z-1,phas,el]
                        cn = conc[x  ,z+1,phas,el]
    
                        up =   mu[x  ,z  ,phas,el]
                        uw =   mu[x-1,z  ,phas,el]
                        ue =   mu[x+1,z  ,phas,el]
                        us =   mu[x  ,z-1,phas,el]
                        un =   mu[x  ,z+1,phas,el]
                        
                        # calculate fluxes in w,e,s,n direction
                        #jw = (-1.0) * ((Mob[phas,el] *np.sqrt(cp*cw) *Vm) /dx) *  (up - uw)
                        #je = (-1.0) * ((Mob[phas,el] *np.sqrt(cp*ce) *Vm) /dx) *  (ue - up)
                        #js = (-1.0) * ((Mob[phas,el] *np.sqrt(cp*cs) *Vm) /dx) *  (up - us)
                        #jn = (-1.0) * ((Mob[phas,el] *np.sqrt(cp*cn) *Vm) /dx) *  (un - up)
    
                        jw = wtW * (-1.0) * ((Mob[phas,el] * (cp+cw)/2.0 *Vm) /dx) *  (up - uw)
                        je = wtE * (-1.0) * ((Mob[phas,el] * (cp+ce)/2.0 *Vm) /dx) *  (ue - up)
                        js = wtS * (-1.0) * ((Mob[phas,el] * (cp+cs)/2.0 *Vm) /dx) *  (up - us)
                        jn = wtN * (-1.0) * ((Mob[phas,el] * (cp+cn)/2.0 *Vm) /dx) *  (un - up)
                        
    
                        # calculate change in cp and new cp (cnew)
                           #  fluxes w,s enter control-volume --> +
                           #  fluxes n,e leave control-volume --> -
                        cnew = cp+ (dt/dx) * (jw-je+js-jn) * diffSwitch
    
                        # store new cp
                        aux[x  ,z  ,phas,el] = cnew    

    # end of space loop, diffusion solver.

    # debug...
    #    plt.matshow(aux[1:dimX+1,1:dimZ+1,pAlpha,elB])
    #    plt.matshow(aux[1:dimX+1,1:dimZ+1,phas,elC])

    # copy new concentration field to grid.
    #    conc[1:dimX-1,1:dimZ-1,:,1:] = aux[1:dimX-1,1:dimZ-1,:,1:]
        conc[1:dimX,1:dimZ,:,1:] = aux[1:dimX,1:dimZ,:,1:]
    # 08/02/2023: the following does not work when aux gets reinitialised. Looks as then wrong parts are copied
    #  or shapes mismatch. Needs some more investigation.
    #conc  = aux

    # diffSwitch - end


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # update composition of dependent element
    #   Here is the origin of the problem with calculating phi from conc!
    #    This line sets cA to 1 when the phase does not exist!!!!!
    # IT SEEMS TO WORK WHEN THIS LINE IS COMMENTED OUT!
    # but need to check if overall conservation of mass is fullfilled!!!
    #
    # Could be that this line is not needed!
    #  diffusion solver assumes this anyway.
    #  pf solver explicitly tracks A atoms anyway
#    conc[:,:,:,elA] = 1.0 - conc[:,:,:,elB] - conc[:,:,:,elC]
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




    reInitAux()

    #print("   ... - finished")   # saving time...
#    print("  ")


 #   """
    # Debug, output testing the diffusion solver.
    if t%outStep==0 and DiffDebug==1:
        #fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=2, ncols=2)
        fig, [(ax1,ax2),(ax3,ax4)] = plt.subplots(nrows=2, ncols=2)

        # Plot composition of elB and elC in Alpha phase
        ax1.plot(dist, conc[:,diffDebugYline, diffDebugPhase, elB])
        ax1.set_title('conc B')

        if diffDebugConc == 0:
            # plot chemical potentials of elements A, B and C
            ax2.plot(dist, mu[:,diffDebugYline, diffDebugPhase, elA])
            ax2.set_title('mu A')
            ax3.plot(dist, mu[:,diffDebugYline, diffDebugPhase, elB])
            #ax3.set_title('mu B')
            ax4.plot(dist, mu[:,diffDebugYline, diffDebugPhase, elC])
            #ax4.set_title('mu C')
        else :
            # plot compositions of elements A, B and C
            ax2.plot(dist, conc[:,diffDebugYline, diffDebugPhase, elA])
            ax2.set_title('conc A')
            ax3.plot(dist, conc[:,diffDebugYline, diffDebugPhase, elB])
            ax4.plot(dist, conc[:,diffDebugYline, diffDebugPhase, elC])
            
        # plot summ of concentrations, for debug
        #ax4.plot(dist, (conc[:,0,pAlpha,elA]+conc[:,0,pAlpha,elB]+conc[:,0,pAlpha,elC]))

        plt.show()
    # end Debug, testing diffusion solver
#    """
    # ========================
    # end of diffusion solver.
    
    
    
    
    
    
    # !!!
    # Solve phase field
    # =================

    #print("Phase field solver - start ...")    # saving time...
    #
    # update boundaries
    # Isolating boundaries, concentration
    conc[0,:,:,:]      = conc[1,:,:,:]
    conc[dimX+1,:,:,:] = conc[dimX,:,:,:]
    conc[:,0,:,:]      = conc[:,1,:,:]
    conc[:,dimZ+1,:,:] = conc[:,dimZ,:,:]
    """
    # periodic boundaries, concentration
    conc[0,:,:,:]      = conc[dimX,:,:,:]
    conc[dimX+1,:,:,:] = conc[1,:,:,:]
    conc[:,0,:,:]      = conc[:,dimZ,:,:]
    conc[:,dimZ+1,:,:] = conc[:,1,:,:]
    # periodic boundaries, phi
    phi[0,:]      = phi[dimX,:]
    phi[dimX+1,:] = phi[1,:]
    phi[:,0]      = phi[:,dimZ]
    phi[:,dimZ+1] = phi[:,1]
    """

    # 08/02/2023: Not crashing, but not changes to phi... :-(
    #   find out why, checking all the different contribution by writing them in aux and visualise.

    #
    # Solve phase field
    for x in range(1,dimX+1):
        for z in range(1,dimZ+1):
            # get data for centre and current neighbouring points
            pp     = phi[x,z]
            pw, pe = phi[x-1,z], phi[x+1,z]
            ps, pn = phi[x,z-1], phi[x,z+1]
            
            # calc anisotropic sigma, mobility
            aSig, aMob = anisoSigma(pp,pw,pe,ps,pn,dx)
            sigma = aSig * sigma0

            # curvature potential without division by mag-grad-phi
            eta2 = eta*eta
            pot = (36/eta) * pp *(1-pp)*(0.5-pp)
            dG_c  = sigma * (((pn+pe+ps+pw-4*pp)/DX2) -pot)

            # Get the composition of all elements in Alpha and Beta
            #   then calculate the thermodynamic potential
            cAalpha = conc[x,z,pAlpha,elA]
            cBalpha = conc[x,z,pAlpha,elB]
            cCalpha = conc[x,z,pAlpha,elC]
            # calculate number of moles of each element in Alpha phase
            nAalpha = cAalpha * (1.0-pp)
            nBalpha = cBalpha * (1.0-pp)
            nCalpha = cCalpha * (1.0-pp)
            # ensure cAalpha is set correctly
            #   if there is no beta phase, all phase concentration have to be zero!
            # ... but this does not work. :-(
            if pp < phiMin :
                cAalpha = 0.0
                #cBalpha = 0.0
                #cCalpha = 0.0
            else :
                cAalpha  = 1.0 - cBalpha - cCalpha
            
            # ... but this seems to work...
            #cAalpha  = (1-pp)*(1.0 - cBalpha - cCalpha)
            #cAalpha  = (1.0-pp) - cBalpha - cCalpha    # This is wrong, violates mass conservation.
            cAalpha  = 1.0 - cBalpha - cCalpha
            nAalpha = cAalpha * (1.0-pp)
                

#            muAAlpha, muBAlpha, muCAlpha = chemPotAlpha(cA, cB, cC)  # old function, to be removed 07/02/23
            muAAlpha, muBAlpha, muCAlpha = chemPot(cAalpha,cBalpha,cCalpha,pAlpha)
            xA = cAalpha * (1-pp)  # Calculate mole fraction coefficient for cross interface flux
            xB = cBalpha * (1-pp)  #   xA is NOT the mole fraction! Variable naming is a little unfortunate
            xC = cCalpha * (1-pp)


            cAbeta = conc[x,z,pBeta,elA] # *pp  # remnant? error? *pp should be removed? Not really relevant, as cAbete gets calculated below...
            cBbeta = conc[x,z,pBeta,elB]
            cCbeta = conc[x,z,pBeta,elC]
            # calculate number of moles of each element in Beta phase
            nAbeta = cAbeta * pp
            nBbeta = cBbeta * pp
            nCbeta = cCbeta * pp
            # ensure cAbeta is set correctly
            #   if there is no beta phase, all phase concentration have to be zero!
            # ... but this does not work. :-(
            """
            if pp > (1.0-phiMin) :      # nicht schoen... needs a better solution.
                cAbeta = 0.0
                #cBbeta = 0.0
                #cCbeta = 0.0
            else :
                cAbeta = 1.0 - cBbeta - cCbeta
            """

            # ... but this seems to work...
            #cAbeta = pp*(1.0 - cBbeta - cCbeta)
            #cAbeta = (pp - cBbeta - cCbeta)      # This is wrong, violates mass conservation.
            cAbeta = 1.0 - cBbeta - cCbeta
            nAbeta = cAbeta * pp

            muABeta, muBBeta, muCBeta  = chemPot(cAbeta,cBbeta,cCbeta,pBeta)
            xA += cAbeta * pp
            xB += cBbeta * pp
            xC += cCbeta * pp


            # calculate delta-mu for each element...
            corrFac = -1.0   # positive or negative? I'm still not sure... check the equations.
            dG_cFac = 1.0
            #if  t>einSchwing:   # testing if a diffuse interface transforms to bulk due to therm-driv-force
            #    dG_cFac = 0.0

            dmuA = corrFac * (muABeta - muAAlpha)/Vm    # add Vm here, making dG's compareable and get it into the correct place for later!
            dmuB = corrFac * (muBBeta - muBAlpha)/Vm    # this needs checking! Are the molar volumes in the correct place???
            dmuC = corrFac * (muCBeta - muCAlpha)/Vm


            # calculate flux across interface
            # curvature fluxes first
            #   using atomistic mobility of the alpha phase (matrix)
            cushion = 100.0
            cushion = 1.0
            cushion = 0.80
            cushion = 0.30
            #cushion = 10.0
            
            
            # pre phase field mobility
            #coefA = (aMob*Mob[pAlpha,elA]*xA*cushion)/(deltaZ*Vm)
            #coefB = (aMob*Mob[pAlpha,elB]*xB*cushion)/(deltaZ*Vm)
            #coefC = (aMob*Mob[pAlpha,elC]*xC*cushion)/(deltaZ*Vm)

            coefA = (aMob*pMob[elA]*xA*cushion)/(deltaZ*Vm)
            coefB = (aMob*pMob[elB]*xB*cushion)/(deltaZ*Vm)
            coefC = (aMob*pMob[elC]*xC*cushion)/(deltaZ*Vm)

            # Chemical driving force fluxes
            #          dx*dx re-arranged to be on the left hand side of the equation, concentration change.
            areaInt  = (6.0/eta) * pp*(1-pp) * dx*dx  # dx*dx as it is 2D, interfacial area
            #aresInt = 0.01

            # phase field flux across interface, due curvature only!!!
            curvSwitch = 1.0    # why this??? 03 Apr 2023
#            if justCurvaturePF == 1 or t<einSchwing:
            if t<einSchwing:
                curvSwitch = 0.0
                #dmuA = 0.0
                #dmuB = 0.0
                #dmuC = 0.0
                """
                fak = 0.75
                fak = -10.0
                dampA = 1.0
                dampB = dampA
                dampC = dampA
                if dG_c != 0 :
                    dampA = fak * np.absolute(dmuA/dG_c)
                    #if dmuA != 0:
                        #print("dmuA ", dmuA, dG_c)
                    
                    dampB = fak * np.absolute(dmuB/dG_c)
                    dampC = fak * np.absolute(dmuC/dG_c)
                    #dmuA *= dampA
                    #dmuB *= dampB
                    #dmuC *= dampC
                else :
                    #dmuA = 0.0
                    #dmuB = 0.0
                    #dmuC = 0.0

                #dmuA = corrFac * fak * dG_c
                #dmuB = corrFac * fak * dG_c
                #dmuC = corrFac * fak * dG_c
                
#                dmuA = 0.0
#                dmuB = 0.0
#                dmuC = 0.0
                """


            #jACurv = coefA * (dG_c)
            #jBCurv = coefB * (dG_c)
            #jCCurv = coefC * (dG_c)

##            dG_c=0   # switch off curvature contribution!
            jACurv = coefA * (dG_cFac*dG_c  +  areaInt * dmuA  *curvSwitch)
            jBCurv = coefB * (dG_cFac*dG_c  +  areaInt * dmuB  *curvSwitch)
            jCCurv = coefC * (dG_cFac*dG_c  +  areaInt * dmuC  *curvSwitch)

            # store fluxes for output, nothing else is happening here.
            jIntFlux[elA] = jACurv
            jIntFlux[elB] = jBCurv
            jIntFlux[elC] = jCCurv
            
#            print("t, dmu ABC ",t, dmuA, dmuB, dmuC)



            # Calc number of moles crossing the interface
            dnA = jACurv * dt #   * areaInt   # Double check this equation!!! should the interfacial area be here! I think so!
            dnB = jBCurv * dt #   * areaInt   #   interfacial area is already included above.
            dnC = jCCurv * dt #   * areaInt
            dnTotalCurv = dnA + dnB + dnC


            
            
            # modifications to get the phase compositions (in at-fraction) correct.

            # update number of moles of each element in each phase
            #  and total number of moles in each phase
            nAbetaNew = nAbeta + dnA
            nBbetaNew = nBbeta + dnB
            nCbetaNew = nCbeta + dnC
            nBeta  = nAbetaNew + nBbetaNew + nCbetaNew

            nAalphaNew = nAalpha - dnA
            nBalphaNew = nBalpha - dnB
            nCalphaNew = nCalpha - dnC
            nAlpha =  nAalphaNew + nBalphaNew + nCalphaNew

#            print(dmuA, dmuB, dmuC,nAlpha,nBeta)
#            print("  dn A,B,C: ",dnA, dnB, dnC)

#!!!!!!!!!!!
#!!!!!!!!!!!
#!!!!!!!!!!!
#!!!!!!!!!!!
            # update phase composition
            # 19/05/2020 this needs checking
            #            and this needs a more elegant solution!!!
            if nBeta > 0 :
                cAbetaNew = nAbetaNew/nBeta
                cBbetaNew = nBbetaNew/nBeta
                cCbetaNew = nCbetaNew/nBeta
            else :
                cAbetaNew = cAbeta
                cBbetaNew = cBbeta
                cCbetaNew = cCbeta
            
            if nAlpha > 0 :
                cAalphaNew = nAalphaNew/nAlpha
                cBalphaNew = nBalphaNew/nAlpha
                cCalphaNew = nCalphaNew/nAlpha
            else :
                cAalphaNew = cAalpha
                cBalphaNew = cBalpha
                cCalphaNew = cCalpha
            

            
            #conc[x,z,pBeta,elA] = cAbetaNew   # storing new phase compositions, overwriting old ones.
            conc[x,z,pBeta,elA] = cAbetaNew   # storing new phase compositions, overwriting old ones.
            conc[x,z,pBeta,elB] = cBbetaNew
            conc[x,z,pBeta,elC] = cCbetaNew
            
            conc[x,z,pAlpha,elA] = cAalphaNew
            conc[x,z,pAlpha,elB] = cBalphaNew
            conc[x,z,pAlpha,elC] = cCalphaNew
            

            # update the order parameter.
            #  ... should be able to calculate this now from number of moles in each phase
            ppNew = nBeta / (nAlpha + nBeta)
            aux[x  ,z  ,pBeta,elA] = ppNew

            # apply ph-min criterion
            

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # THIS NEED SORTING OUT! 
            # I think there is something wrong here.... :-(

            # update order parameter
            #   oder parameter phi = fraction of beta phase!
            #   store increments of phi, then add them to phi once calculated for the entire field
            #   By definition in this example, \sum\sum n_i^k = 1
            #   summing up number of moles of all elements in all phases present at a grid point is one.
            #   increment dphi is thus the sum of the changes of moles in the beta phase.

            # this gives a stable phase field
            #   but I am not sure it gives correct phase compositions... and phi evolution
#            aux[x  ,z  ,pBeta ,elA] = pp - dnTotalCurv
#            aux[x  ,z  ,pAlpha,elA] = pp + dnTotalCurv

            # this is what I've wrote in the paper draft
            #   but it gives phi=0.5 everywhere if both phases have the same composition
            #     interface can spread beyond the interface region... BAD!
            #
            #  This will only work if the phase compositions have been initialised 
            #    correctly. 
            #    Molar fractions/compositions should be zero outside areas o
            #    of phase existence.
#            cAlphaTotal = cAalphaNew + cBalphaNew + cCalphaNew
#            cBetaTotal  = cAbetaNew  + cBbetaNew +  cCbetaNew
            
#            ppNew = cBetaTotal / (cAlphaTotal + cBetaTotal)
            #ppNew = pp + dnTotalCurv
#            aux[x  ,z  ,pAlpha,elA] = ppNew
            
            # store driving forces in fields, for output and development
            dGfield[x,z,DGcurv] = dG_c

            dGfield[x,z,DGA]    = areaInt * dmuA
            dGfield[x,z,DGB]    = areaInt * dmuB
            dGfield[x,z,DGC]    = areaInt * dmuC
            dGfield[x,z,DGsum]    = dG_c + areaInt * (dmuA+dmuB+dmuC)

            dGfield[x,z,AREA]    = areaInt


            # Old debug?
            #if pp+dnTotalCurv < 0 :
            #    print("Uwaga: ",t,x,z,pp,dnTotalCurv)
            
#dbg            if dnTotalCurv != 0.0 :
#dbg                print("dn... = ", dnTotalCurv,pp,aux[x,z,pAlpha,elA])
            

            # THIS NEED SORTING OUT! 
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # check mass conservation ...
            
    # update phi by adding increments
#    phi[:,:] = phi[:,:] + aux[:,:,pBeta,elA]
    phi[:,:] = aux[:,:,pBeta,elA]
    reInitAux()
    #print("   ... - finished")   # saving time...
    #print("  ")

    
    # output?
    # =======
    if t%outStep==0 and DiffDebug==0:
        creatResTabLine()
#        updateConcTotal()
#        fra = calcFracTotal()
        fra = resTabLine[0,ifrac]
#        resTabLine[0,iTime] = time
#        resTabLine[0,ifrac] = fra
        resTable = np.append(resTable, resTabLine, axis=0)
 #       resTabLine[:,:] = 0.0

        print ("output: ",t, " / ",noTimeSteps,"  ", time, "s")
        print("Phase frac = ", fra)
        if outPYT != 0 :
            plotImage(time,rowsToPlot)
        if outVTK != 0 :    
            writeVTK(t)
        if outGRA != 0:
            plotGraph()

        print(" ")
        
#   print("End of iteration ",t)
    #print(" ")         # saving time...        
# end of time loop

# Plot some results

if outGRA != 0 :
    # Beta fraction
    plt.plot(resTable[:,iTime],resTable[:,ifrac])
    plt.xlabel('time [s]')
    plt.ylabel('beta frac []')
    plt.title('beta frac vs. time')
    plt.show()
    
    
    # phase compositions, isothermal section style B-C axis
    resTabLines = len(resTable[:,0])
    tieLines = [1, 2, 4, 8, 12, 25, 30, 35, 40, 45, 50, 75, 90, 100]
    #TieLineB= [resTable[-1,icBbeta], resTable[-1,icBalph]]
    #TieLineC= [resTable[-1,icCbeta], resTable[-1,icCalph]]
    
    plt.plot(resTable[:,icBalph],resTable[:,icCalph], label='alpha')
    plt.plot(resTable[:,icBbeta],resTable[:,icCbeta], label='beta')
    
    # -- plot a number of transient tie-lines
    for i in tieLines:
        tieTime = int(i/100 * resTabLines)-1
        print(tieTime)
        TieLineB= [resTable[tieTime,icBbeta], resTable[tieTime,icBalph]]
        TieLineC= [resTable[tieTime,icCbeta], resTable[tieTime,icCalph]]
        plt.plot(TieLineB[:], TieLineC[:],  linewidth=0.5, color='green')
        
        #tieTime = 10
        #TieLineB= [resTable[tieTime,icBbeta], resTable[tieTime,icBalph]]
        #TieLineC= [resTable[tieTime,icCbeta], resTable[tieTime,icCalph]]
        #plt.plot(TieLineB[:], TieLineC[:], '--', linewidth=1)
        
        
        #plt.plot(TieLineB2020[:], TieLineC2020[:], '--', linewidth=1, color='black')
        #plt.plot(TieLineB2025[:], TieLineC2025[:], '--', linewidth=1, color='black')
        #plt.plot(TieLineB1040[:], TieLineC1040[:], '--', linewidth=1, color='black')
        plt.xlabel('B')
        plt.ylabel('C')
        plt.title('Isothermal section of PD A'+str(concBInit_alph)+'B'+str(concCInit_alph)+'C')
        plt.legend()
        # Shrink current axis by 20%
        #box = plt.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    print("Final tie-line")
    print("ElB, beta, alpha:", TieLineB)
    print("ElC, beta, alpha:", TieLineC)
    
    
    # Tie line animation
    resTabLines = len(resTable[:,0])
    
    
    # -- plot a number of transient tie-lines
    section = 1.0
    for i in range(0, int(resTabLines*section), int(resTabLines*section/400)):
        #tieTime = int(i/100 * resTabLines)-1
        tieTime = i
        print(tieTime)
        
        plt.rcParams['figure.dpi']=100
        plt.plot(resTable[:,icBalph],resTable[:,icCalph], '--', label='alpha')
        plt.plot(resTable[:,icBbeta],resTable[:,icCbeta], '--',  label='beta')
        
        TieLineB= [resTable[tieTime,icBbeta], resTable[tieTime,icBalph]]
        TieLineC= [resTable[tieTime,icCbeta], resTable[tieTime,icCalph]]
        plt.plot(TieLineB[:], TieLineC[:],  linewidth=1, color='green', )
        
        plt.plot(TieLineB2020[:], TieLineC2020[:], '--', linewidth=0.5, color='black')
        plt.plot(TieLineB2025[:], TieLineC2025[:], '--', linewidth=0.5, color='black')
        plt.plot(TieLineB1040[:], TieLineC1040[:], '--', linewidth=0.5, color='black')
        plt.plot(TieLineB1225[:], TieLineC1225[:], '--', linewidth=0.5, color='black')
        plt.plot(TieLineB1625[:], TieLineC1625[:], '--', linewidth=0.5, color='black')
        #    plt.plot(TieLineB2007[:], TieLineC2007[:], '--', linewidth=0.5, color='black')
        plt.plot(TieLineB0625[:], TieLineC0625[:], '--', linewidth=0.5, color='black')
        plt.xlabel('B')
        plt.ylabel('C')
        plt.title('Isothermal section of PD A'+str(concBInit_alph)+'B'+str(concCInit_alph)+'C')
        plt.legend()
        #    plt.xlim(0.195,0.205)
        #    plt.ylim(0.24,0.26)
        # Shrink current axis by 20%
        #box = plt.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    
    
    # phase compositions, as f(time)
    plt.plot(resTable[:,iTime],resTable[:,icAalph], label='Aalph')
    plt.plot(resTable[:,iTime],resTable[:,icAbeta], label='Abeta')
    plt.plot(resTable[:,iTime],resTable[:,icBalph], label='Balph')
    plt.plot(resTable[:,iTime],resTable[:,icBbeta], label='Bbeta')
    plt.plot(resTable[:,iTime],resTable[:,icCalph], label='Calph')
    plt.plot(resTable[:,iTime],resTable[:,icCbeta], label='Cbeta')
    plt.xlabel('time [s]')
    plt.ylabel('conc')
    plt.title('phase compositions vs. time')
    plt.legend()
    plt.show()
    
    # chem potentials, as f(time)
    plt.plot(resTable[:,iTime],resTable[:,imuAalph], label='muAalph')
    plt.plot(resTable[:,iTime],resTable[:,imuAbeta], label='muAbeta')
    plt.plot(resTable[:,iTime],resTable[:,imuBalph], label='muBalph')
    plt.plot(resTable[:,iTime],resTable[:,imuBbeta], label='muBbeta')
    plt.plot(resTable[:,iTime],resTable[:,imuCalph], label='muCalph')
    plt.plot(resTable[:,iTime],resTable[:,imuCbeta], label='muCbeta')
    plt.xlabel('time')
    plt.ylabel('mu')
    plt.title('chemical potentials vs. time')
    #plt.legend(['Aalph','Balph','Calph','Abeta','Bbeta','Cbeta'])
    plt.legend(loc='lower right')
    plt.show()
    
    # chem potential differences, as f(time)
    plt.plot(resTable[:,iTime],resTable[:,imuAbeta]-resTable[:,imuAalph], label='dmu A')
    plt.plot(resTable[:,iTime],resTable[:,imuBbeta]-resTable[:,imuBalph], label='dmu B')
    plt.plot(resTable[:,iTime],resTable[:,imuCbeta]-resTable[:,imuCalph], label='dmu C')
    plt.xlabel('time')
    plt.ylabel('??')
    plt.title('chemical potential differences vs. time')
    #plt.legend(['Aalph','Balph','Calph','Abeta','Bbeta','Cbeta'])
    plt.legend(loc='lower right')
    plt.show()
    
    # slopes of tanget to G curve, as f(time)
    plt.plot(resTable[:,iTime],resTable[:,imuBalph]-resTable[:,imuAalph], label='slope AB, alpha')
    plt.plot(resTable[:,iTime],resTable[:,imuCalph]-resTable[:,imuAalph], label='slope AC, alpha')
    plt.plot(resTable[:,iTime],resTable[:,imuCalph]-resTable[:,imuBalph], label='slope AC, alpha')
    plt.plot(resTable[:,iTime],resTable[:,imuBbeta]-resTable[:,imuAbeta], label='slope AB, alpha')
    plt.plot(resTable[:,iTime],resTable[:,imuCbeta]-resTable[:,imuAbeta], label='slope AC, alpha')
    plt.plot(resTable[:,iTime],resTable[:,imuCbeta]-resTable[:,imuBbeta], label='slope AC, alpha')
    plt.xlabel('time')
    plt.ylabel('??')
    plt.title('slopes of tangent vs. time')
    #plt.legend(['Aalph','Balph','Calph','Abeta','Bbeta','Cbeta'])
    plt.legend(loc='lower right')
    plt.show()
    
    # partition ratios
    plt.plot(resTable[:,iTime],resTable[:,icAalph]/resTable[:,icAbeta], label='k A')
    plt.plot(resTable[:,iTime],resTable[:,icBalph]/resTable[:,icBbeta], label='k B')
    plt.plot(resTable[:,iTime],resTable[:,icCalph]/resTable[:,icCbeta], label='k C')
    plt.xlabel('time')
    plt.title('partition ratios vs. time')
    plt.legend(loc='lower right')
    plt.show()
    
    # total composition
    #plt.plot(resTable[:,iTime],resTable[:,icAtot], label='A total')
    plt.plot(resTable[:,iTime],resTable[:,icBtot], label='B total')
    plt.plot(resTable[:,iTime],resTable[:,icCtot], label='C total')
    plt.xlabel('time')
    plt.title('total composition vs. time')
    plt.legend(loc='lower right')
    plt.show()
    
    # flux across interface
    plt.plot(resTable[:,iTime],resTable[:,ijA], label='A flux')
    plt.plot(resTable[:,iTime],resTable[:,ijB], label='B flux')
    plt.plot(resTable[:,iTime],resTable[:,ijC], label='C flux')
    plt.plot(resTable[:,iTime],resTable[:,inetFlx], label='net Flx')
    plt.xlabel('time')
    plt.title('cross interface flux vs. time')
    plt.legend(loc='lower right')
    plt.show()
    
writeResTab()
# finalise...
