#!/bin/python
import os 
import numpy as np 

ff = open('transfer_manifest.txt', 'w')
for i in [78, 79]+list(np.arange(81, 100)): #range(2000): 
    ff.write('%i/CAMB.params %i/CAMB.params\n' % (i, i))
    ff.write('%i/Cosmo_params.dat %i/Cosmo_params.dat\n' % (i, i))
    ff.write('%i/ICs/2LPT.param %i/ICs/2LPT.param\n' % (i,i))
    ff.write('%i/ICs/ics.0.hdf5 %i/ICs/ics.0.hdf5\n' % (i,i))
    ff.write('%i/ICs/ics.1.hdf5 %i/ICs/ics.1.hdf5\n' % (i,i))
    ff.write('%i/ICs/ics.2.hdf5 %i/ICs/ics.2.hdf5\n' % (i,i))
    ff.write('%i/ICs/ics.3.hdf5 %i/ICs/ics.3.hdf5\n' % (i,i))
    ff.write('%i/ICs/ics.4.hdf5 %i/ICs/ics.4.hdf5\n' % (i,i))
    ff.write('%i/ICs/ics.5.hdf5 %i/ICs/ics.5.hdf5\n' % (i,i))
    ff.write('%i/ICs/ics.6.hdf5 %i/ICs/ics.6.hdf5\n' % (i,i))
    ff.write('%i/ICs/ics.7.hdf5 %i/ICs/ics.7.hdf5\n' % (i,i))
    ff.write('%i/ICs/inputspec_ics.txt %i/ICs/inputspec_ics.txt\n' % (i,i))
    ff.write('%i/ICs/Pk_mm_z=0.000.txt %i/ICs/Pk_mm_z=0.000.txt\n' % (i,i))
ff.close() 

