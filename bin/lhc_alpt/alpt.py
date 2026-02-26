#!/bin/python
''' python script to run 
'''
import os, sys
import numpy as np 
import matplotlib.pyplot as plt
from hodalpt.sims import alpt as CS

# directory where Quijote LHC is located /corral/utexas/AST25023/simbig/quijote/latinhypercube_hr
dir_lhc = sys.argv[1] 
i_lhc = int(sys.argv[2]) # LHC realization

ic_path = os.path.join(dir_lhc, str(i_lhc), 'ICs')
outdir = os.path.join(dir_lhc, str(i_lhc), 'alpt')

# run subgrid off
if not os.path.isfile(os.path.join(outdir, 'Quijote_ICs_delta_z127_n256_CIC.DAT')): 
    CS.CSbox_alpt(ic_path, outdir, seed=0, dgrowth_short=5., make_ics=True, subgrid=False, silent=False)
else: 
    CS.CSbox_alpt(ic_path, outdir, seed=0, dgrowth_short=5., make_ics=False, subgrid=False, silent=False)

# run subgrid on 
CS.CSbox_alpt(ic_path, outdir, seed=0, dgrowth_short=5., make_ics=False, subgrid=True, silent=False)
