'''

generate and run slurm scripts for TACC Lonestar

'''
import os, sys
import numpy as np


def run_alpt_lhc(i0, i1):
    ''' run ALPT for specific LHC realization 
    '''
    dir_lhc = '/corral/utexas/AST25023/simbig/quijote/latinhypercube_hr/'

    # write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -J alpt.lhc.%i_%i' % (i0, i1),
        '#SBATCH -o o/alpt.lhc.%i_%i' % (i0, i1),
        '#SBATCH -p development', 
        '#SBATCH -N 1',               
        '#SBATCH -n 1',               
        '#SBATCH --time=00:10:00',
        '#SBATCH -A AST25023', 
        '',
        "module purge ",
        "module load fftw3/3.3.10 gsl", 
        "module load intel",  
        "module load impi", 
        "", 
        "unset PYTHONPATH", 
        "source ~/.bashrc", 
        "", 
        "conda activate simbig",
        '',
        ''])
    
    for i_lhc in range(i0, i1): 
        a += "python /home1/11004/chahah/projects/hodalpt/bin/lhc_alpt/alpt.py %s %i\n" % (dir_lhc, i_lhc)

    # create the script.sh file, execute it and remove it
    f = open(os.path.join(os.environ['WORK'], 'script.slurm'),'w')
    f.write(a)
    f.close()
    os.system('sbatch %s' % os.path.join(os.environ['WORK'], 'script.slurm'))
    #os.system('rm script.slurm')
    return None


if __name__=="__main__": 
    run_alpt_lhc(1, 2)
