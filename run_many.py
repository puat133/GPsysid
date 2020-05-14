'''
This python file will run many simulations
with different parameters
'''
import subprocess
import numpy as np
import a
import time
# samples=$1 
# lQ=$2
# Vgain=$3
# ratioL=$3
# ell=$4
# folderName=$5






if __name__=='__main__':

    pause = 10# seconds
    samples = 100
    ells = np.array([1.])
    lQs  = np.array([100])
    Vgains = np.array([1e2,1e3,1e4,1e5,1e6])
    ratioLs = np.array([2])
    for ell in ells:
        for lQ in lQs:
            for Vgain in Vgains:
                for ratioL in ratioLs:
                    folderName = "ell_{}_lQ_{}_Vgain_{}_ratioL_{}".format(ell,lQ,Vgain,ratioL)
                    subprocess.run(['sbatch','runner.slrm',str(samples),str(lQ),str(Vgain),
                                str(ratioL),str(ell),folderName])

                    #pause 1 second after submission
                    time.sleep(pause)
                    

