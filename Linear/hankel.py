import numpy as np
import numba as nb


#stack x to a matrix dimension of (nx r) x (nT-r), r is the horizon
@nb.njit(fastmath=True,cache=True )
def constructHistorical(input_record,horizon,past=True):
    nx = input_record.shape[1]
    nT = input_record.shape[0]
    
    startIndex = 0 if past else horizon
    Xp = np.zeros((nT-(horizon+startIndex),nx*horizon))
    for i in range(nT-(horizon+startIndex)):
        Xp[i,:] = input_record[startIndex+i:(startIndex+horizon+i),:].flatten()
        
    return Xp
