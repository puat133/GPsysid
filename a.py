import numpy as np
import numba as nb
import util
from numba.typed import List,Dict #numba typedList and typedDict

njitParallel = nb.njit(parallel=True,fastmath=True)
njitSerial = nb.njit(parallel=False,fastmath=True)


class dynamic:
    def __init__(self,dynamicFun,measFun,nx,nu):
        self.dynamicFun = None
        self.measFun = None
        self.nx = 1
        self.nu = 1
