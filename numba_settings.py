import numba as nb


CACHE = True
PARALLEL = True
FASTMATH = True
jitSerial = nb.jit(parallel=False,fastmath=FASTMATH,cache=CACHE)
jitParallel = nb.jit(parallel=PARALLEL,fastmath=FASTMATH,cache=CACHE)
njitParallel = nb.njit(parallel=PARALLEL,fastmath=FASTMATH,cache=CACHE)
njitSerial = nb.njit(parallel=False,fastmath=FASTMATH,cache=CACHE)