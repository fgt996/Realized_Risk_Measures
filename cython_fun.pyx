
import cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sim_from_inn(double[:,::1] innovations, double[::1] mu, double[::1] sigma):
    cdef:
        unsigned int i, j, k
        unsigned int N = innovations.shape[0]
        unsigned int c = innovations.shape[1]
        unsigned int N_exp = len(mu)
        double[::1] intra_day = np.empty(c)
        double[:,::1] intra_day_mat = np.empty((N,c))

    for k in range(N_exp):
        for i in range(N):
            
            for j in range(c):
                intra_day_mat[i,j] = mu[k] + sigma[k]*innovations[i,j]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ar1_predict_update(double start_y, double[::1] y, double[::1] params):
    cdef:
        unsigned int t
        unsigned int N = len(y)
        double[::1] output = np.empty(N)

    output[0] = params[0] + params[1]*start_y
    for t in range(1, N):
        output[t] = params[0] + params[1]*y[t-1]
    
    return np.asarray(output)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fill_nan(double[::1] data, double curr_val):
    cdef:
        unsigned int i
        unsigned int N = len(data)
        double[::1] output=np.empty(N)

    for i in range(N):
        if (data[i]>-1) or (data[i]<1):
            curr_val = data[i]
        output[i] = curr_val
    
    return np.asarray(output)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ESWA(double[::1] data_train, double[::1] data_test, double alpha):
    cdef:
        unsigned int i
        unsigned int N_train = len(data_train)
        unsigned int N_test = len(data_test)
        double curr_est = 0
        double[::1] output=np.empty(N_test)

    for i in range(N_train):
        curr_est = alpha*data_train[i] + (1-alpha)*curr_est
    
    for i in range(N_test):
        output[i] = curr_est
        curr_est = alpha*data_test[i] + (1-alpha)*curr_est
    
    return np.asarray(output)

