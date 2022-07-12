from libc.stdlib cimport rand, RAND_MAX, malloc, free
import numpy as np
from collections import Counter
import time
# New compilation

cdef double p01(double t1, double t2):
     return np.abs(np.cos(t1/2) * np.sin(t2/2))**2

cdef double p10(double t1, double t2):
     return np.abs(np.sin(t1/2) * np.cos(t2/2))**2

cdef double p00(double t1, double t2):
     return np.abs(np.cos(t1/2) * np.cos(t2/2))**2

cdef double p11(double t1, double t2):
     return np.abs(np.sin(t1/2) * np.sin(t2/2))**2
 
cdef double p0110(double t1, double f1, double t2, double f2):
     return (p01(t1,t2) + p10(t1,t2) + 2 * np.sqrt(p01(t1,t2)*p10(t1,t2)) * np.cos(f2-f1)) / 2

cdef double G(double c0110, double c01, double c10):
     return (2*c0110 - c01 - c10) / (2 * np.sqrt(c01 * c10))

# F
cdef double F(double *g):
    return g[0]**2 + g[1]**2 + g[2]**2 - 2*g[0]*g[1]*g[2]


cdef int * binomial(double p[], const int size_p, const int num):
    # Store the length of the p list
    cdef int m = size_p+1

    # Create a array of cumulative probabilities
    cdef double cumulative = 0
    cdef double *cp = <double *> malloc(m * sizeof(double))

    # The first element of cp is 0
    cp[0] = 0

    # Fill the cumulative probs
    cdef int i, j
    cdef double r
    
    for i in range(m-1):
        cumulative += p[i]
        cp[i+1] = cumulative

    # Generate the numbers 
#    cdef int *gen = <int *> malloc(num * sizeof(int))
    cdef int gen[num]

    for i in range(num):
        r = rand() / (RAND_MAX * 1.0)

        for j in range(m):
            if cp[j] <= r < cp[j+1]:
                gen[i] = j
    free(cp)
    return gen

cpdef binomial_dist(double[:,:] param, const int num_samples, const int num_trials):
    cdef double t1, f1, t2, f2
    cdef int i, j, count1=0, count2=0, count12=0
    cdef double c0110, c01, c10
    cdef double p_singles[4]
    cdef double p_super[2]
    cdef double g[num_trials][4]
    cdef int single_output[num_samples]
    cdef int super_output[num_samples]

#    param = param_list[0]

    sample = []
    for i in range(num_trials):
        for j in range(3):
            count1 = 0
            count2 = 0
            count12 = 0
            t1 = param[j][0]
            f1 = param[j][1]
            t2 = param[(j+1)%3][0]
            f2 = param[(j+1)%3][1]

            p_singles[0] = p00(t1,t2)
            p_singles[1] = p01(t1,t2)
            p_singles[2] = p10(t1,t2)
            p_singles[3] = p11(t1,t2)
            p_super[0] = p0110(t1,f1,t2,f2)
            p_super[1] = 1-p0110(t1,f1,t2,f2)

            single_output = binomial(p_singles, 4, num_samples)
            super_output = binomial(p_super, 2, num_samples)
            for k in range(num_samples):
                if single_output[k] == 1:
                    count1 += 1
                if single_output[k] == 2:
                    count2 += 1
                if super_output[k] == 0:
                    count12 += 1

            g[i][j] = G(count12, count1, count2)


        g[i][3] = F(g[i])

#    ga = np.array(g)
#
#    all_cfs = []
#    st = time.time()
#    for k in range(4):
#        data = ga[:,k]
#        data.sort()
#        ca = data[int(0.005 * len(data))]
#        cb = data[int(0.995 * len(data))]
#
#        all_cfs.append([ca,cb])
#    print(f'CF finished in {time.time()-st} secs')

    return g


