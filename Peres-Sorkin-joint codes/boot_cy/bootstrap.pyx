import pickle
import numpy as np, matplotlib.pyplot as plt
cimport numpy as np
from collections import Counter
# new compilation

'''
1. Import the data for each set of three states, i.e., res[i].
2. Collect all the single and super clicks of each pair of states (chosen from the set of three-states).
3. Convert the clicks in list form to strings using join function and then converting them to integers by using the int(x,2) function. Mapping this function over the list will be faster.
4. Convert the list of numbers to numpy arrays with fixed dtype. Now we are ready to enter the loop.
5.
'''
cdef bootstrap(long[:] clicks, const long size_clicks):
    return 0

cpdef bootclicks(long[:] clicks_a_bell, long [:]clicks_b_bell, long[:] clicks_c_bell, long[:] clicks_a_comp, long[:] clicks_b_comp, long[:] clicks_c_comp, const long size_clicks, const long num_trials):

    cdef int i
    cdef float ga, gb, gc, f
    cdef double all_gammas[num_trials][4]
    cdef double g[4]
    cdef long clicks_a_bell_boot[size_clicks]
    cdef long clicks_b_bell_boot[size_clicks]
    cdef long clicks_c_bell_boot[size_clicks]
    cdef long clicks_a_comp_boot[size_clicks]
    cdef long clicks_b_comp_boot[size_clicks]
    cdef long clicks_c_comp_boot[size_clicks]
#    cdef double counts_a_bell_boot, counts_b_bell_boot, counts_c_bell_boot, counts_a_comp_boot, counts_b_comp_boot, counts_c_comp_boot
    cdef int ca12, ca1, ca2, cb12, cb1, cb2, cc12, cc1, cc2
    
    
#    all_gammas = []
#
    for i in range(num_trials):
        clicks_a_bell_boot = np.random.choice(clicks_a_bell, size=size_clicks)
        clicks_b_bell_boot = np.random.choice(clicks_b_bell, size=size_clicks)
        clicks_c_bell_boot = np.random.choice(clicks_c_bell, size=size_clicks)
    
        clicks_a_comp_boot = np.random.choice(clicks_a_comp, size=size_clicks)
        clicks_b_comp_boot = np.random.choice(clicks_b_comp, size=size_clicks)
        clicks_c_comp_boot = np.random.choice(clicks_c_comp, size=size_clicks)
    
        counts_a_bell_boot = Counter(clicks_a_bell_boot)
        ca12 = counts_a_bell_boot[1]
        counts_b_bell_boot = Counter(clicks_b_bell_boot)
        cb12 = counts_b_bell_boot[1]
        counts_c_bell_boot = Counter(clicks_c_bell_boot)
        cc12 = counts_c_bell_boot[1]

    
        counts_a_comp_boot = Counter(clicks_a_comp_boot)
        ca1 = counts_a_comp_boot[1]
        ca2 = counts_a_comp_boot[2]
        counts_b_comp_boot = Counter(clicks_b_comp_boot)
        cb1 = counts_b_comp_boot[1]
        cb2 = counts_b_comp_boot[2]
        counts_c_comp_boot = Counter(clicks_c_comp_boot)
        cc1 = counts_c_comp_boot[1]
        cc2 = counts_c_comp_boot[2]

        g[0] = (2*ca12 - ca1 - ca2) / (2 * np.sqrt(ca1 * ca2))
        g[1] = (2*cb12 - cb1 - cb2) / (2 * np.sqrt(cb1 * cb2))
        g[2] = (2*cc12 - cc1 - cc2) / (2 * np.sqrt(cc1 * cc2))
    
        g[3] = g[0]**2 + g[1]**2 + g[2]**2 - 2 * g[0] * g[1] * g[2]
    
        all_gammas[i][0] = g[0]
        all_gammas[i][1] = g[1]
        all_gammas[i][2] = g[2]
        all_gammas[i][3] = g[3]


    return all_gammas
#
