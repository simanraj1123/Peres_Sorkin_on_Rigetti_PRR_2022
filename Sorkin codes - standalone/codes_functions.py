# Importing essential libraries
import numpy as np, matplotlib.pyplot as plt, random, time
from functools import lru_cache
from pyquil import Program, get_qc
from pyquil.gates import *
import os
from pyquil.quilatom import quil_sin, quil_cos, Parameter
from pyquil.quilbase import DefGate
from pyquil.latex import display, to_latex
# import Peres_helpers as hf
import pickle
from collections import Counter
import time

# Number of shots and z-score for confidence interval
N_SHOTS = 10_000
Z_SCORE = 3

# Generating random angles
e=0
def params_complex():
	'''
	Generates parameters to prepare COMPLEX quantum states.
    
    Returns:
        A list of three tuples. The first element of each tuple is the value of theta 
        and the second element is the value of phi.
    
	'''
	theta = np.arccos(np.cos(e) - 2 * np.array([random.uniform(0,1) for _ in range(2)]))
	phi = np.array([2*np.pi*random.uniform(0,1) for _ in range(2)])
	params = zip(theta, phi)
	return list(params)

# Combined circuit for creating and projecting 3-level states
def circuit_123(a,b):
    circ = Program()
    
    theta = circ.declare('theta', 'REAL', 2)
    phi = circ.declare('phi', 'REAL', 2)
    t = circ.declare('t', 'REAL', 2)
    c = circ.declare('ro', 'BIT', 2)
    
    circ += RY(theta[0], a)
    circ += RZ(phi[0], a)
    
    circ += RY(theta[1], b)
    circ += RZ(phi[1]/2, b)
    
    circ += CNOT(a, b)
    
    circ += RZ(-phi[1]/2, b)
    circ += RY((t[1]-theta[1])/2, b)
    
    circ += CNOT(a,b)
    
    circ += RY((-theta[1] - t[1])/2, b)
    
    circ += RY(-t[0], a)
    
    circ += MEASURE(a, c[0])
    circ += MEASURE(b, c[1])
    
    circ.wrap_in_numshots_loop(N_SHOTS)
    
    return circ

# The theoretical probabilities and kappa
def theoretical_probs(u):
    theta, phi = [], []
    for i in range(len(u)):
        theta.append(u[i][0])
        phi.append(u[i][1])
        
    probs = {
        'P_123': (1/3) * np.abs(np.cos(theta[0]/2)*np.exp(-1j*phi[0]/2) + np.sin(theta[0]/2)*np.cos(theta[1]/2)*np.exp(-1j*(phi[1]-phi[0])/2) + np.sin(theta[0]/2)*np.sin(theta[1]/2)*np.exp(1j*(phi[1]+phi[0])/2))**2, 
        'P_12': (1/2) * np.abs(np.cos(theta[0]/2)*np.exp(-1j*phi[0]/2) + np.sin(theta[0]/2)*np.cos(theta[1]/2)*np.exp(-1j*(phi[1]-phi[0])/2))**2, 
        'P_23': (1/2) * np.abs(np.sin(theta[0]/2)*np.cos(theta[1]/2)*np.exp(-1j*(phi[1]-phi[0])/2) + np.sin(theta[0]/2)*np.sin(theta[1]/2)*np.exp(1j*(phi[1]+phi[0])/2))**2, 
        'P_31': (1/2) * np.abs(np.cos(theta[0]/2)*np.exp(-1j*phi[0]/2) + np.sin(theta[0]/2)*np.sin(theta[1]/2)*np.exp(1j*(phi[1]+phi[0])/2))**2, 
        'P_1': np.cos(theta[0]/2)**2,
        'P_2': np.sin(theta[0]/2)**2 * np.cos(theta[1]/2)**2,
        'P_3': np.sin(theta[0]/2)**2 * np.sin(theta[1]/2)**2,
    }
    
    probs['Kappa'] = 3 * probs['P_123'] - 2 * (probs['P_12'] + probs['P_23'] + probs['P_31']) + probs['P_1'] + probs['P_2'] + probs['P_3']
    
    return probs

# Getting the theoretical fluctuations with 1000 repetitions of the same circuit
ci_a = 0.005
ci_b = 1 - ci_a
def get_ci(states):
    ci_list = []
    v_list = []
    for state in states:
        probs = theoretical_probs(state)
        k_list = []
        for i in range(1000):
            c123 = Counter(random.choices(['0','1'], weights=[probs['P_123'], 1-probs['P_123']], k=N_SHOTS))['0']
            c12 = Counter(random.choices(['0','1'], weights=[probs['P_12'], 1-probs['P_12']], k=N_SHOTS))['0']
            c23 = Counter(random.choices(['0','1'], weights=[probs['P_23'], 1-probs['P_23']], k=N_SHOTS))['0']
            c31 = Counter(random.choices(['0','1'], weights=[probs['P_31'], 1-probs['P_31']], k=N_SHOTS))['0']
            csingles = Counter(random.choices(['0','1','2'], weights=[probs['P_1'], probs['P_2'], probs['P_3']], k=N_SHOTS))

            k = 3*c123 - 2*(c12+c23+c31) + csingles['0'] + csingles['1'] + csingles['2']
            k_list.append(k/10_000)
        
        k_list.sort()
        ci = [k_list[int(ci_a*len(k_list)) - 1], k_list[int(ci_b*len(k_list)) - 1]]
        # ci = [k_list[int(0.00*len(k_list))], k_list[int(1*(len(k_list)-1))]]
        ci_list.append(ci)
        v_list.append(max(k_list))

    return ci_list