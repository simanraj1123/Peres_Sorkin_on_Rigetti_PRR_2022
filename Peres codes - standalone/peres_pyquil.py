import numpy as np, matplotlib.pyplot as plt, random, time, datetime
from functools import lru_cache
from pyquil import Program, get_qc
from pyquil.gates import *
import os, sys
from pyquil.quilatom import quil_sin, quil_cos, Parameter
from pyquil.quilbase import DefGate
from pyquil.latex import display, to_latex
# import Peres_helpers as hf
import pickle
from collections import Counter
from scipy.optimize import curve_fit as cf
sys.path.append('binomial_cython')
from binomial import binomial_dist

Z_SCORE = 3
N_SHOTS = 10_000

# Gammas for different pairs of states. This function returns the click-counts for the different configs.
def g(u):
    '''
    Calls the sigma function with different values of parameters correponding to the configurations, |ψ12>, |ψ1> and |ψ2>. Returns a
    dictionary with configurations as keys and output as values (which are lists).
    '''
    params = list(zip(*u)) # Unpack parameters
    theta, phi = params[0], params[1] # Store thetas and phis in seperate tuples.
    
    s12 = qc.run(exe, memory_map={'theta': theta, 'phi': phi}) # Stores the output of the circuit run.
    counts_s12 = Counter([''.join(list(map(str, elem))) for elem in s12])
    
    return {'Clicks': s12, 'Counts': counts_s12}

# Computing all the three gammas. This function returns the click-counts for all the configs.
def f(u):
	'''
	Calls the g function to run the circuit for different configurations and returns a dictionary with 'a', 'b', 'c' as keys and the corresponding 
	outputs of the three configurations. This marks the end of what the Quantum computer must be used for. After this it is all about post-
	processing the data.
	'''
	alpha = g([u[0], u[1]]) # Running for alpha
	beta = g([u[1], u[2]]) # Running for beta
	gamma = g([u[2], u[0]]) # Running for gamma

	res = {'a': alpha, 'b': beta, 'c': gamma}

	return res


# The circuit for constructing the product state and measuring in Bell basis.
def circuit_bell(qubit1, qubit2):
    circ = Program()
    
    c = circ.declare('ro', 'BIT', 2)
    theta = circ.declare('theta', 'REAL', 2)
#     thetam = circ.declare('thetam', 'REAL', 1)
    phi = circ.declare('phi', 'REAL', 2)
    
    # Preparation of states.
    circ += RY(theta[0], qubit1)
    circ += RZ(phi[0], qubit1)
    
    circ += RY(theta[1], qubit2)
    circ += RZ(phi[1], qubit2)
    
    # Measuring in psi+ basis
    # REMOVED FOR TEST OF SIMPLER CIRCUIT
    circ += CNOT(qubit1, qubit2)
#     circ += RY(qubit1, thetam)
    circ += H(qubit1)

    circ += MEASURE(qubit1, c[0])
    circ += MEASURE(qubit2, c[1])
    
    circ.wrap_in_numshots_loop(N_SHOTS)
    
    return circ

# The circuit for constructing the product state and measuring in Computational basis.
def circuit_comp(qubit1, qubit2):
    circ = Program()
    
    c = circ.declare('ro', 'BIT', 2)
    theta = circ.declare('theta', 'REAL', 2)
    phi = circ.declare('phi', 'REAL', 2)
    
    # Preparation of states.
    circ += RY(theta[0], qubit1)
    circ += RZ(phi[0], qubit1)
    
    circ += RY(theta[1], qubit2)
    circ += RZ(phi[1], qubit2)
    
    circ += MEASURE(qubit1, c[0])
    circ += MEASURE(qubit2, c[1])
    
    circ.wrap_in_numshots_loop(N_SHOTS)
    
    return circ

# Declaring variables to store quantum circuit and the executable.
qc = None
exe = None
def run_peres(q1, q2, trial, engine, states):
    global qc
    global exe
    result_list = []
    print(f'Engine requested: {engine}')
    if engine == 'qvm':
        qc = get_qc('Aspen-9', as_qvm=True) # Initialise QPU.
        
    elif engine == 'Aspen':
        qc = get_qc('Aspen-9')
        
    elif engine == 'noisy-Aspen':
        qc = get_qc('Aspen-9', as_qvm=True, noisy=True)
        
    else:
        qc = get_qc('2q-qvm')
    # qc = get_qc('2q-qvm')

    circ = circuit_bell(q1,q2)
    exe = qc.compile(circ)
    
    print('Running Bell-state measurements')
    for i in range(len(states)):
        data = states[i]
#         if engine == 'Aspen':
#             data['Device_details'] = qc.device.get_isa()

        data[f'Counts_bell_{engine}'] = f(data['State_params'])

#         states.append(data)

        print(f'Done with iteration {i}', end='\r')
    
    print('\n')
    print('Running Computational measurements')
    circ = circuit_comp(q1,q2)
#     global exe
    exe = qc.compile(circ)
    i=0
    for i in range(len(states)):
        data = states[i]
        data[f'Counts_comp_{engine}'] = f(data['State_params'])
        print(f'Done with iteration {i}', end='\r')
        i += 1
    
#     for data in result_list:
#         data['Gamma'] = get_gammas(data['Counts_bell'], data['Counts_comp'])

#         data = gamma_theory(data)
    
    folder = f'product_peres_{engine}_{datetime.date.today()}_{q1}_{q2}_bits_{N_SHOTS}_shots_trial_{trial}'
    print(f'Creating folder {folder}')
    os.system(f'mkdir -p {folder}')
    with open(f'{folder}/result_list_trial_{trial}', 'wb') as file:
        pickle.dump(states, file)
    print(f'Results saved in file {folder}/result_list_trial_{trial}')
    
    print()
    print('Completed.')
    return states

def get_good_qbits(cz_fid, meas_fid):
    qc = get_qc('Aspen-9')
    details = qc.device.get_isa()

    good_edges = []
    for edge in details.edges:
        gates = edge.gates
        for gate in gates:
            if gate.operator == 'CZ':
                try:
                    if gate.fidelity > cz_fid:
                        good_edges.append(edge)
                except TypeError:
                    continue
    good_qubits = {}
    for edge in good_edges[:]:
        q1, q2 = edge.targets
        for q in details.qubits:
            if q.id in [q1,q2]:
                gates = q.gates
                for gate in gates:
                    if gate.operator == 'MEASURE':
                        if gate.fidelity < meas_fid:
                            try:
                                good_edges.remove(edge)
                            except ValueError:
#                                 print('Edge already removed')
                                continue
    good_qubits = []
    for edge in good_edges:
        q1, q2 = edge.targets
        good_qubits.append({'Edge': edge})
        for q in details.qubits:
            if q.id == q1:
                good_qubits[-1]['Qubit1'] = q
            elif q.id == q2:
                good_qubits[-1]['Qubit2'] = q
    return good_qubits