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
from pyquil.noise import add_decoherence_noise

Z_SCORE = 3
N_SHOTS = 10_000

# Amplitude damping functions
def damping_channel(damp_prob):
    """
    Generate the Kraus operators corresponding to an amplitude damping
    noise channel.

    :params float damp_prob: The one-step damping probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    damping_op = np.sqrt(damp_prob) * np.array([[0, 1],
                                                [0, 0]])

    residual_kraus = np.diag([1, np.sqrt(1-damp_prob)])
    return [residual_kraus, damping_op]

def append_kraus_to_gate(kraus_ops, g):
    """
    Follow a gate `g` by a Kraus map described by `kraus_ops`.

    :param list kraus_ops: The Kraus operators.
    :param numpy.ndarray g: The unitary gate.
    :return: A list of transformed Kraus operators.
    """
    return [kj.dot(g) for kj in kraus_ops]


def append_damping_to_gate(gate, damp_prob):
    """
    Generate the Kraus operators corresponding to a given unitary
    single qubit gate followed by an amplitude damping noise channel.

    :params np.ndarray|list gate: The 2x2 unitary gate matrix.
    :params float damp_prob: The one-step damping probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    return append_kraus_to_gate(damping_channel(damp_prob), gate)


# Dephasing noise model
def dephasing_kraus_map(p=.1):
    """
    Generate the Kraus operators corresponding to a dephasing channel.

    :params float p: The one-step dephasing probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    return [np.sqrt(1-p)*np.eye(2), np.sqrt(p)*np.diag([1, -1])]

def tensor_kraus_maps(k1, k2):
    """
    Generate the Kraus map corresponding to the composition
    of two maps on different qubits.

    :param list k1: The Kraus operators for the first qubit.
    :param list k2: The Kraus operators for the second qubit.
    :return: A list of tensored Kraus operators.
    """
    return [np.kron(k1j, k2l) for k1j in k1 for k2l in k2]


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
def circuit_bell(qubit1, qubit2, damping_per_I, p, pmeas_1, pmeas_2, t1):
    circ = Program()
    
    c = circ.declare('ro', 'BIT', 2)
    theta = circ.declare('theta', 'REAL', 2)
    phi = circ.declare('phi', 'REAL', 2)
    
    corrupted_CZ = append_kraus_to_gate(
    tensor_kraus_maps(
        dephasing_kraus_map(p),
        dephasing_kraus_map(p)
    ),
    np.diag([1, 1, 1, -1]))
    
    circ.define_noisy_gate("I", [qubit1], append_damping_to_gate(np.eye(2), damping_per_I))
    circ.define_noisy_gate("I", [qubit2], append_damping_to_gate(np.eye(2), damping_per_I))
    
    circ.define_noisy_gate("CZ", [qubit1, qubit2], corrupted_CZ)
    
    circ.define_noisy_readout(qubit1, p00=pmeas_1[0], p11=pmeas_1[1])
    circ.define_noisy_readout(qubit2, p00=pmeas_2[0], p11=pmeas_2[1])
    
    # Preparation of states.
    circ += RY(theta[0], qubit1)
    circ += RZ(phi[0], qubit1)
    
    circ += RY(theta[1], qubit2)
    circ += RZ(phi[1], qubit2)
    
#     Appending I's at the end to simulate amplitude damping.
#     circ += I(qubit1)
#     circ += I(qubit2)
    
    # Measuring in psi+ basis
    circ += CNOT(qubit1, qubit2)
    circ += H(qubit1)
    
    # Appending I's at the end to simulate amplitude damping.
#     circ += I(qubit1)
#     circ += I(qubit2)
    
#     circ = add_decoherence_noise(circ, T1=t1)
    
    

    circ += MEASURE(qubit1, c[0])
    circ += MEASURE(qubit2, c[1])
    
    
    
    
    circ.wrap_in_numshots_loop(N_SHOTS)
    
    return circ

# The circuit for constructing the product state and measuring in Computational basis.
def circuit_comp(qubit1, qubit2, damping_per_I, p, pmeas_1, pmeas_2, t1):
    circ = Program()
    corrupted_CZ = append_kraus_to_gate(
    tensor_kraus_maps(
        dephasing_kraus_map(p),
        dephasing_kraus_map(p)
    ),
    np.diag([1, 1, 1, -1]))
    
    circ.define_noisy_gate("I", [qubit1], append_damping_to_gate(np.eye(2), damping_per_I))
    circ.define_noisy_gate("I", [qubit2], append_damping_to_gate(np.eye(2), damping_per_I))
    
    circ.define_noisy_gate("CZ", [qubit1, qubit2], corrupted_CZ)
    
    circ.define_noisy_readout(qubit1, p00=pmeas_1[0], p11=pmeas_1[1])
    circ.define_noisy_readout(qubit2, p00=pmeas_2[0], p11=pmeas_2[1])
    
    c = circ.declare('ro', 'BIT', 2)
    theta = circ.declare('theta', 'REAL', 2)
    phi = circ.declare('phi', 'REAL', 2)
    
    # Preparation of states.
    circ += RY(theta[0], qubit1)
    circ += RZ(phi[0], qubit1)
    
    circ += RY(theta[1], qubit2)
    circ += RZ(phi[1], qubit2)
    
    # Appending I's at the end to simulate amplitude damping.
#     circ += I(qubit1)
#     circ += I(qubit2)
    
#     circ = add_decoherence_noise(circ, T1=t1)
    
    
    circ += MEASURE(qubit1, c[0])
    circ += MEASURE(qubit2, c[1])
    
    
    
    circ.wrap_in_numshots_loop(N_SHOTS)
    
    return circ

def add_noises(circ, qubit1, qubit2, damping_per_I, p, pmeas_1, pmeas_2, t1):
    corrupted_CZ = append_kraus_to_gate(
    tensor_kraus_maps(
        dephasing_kraus_map(p),
        dephasing_kraus_map(p)
    ),
    np.diag([1, 1, 1, -1]))
    
    circ.define_noisy_gate("I", [0], append_damping_to_gate(np.eye(2), damping_per_I))
    circ.define_noisy_gate("I", [1], append_damping_to_gate(np.eye(2), damping_per_I))
    
    circ.define_noisy_gate("CZ", [0, 1], corrupted_CZ)
    
    circ.define_noisy_readout(0, p00=pmeas_1[0], p11=pmeas_1[1])
    circ.define_noisy_readout(1, p00=pmeas_2[0], p11=pmeas_2[1])
    
    circ = add_decoherence_noise(circ, T1=t1)
    
    return circ

# Declaring variables to store quantum circuit and the executable.
qc = None
exe = None
def run_peres(q1, q2, trial, engine, states, damping_per_I, p, p00, p11, t1, t2):
    global qc
    global exe
    result_list = []
    print(f'Engine requested: {engine}')
    if engine == 'qvm':
        qc = get_qc('Aspen-9', as_qvm=True) # Initialise QPU.
    elif engine == 'Aspen':
        qc = get_qc('Aspen-9')
    else:
#         qc = get_qc('Aspen-9', as_qvm=True, noisy=True)
        qc = get_qc('2q-qvm')

    circ = circuit_bell(q1,q2, damping_per_I, p, p00, p11, t1)
#     circ_n = qc.compiler.quil_to_native_quil(circ)
#     noisy_circ = add_decoherence_noise(circ_n, T1=t1, T2=t2, ro_fidelity={0: p00[0], 1: p11[0]})
#     print(noisy_circ)
#     noisy_circ.wrap_in_numshots_loop(N_SHOTS)
#     circ = qc.compiler.quil_to_native_quil(circ)
#     circ = add_noises(circ, q1,q2, damping_per_I, p, p00, p11, t1)
#     circ = add_decoherence_noise(circ, T1=t1)
#     circ.wrap_in_numshots_loop(N_SHOTS)
#     qc.qam.gate_noise = [0.1,0,0]
    exe = qc.compile(circ)
    
    print('Running Bell-state measurements')
    for i in range(len(states)):
        data = states[i]
#         data = {}
#         data['State_params'] = params_complex() if specified_params == 'None' else specified_params[i]

        data[f'Counts_bell_{engine}'] = f(data['State_params'])

#         states.append(data)

        print(f'Done with iteration {i}', end='\r')
    
    print('\n')
    print('Running Computational measurements')
    circ = circuit_comp(q1,q2, damping_per_I, p, p00, p11, t1)
#     circ_n = qc.compiler.quil_to_native_quil(circ)
#     noisy_circ = add_decoherence_noise(circ_n, T1=t1, T2=t2, ro_fidelity={0: p00[0], 1: p11[0]})
#     noisy_circ.wrap_in_numshots_loop(N_SHOTS)
#     global exe
#     circ = qc.compiler.quil_to_native_quil(circ)
#     circ = add_noises(circ, q1,q2, damping_per_I, p, p00, p11, t1)
#     circ = add_decoherence_noise(circ, T1=t1)
#     circ.wrap_in_numshots_loop(N_SHOTS)
#     print(circ)
#     qc.qam.gate_noise = [0.1,0,0]
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