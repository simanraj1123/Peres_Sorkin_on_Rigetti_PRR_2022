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
sys.path.append('boot_cy')
from bootstrap import bootclicks

Z_SCORE = 3
N_SHOTS = 10_000

# Generarating parameters theta and phi for randomly generating states on the Bloch-sphere.
def params_real():
	'''
	Generates parameters to prepare random REAL quantum states.
	'''
	theta = np.arccos(1 - 2 * np.array([random.uniform(0,1) for _ in range(3)]))
	phi = np.array([(np.pi)*random.randint(0,1) for _ in range(3)])
	params = zip(theta, phi)
	return list(params)
def params_complex():
	'''
	Generates parameters to prepare COMPLEX quantum states.
	'''
	theta = np.arccos(1 - 2 * np.array([random.uniform(0,1) for _ in range(3)]))
	phi = np.array([2*np.pi*random.uniform(0,1) for _ in range(3)])
	params = zip(theta, phi)
	return list(params)

# Calculate the gamma values and the F value.
def get_gammas(counts_bell, counts_comp):
    res = {}
    for gamma in counts_bell.keys():
        counts_12 = Counter([''.join(list(map(str, elem))) for elem in counts_bell[gamma]['Clicks']])['01']# CHANGED FROM 01 TO 00 FOR TEST OF SIMPLER CIRCUIT.
        counts_1 = Counter([''.join(list(map(str, elem))) for elem in counts_comp[gamma]['Clicks']])['01']# CHANGED FROM 01 TO 00 FOR TEST OF SIMPLER CIRCUIT
        counts_2 = Counter([''.join(list(map(str, elem))) for elem in counts_comp[gamma]['Clicks']])['10']
        #Counter([''.join(list(map(str, elem))) for elem in s2])
        g = (2*counts_12 - counts_1 - counts_2) / (2 * np.sqrt(counts_1*counts_2))
        res[gamma] = g

    res['F'] = res['a']**2 + res['b']**2 + res['c']**2 - 2 * res['a'] * res['b'] * res['c']
    return res

# Theoretical value of gammas and F.
def gamma_theory(data):
    u = data['State_params']
    g12 = np.cos(u[1][1] - u[0][1])
    g23 = np.cos(u[2][1] - u[1][1])
    g31 = np.cos(u[0][1] - u[2][1])
    
    data['Gammas_theory'] = {'a': g12, 'b': g23, 'c': g31}
    
    f = g12**2 + g23**2 + g31**2 - 2 * g12 * g23 * g31
    
    data['Gammas_theory']['F'] = f
    
    return data

def compute_gammas(result_list, engine):
    for data in result_list:
        data[f'Gamma_{engine}'] = get_gammas(data[f'Counts_bell_{engine}'], data[f'Counts_comp_{engine}'])

        data = gamma_theory(data)
        
    return result_list

def get_theory_cfs(result_list): 
    for res in result_list:
        counts = binomial_dist(np.array(res['State_params'], dtype=float), 10_000, 10_000)
        counts = np.array(counts)

        all_cfs = []

        for i in range(4):
            data = counts[:,i]
            data.sort()
            ca, cb = data[int(0.005 * len(data))], data[int(0.995 * len(data))]
            all_cfs.append([ca, cb])

        res['cfs_theory'] = {}
        res['cfs_theory']['a'] = all_cfs[0]
        res['cfs_theory']['b'] = all_cfs[1]
        res['cfs_theory']['c'] = all_cfs[2]
        res['cfs_theory']['F'] = all_cfs[3]
        
    return result_list

def plot_gammas(result_list):
    fig = plt.figure(figsize=(17, 10))
    all_a = []
    all_b = []
    all_c = []
    all_F = []

    all_a_theory = []
    all_b_theory = []
    all_c_theory = []
    all_F_theory = []

    all_a_errors = []
    all_b_errors = []
    all_c_errors = []
    all_F_errors = []

    for data in result_list:
        all_a.append(data['Gamma']['a'])
        all_b.append(data['Gamma']['b'])
        all_c.append(data['Gamma']['c'])
        all_F.append(data['Gamma']['F'])

        all_a_theory.append(data['Gammas_theory']['a'])
        all_b_theory.append(data['Gammas_theory']['b'])
        all_c_theory.append(data['Gammas_theory']['c'])
        all_F_theory.append(data['Gammas_theory']['F'])

    x = np.array(list(range(1,len(result_list)+1)))
#     fig = plt.figure(figsize=(17,10))

    plt.subplot(2,2,1)
    plt.plot(x, all_a, 'o', color='blue', markersize=4)
#         plt.plot(x, all_a_theory, '*', color='red', markersize=4)
#         plt.errorbar(x, m12, yerr=[m12 - cf12[:,0], cf12[:,1] - m12], fmt='.', marker='', capsize=4, color='red')
#         plt.errorbar(x, m12_as, yerr=[m12_as - cf12_as[:,0], cf12_as[:,1] - m12_as], fmt='.', marker='', capsize=4, color='blue')
    plt.axhline(y=1, ls='dashed', alpha=0.5)
    plt.axhline(y=-1, ls='dashed', alpha=0.5)
    plt.xticks(x)
    plt.xlabel('$\\gamma_{12}$', size=14)

    plt.subplot(2,2,2)
    plt.plot(x, all_b, 'o', color='blue', markersize=4)
#         plt.plot(x, all_b_theory, '*', color='blue', markersize=4)
#         plt.errorbar(x, m23, yerr=[m23 - cf23[:,0], cf23[:,1] - m23], fmt='.', marker='', capsize=4, color='red')
#         plt.errorbar(x, m23_as, yerr=[m23_as - cf23_as[:,0], cf23_as[:,1] - m23_as], fmt='.', marker='', capsize=4, color='blue')
    plt.axhline(y=1, ls='dashed', alpha=0.5)
    plt.axhline(y=-1, ls='dashed', alpha=0.5)
    plt.xticks(x)
    plt.xlabel('$\\gamma_{23}$', size=14)

    plt.subplot(2,2,3)
    plt.plot(x, all_c, 'o', color='blue', markersize=4)
#         plt.plot(x, all_c_theory, '*', color='darkgreen', markersize=4)
#         plt.errorbar(x, m31, yerr=[m31 - cf31[:,0], cf31[:,1] - m31], fmt='.', marker='', capsize=4, color='red')
#         plt.errorbar(x, m31_as, yerr=[m31_as - cf31_as[:,0], cf31_as[:,1] - m31_as], fmt='.', marker='', capsize=4, color='blue')
    plt.axhline(y=1, ls='dashed', alpha=0.5)
    plt.axhline(y=-1, ls='dashed', alpha=0.5)
    plt.xticks(x)
    plt.xlabel('$\\gamma_{31}$', size=14)

    plt.subplot(2,2,4)
    plt.plot(x, all_F, 'o', color='blue', markersize=4)
#         plt.plot(x, all_F_theory, '*', color='maroon', markersize=4)
#         plt.errorbar(x, mF, yerr=[mF - cfF[:,0], cfF[:,1] - mF], fmt='.', marker='', capsize=4, color='red')
#         plt.errorbar(x, mF_as, yerr=[mF_as - cfF_as[:,0], cfF_as[:,1] - mF_as], fmt='.', marker='', capsize=4, color='blue')
    plt.axhline(y=1, ls='dashed', alpha=0.5)
    plt.axhline(y=-1, ls='dashed', alpha=0.5)
    plt.xticks(x)
    plt.xlabel('$F$', size=14)
#         fit = lambda x, a, b: a*x + b
#         xdata = np.arange(1,len(result_list)+1)
#         ydata = [result_list[i]['Gamma']['F'] for i in range(len(result_list))]
#         popt, pcov = cf(fit, xdata, ydata)
#         plt.plot(xdata, fit(xdata, *popt), ls='dashed', alpha=0.5)
#         print(popt)

    return fig



def get_cfs_boot(res, engine):
    for res0 in res[:]:
        clicks_a_bell = res0[f'Counts_bell_{engine}']['a']['Clicks']
        clicks_a_bell = [''.join(list(map(str, elem))) for elem in clicks_a_bell]
        clicks_a_bell = np.array(list(map(int, clicks_a_bell, [2 for _ in range(len(clicks_a_bell))])), dtype=int)
    
        clicks_b_bell = res0[f'Counts_bell_{engine}']['b']['Clicks']
        clicks_b_bell = [''.join(list(map(str, elem))) for elem in clicks_b_bell]
        clicks_b_bell = np.array(list(map(int, clicks_b_bell, [2 for _ in range(len(clicks_b_bell))])), dtype=int)
    
        clicks_c_bell = res0[f'Counts_bell_{engine}']['c']['Clicks']
        clicks_c_bell = [''.join(list(map(str, elem))) for elem in clicks_c_bell]
        clicks_c_bell = np.array(list(map(int, clicks_c_bell, [2 for _ in range(len(clicks_c_bell))])), dtype=int)
    
        clicks_a_comp = res0[f'Counts_comp_{engine}']['a']['Clicks']
        clicks_a_comp = [''.join(list(map(str, elem))) for elem in clicks_a_comp]
        clicks_a_comp = np.array(list(map(int, clicks_a_comp, [2 for _ in range(len(clicks_a_comp))])), dtype=int)
    
        clicks_b_comp = res0[f'Counts_comp_{engine}']['b']['Clicks']
        clicks_b_comp = [''.join(list(map(str, elem))) for elem in clicks_b_comp]
        clicks_b_comp = np.array(list(map(int, clicks_b_comp, [2 for _ in range(len(clicks_b_comp))])), dtype=int)
    
        clicks_c_comp = res0[f'Counts_comp_{engine}']['c']['Clicks']
        clicks_c_comp = [''.join(list(map(str, elem))) for elem in clicks_c_comp]
        clicks_c_comp = np.array(list(map(int, clicks_c_comp, [2 for _ in range(len(clicks_c_comp))])), dtype=int)
    
        all_gammas = bootclicks(clicks_a_bell, clicks_b_bell, clicks_c_bell, clicks_a_comp, clicks_b_comp, clicks_c_comp, len(clicks_a_bell), 10000)
    
        all_gammas = np.array(all_gammas)
    
        all_ga = all_gammas[:,0]
        all_gb = all_gammas[:,1]
        all_gc = all_gammas[:,2]
        all_f = all_gammas[:,3]
    
        all_ga.sort()
        all_gb.sort()
        all_gc.sort()
        all_f.sort()
    
        ca1, ca2 = all_ga[int(0.005*len(all_ga))], all_ga[int(0.995*len(all_ga))]
        cb1, cb2 = all_gb[int(0.005*len(all_gb))], all_gb[int(0.995*len(all_gb))]
        cc1, cc2 = all_gc[int(0.005*len(all_gc))], all_gc[int(0.995*len(all_gc))]
        cf1, cf2 = all_f[int(0.005*len(all_f))], all_f[int(0.995*len(all_f))]
        
        res0[f'cfs_{engine}'] = {}
        res0[f'cfs_{engine}']['a'] = [ca1, ca2]
        res0[f'cfs_{engine}']['b'] = [cb1, cb2]
        res0[f'cfs_{engine}']['c'] = [cc1, cc2]
        res0[f'cfs_{engine}']['F'] = [cf1, cf2]
    
    return res