#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:22:08 2024

@author: fmry
"""

#%% Modules

import numpy as np

import os

import time

#%% Submit job

def submit_job():
    
    os.system("bsub < submit_runtime.sh")
    
    return

#%% Generate jobs

def generate_job(manifold, d, method, geometry, batch_size):

    with open ('submit_runtime.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpua100
    #BSUB -J {method}_{geometry[0]}{manifold}{d}_{batch_size}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "rusage[mem=10GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 
    
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    module swap python3/3.10.12
    
    python3 runtime.py \\
        --manifold {manifold} \\
        --geometry {geometry} \\
        --dim {d} \\
        --T 100 \\
        --method {method} \\
        --batch_size {batch_size}
        --tol 1e-4 \\
        --max_iter 1000 \\
        --number_repeats 5 \\
        --timing_repeats 5 \\
        --seed 2712 \\
        --save_path timing_gpu/
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    geomtries = ['Riemannian']
    scipy_methods = []#["BFGS", 'CG', 'dogleg', 'trust-ncg', 'trust-exact']
    jax_methods = ["ADAM", "SGD"]
    methods = ['AdaGEORCE', 'RegGEORCE', 'GEORCE', 'init', 'ground_truth']
    methods += jax_methods + scipy_methods
    #sphere
    runs = {"Sphere": {100: [10, 50, 101], 
                       500: [100, 250, 501], 
                       1000: [100, 250, 1001],
                       5000: [100, 250, 5001],
                       10000: [100, 250, 10001],
                       },
            "Ellipsoid": {100: [10, 50, 101], 
                               500: [100, 250, 501], 
                               1000: [100, 250, 1001],
                               5000: [100, 250, 5001],
                               10000: [100, 250, 10001],
                               },
            }
    
    for geo in geomtries:
        for man,vals in runs.items():
            for d, batches in vals.items():
                for batch in batches:
                    for m in methods:
                        time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                        generate_job(man, d, m, geo, batch)
                        try:
                            submit_job()
                        except:
                            time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                            try:
                                submit_job()
                            except:
                                print(f"Job script with {geo}, {man}, {m}, {d}, {batch} failed!")

#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)