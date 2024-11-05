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
    #BSUB -q gpuv100
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
        --tol 0.0001 \\
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
    runs = {"mnist": {8: [100, 200, 28*28], 
                      16: [100, 200, 28*28],
                      32: [100, 200, 28*28],
                      64: [100, 200, 28*28],
                      128: [100, 200, 28*28],
                      256: [100, 200, 28*28],
                      512: [100, 200, 28*28],
                       },
            "svhn": {8: [100, 200, 32*32*3], 
                     16: [100, 200, 32*32*3],
                     32: [100, 200, 32*32*3],
                     64: [100, 200, 32*32*3],
                     128: [100, 200, 32*32*3],
                     256: [100, 200, 32*32*3],
                     512: [100, 200, 32*32*3],
                     },
            "celeba": {8: [100, 200, 64*64*3], 
                       16: [100, 200, 64*64*3],
                       32: [100, 200, 64*64*3],
                       64: [100, 200, 64*64*3],
                       128: [100, 200, 64*64*3],
                       256: [100, 200, 64*64*3],
                       512: [100, 200, 64*64*3],
                       },
            "Sphere": {500: [100, 200, 501], 
                       1000: [100, 200, 1001],
                       5000: [100, 200, 5001],
                       10000: [100, 200, 10001],
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