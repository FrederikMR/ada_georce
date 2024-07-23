#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

####################

from .manifold import StochasticRiemannianManifold

#%% Code

class nSParaboloid(StochasticRiemannianManifold):
    def __init__(self,
                 dim:int=2,
                 seed:int=2712,
                 )->None:

        self.dim = dim
        self.emb_dim = dim+1
        super().__init__(f=self.f_standard, invf=self.invf_standard)
        
        self.seed = 2712
        self.key = jrandom.key(self.seed)
        
        return
    
    def __str__(self)->str:
        
        return f"Paraboloid of dimension {self.dim} equipped with the pull back metric"
    
    def f_standard(self,
                   z:Array,
                   N_samples:int=1,
                   )->Array:
        
        self.key, subkey = jrandom.split(self.key)
        
        coef = jrandom.normal(subkey, shape=(N_samples, 2))
        
        z = jnp.tile(z, (N_samples, 1))
        
        return jnp.hstack((z, jnp.sum(coef*z**2, axis=-1).reshape(-1,1)))

    def invf_standard(self,
                      x:Array,
                      )->Array:
        
        return x[:-1]
        