#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 01:25:57 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *
from geometry.riemannian.manifolds import RiemannianManifold
    
#%% Indicator Manifold
    
class IndicatorManifold(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 extrinsic_batch_size:int=None,
                 intrinsic_batch_size:int=None,
                 seed:int = 2712,
                 )->None:
        
        self.M = M
        if extrinsic_batch_size is None:
            self.extrinsic_batch_size = M.emb_dim
        else:
            self.extrinsic_batch_size = extrinsic_batch_size
        
        if intrinsic_batch_size is None:
            self.intrinsic_batch_size = M.dim
        else:
            self.intrinsic_batch_size = intrinsic_batch_size
            
        self.seed = seed
        self.key = jrandom.key(self.seed)
        
        if self.intrinsic_batch_size is None:
            self.scaling = self.M.emb_dim/extrinsic_batch_size
        else:
            self.scaling = (self.M.emb_dim/self.extrinsic_batch_size)*(self.M.dim/self.intrinsic_batch_size)
        self.SG = self.SG_pull_back
        
        self.extrinsic_batch = jnp.arange(0,self.M.emb_dim,1)
        self.instrinsic_batch = jnp.arange(0,self.M.dim,1)
            
        return
        
    def __str__(self)->str:
        
        return "Riemannian Manifold base object"
    
    def random_batch(self,
                     )->Array:
        
        self.key, subkey = jrandom.split(self.key)

        extrinsic_batch = jrandom.choice(subkey, 
                                         a=self.extrinsic_batch,
                                         shape=(self.extrinsic_batch_size,), 
                                         replace=False,
                                         )

        self.key, subkey = jrandom.split(self.key)

        intrinsic_batch = jrandom.choice(subkey, 
                                         a=self.instrinsic_batch,
                                         shape=(self.M.dim-self.intrinsic_batch_size,), 
                                         replace=False,
                                         )
            
        return extrinsic_batch, intrinsic_batch
    
    def Sf(self, 
           z:Array, 
           extrinsic_batch:Array,
           )->Array:
        
        return self.M.f(z)[extrinsic_batch]
            
    
    def SJ(self, 
           z:Array, 
           extrinsic_batch:Array,
           intrinsic_batch:Array=None,
           )->Array:

        z = z.at[intrinsic_batch].set(lax.stop_gradient(z[intrinsic_batch]))
        
        return jacfwd(self.Sf, argnums=0)(z, extrinsic_batch)
    
    def SG_pull_back(self, 
                     z:Array, 
                     extrinsic_batch:Array,
                     intrinsic_batch:Array,
                     )->Array:
        
        SJf = self.SJ(z, extrinsic_batch, intrinsic_batch)
        
        return self.scaling*jnp.einsum('ik,il->kl', SJf, SJf)
    
#%% Stochastic Riemannian Manifold

class StochasticRiemannianManifold(ABC):
    def __init__(self,
                 SG:Callable[[Array], Array]=None,
                 f:Callable[[Array], Array]=None,
                 invf:Callable[[Array],Array]=None,
                 seed:int=2712,
                 )->None:
        
        self.f = f
        self.invf = invf
        if ((SG is None) and (f is None)):
            raise ValueError("Both the metric, g, and chart, f, is not defined")
        elif (G is None):
            self.SG = lambda z, N_samples: self.pull_back_metric(z, N_samples)
        else:
            self.SG = G
            
        self.seed = seed
        self.key = jrandom.key(2712)
            
        return
        
    def __str__(self)->str:
        
        return "Riemannian Manifold base object"
    
    def Jf(self,
           z:Array,
           N_samples:int=1,
           )->Array:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            return jacfwd(self.f, argnums=0)(z, N_samples)
        
    def pull_back_metric(self,
                         z:Array,
                         N_samples:int=1,
                         )->Array:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            Jf = self.Jf(z, N_samples)
            return jnp.einsum('tik,til->tkl', Jf, Jf)
    
    def DSG(self,
            z:Array,
            N_samples:int=1,
            )->Array:

        return jacfwd(self.G, argnums=1)(z,N_samples)
    
    def EG(self,
           z:Array,
           N_samples:int=1,
           )->Array:
        
        return jnp.mean(self.G(z,N_samples), axis=0)
    
    def SGinv(self,
              z:Array,
              N_samples:int=1,
              )->Array:
        
        return vmap(jnp.linalg.inv, in_axes=(0,None))(self.G(z,N_samples))
    
    def christoffel_symbols(self,
                            z:Array,
                            N_samples:int=1,
                            )->Array:
        
        Dgx = self.DSG(z, N_samples)
        gsharpx = self.SGinv(z, N_samples)
        
        return 0.5*(jnp.einsum('tim,tkml->tikl',gsharpx,Dgx)
                   +jnp.einsum('tim,tlmk->tikl',gsharpx,Dgx)
                   -jnp.einsum('tim,tklm->tikl',gsharpx,Dgx))
    
    def energy(self, 
               gamma:Array,
               N_samples:int=1,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.EG(g,N_samples))(gamma)
        integrand = jnp.einsum('ti,tij,ktj->t', dgamma, g[:-1], dgamma)
        
        return jnp.trapz(integrand, dx=dt)
    
    def length(self,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.EG(g))(gamma)
        integrand = jnp.sqrt(jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma))
            
        return jnp.trapz(integrand, dx=dt)