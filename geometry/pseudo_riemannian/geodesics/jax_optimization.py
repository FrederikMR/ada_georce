#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:48:55 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.pseudo_riemannian.manifolds import PseudoRiemannianManifold
from geometry.stochastic.manifolds import IndicatorManifold

#%% Gradient Descent Estimation of Geodesics

class JAXOptimization(ABC):
    def __init__(self,
                 M:PseudoRiemannianManifold,
                 intrinsic_batch_size:int=None,
                 extrinsic_batch_size:int=None,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 lr_rate:float=0.01,
                 optimizer:Callable=None,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-4,
                 seed:int=2712,
                 )->None:
        
        if extrinsic_batch_size is None:
            self.extrinsic_batch_size = M.emb_dim
        else:
            self.extrinsic_batch_size = extrinsic_batch_size
        
        if intrinsic_batch_size is None:
            self.intrinsic_batch_size = M.dim
        else:
            self.intrinsic_batch_size = intrinsic_batch_size
        
        self.M = M
        self.T = T
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        
        self.extrinsic_batch_size = extrinsic_batch_size
        self.intrinsic_batch_size = intrinsic_batch_size
        
        if optimizer is None:
            self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr_rate)
        else:
            self.opt_init, self.opt_update, self.get_params = optimizer(lr_rate)
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
            
        self.z0 = None
        self.zT = None
        
        return
    
    def __str__(self)->str:
        
        return "Geodesic Computation Object using JAX Optimizers"
    
    def energy(self, 
               zt:Array,
               *args,
               )->Array:
        
        def energy_path(energy:Array,
                        y:Tuple[Array],
                        )->Tuple[Array]:
            
            z1, z2 = y
            
            f1, f2 = self.PM.Sf(z1, extrinsic_batch), self.PM.Sf(z2, extrinsic_batch)
            
            dz = f2-f1

            energy += jnp.sum(dz**2)
            
            return (energy,)*2
        
        extrinsic_batch, intrinsic_batch = self.PM.random_batch()
        term1 = self.PM.Sf(zt[0], extrinsic_batch)-self.PM.Sf(self.z0, extrinsic_batch)
        energy_init = jnp.sum(term1**2)
        
        zt = zt.at[:,intrinsic_batch].set(lax.stop_gradient(zt[:,intrinsic_batch]))
        
        zt = jnp.vstack((zt, self.zT))
        
        energy, _ = lax.scan(energy_path,
                             init=energy_init,
                             xs=(zt[:-1], zt[1:]),
                             )
        
        return energy
    
    def energy2(self, 
                zt:Array,
                *args,
                )->Array:
        
        def energy_path(energy:Array,
                        y:Tuple[Array],
                        )->Tuple[Array]:
            
            z, dz = y
            
            SG = self.PM.SG(z, extrinsic_batch, intrinsic_batch)

            energy += jnp.einsum('i,ij,j->', dz, SG, dz)
            
            return (energy,)*2
        
        extrinsic_batch, intrinsic_batch = self.PM.random_batch()
        
        SG0 = self.PM.SG(self.z0, extrinsic_batch, intrinsic_batch)
        term1 = zt[0]-self.z0
        energy_init = jnp.einsum('i,ij,j->', term1, SG0, term1)
        
        zt = jnp.vstack((zt, self.zT))
        
        energy, _ = lax.scan(energy_path,
                             init=energy_init,
                             xs=(zt[:-1], zt[1:]-zt[:-1]),
                             )
        
        return energy
    
    def Denergy(self,
                zt:Array,
                )->Array:
        
        return grad(self.energy)(zt)
    
    def cond_fun(self, 
                 carry:Tuple[Array, Array, Array, int],
                 )->Array:
        
        zt, grad, opt_state, idx = carry

        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                   carry:Tuple[Array, Array, Array, int],
                   )->Array:
        
        zt, grad, opt_state, idx = carry
        
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)

        grad = self.Denergy(zt)
        
        return (zt, grad, opt_state, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array, Array],
                 idx:int,
                 )->Array:
        
        zt, opt_state = carry
        
        grad = self.Denergy(zt)
        opt_state = self.opt_update(idx, grad, opt_state)
        zt = self.get_params(opt_state)
        
        return ((zt, opt_state),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.PM = IndicatorManifold(self.M, 
                                    self.extrinsic_batch_size, 
                                    self.intrinsic_batch_size,
                                    self.seed)
        
        self.z0 = z0
        self.zT = zT
        
        zt = self.init_fun(z0,zT,self.T)
        
        opt_state = self.opt_init(zt)
        
        if step == "while":
            grad = self.Denergy(zt)
        
            zt, grad, _, idx = lax.while_loop(self.cond_fun, 
                                              self.while_step,
                                              init_val=(zt, grad, opt_state, 0)
                                              )
        
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
            _, val = lax.scan(self.for_step,
                              init=(zt, opt_state),
                              xs = jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            
            grad = vmap(self.Denergy)(zt)
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
        
        return zt, grad, idx