#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.riemannian.manifolds import RiemannianManifold
from geometry.stochastic.manifolds import IndicatorManifold

#%% Adaptive GEORCE

class AdaGEORCE(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 batch_size:int=None,
                 lr_rate:float=0.1,
                 beta1:float=0.5,
                 beta2:float=0.5,
                 T:int=100,
                 max_iter:int=1000,
                 tol:float=1e-1,
                 eps:float=1e-8,
                 eps_conv:float=0.1,
                 kappa_conv:float=0.99,
                 seed:int=2712,
                 )->None:
        
        if batch_size is None:
            self.batch_size = M.emb_dim
        else:
            self.batch_size = batch_size
        
        self.M = M
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
            
        self.lr_rate = lr_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.eps_conv = eps_conv
        self.kappa_conv = kappa_conv
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Adaptive GEORCE Geodesic Object"
    
    def sample_estimates(self,
                         zt:Array,
                         ut:Array,
                         )->Tuple[Array]:
        
        batch = self.PM.random_batch()
        SG0 = self.PM.SG(self.z0, batch)
        sgt, SG = self.sgt(zt, ut[1:], batch)
        rg = jnp.sum(sgt**2)
        SG_concat = jnp.vstack((SG0.reshape(-1, self.M.dim, self.M.dim),
                                SG))
        
        return SG_concat, sgt, rg
    
    def energy(self, 
               zt:Array,
               ut:Array,
               )->Array:
        
        SG, sgt, rg = self.sample_estimates(zt, ut)
        
        term1 = zt[0]-self.z0
        val1 = jnp.einsum('i,ij,j->', term1, SG[0], term1)
        
        term2 = zt[1:]-zt[:-1]
        val2 = jnp.einsum('ti,tij,tj->t', term2, SG[1:-1], term2)
        
        term3 = self.zT-zt[-1]
        val3 = jnp.einsum('i,ij,j->', term3, SG[-1], term3)
        
        return val1+jnp.sum(val2)+val3, (SG, sgt, rg)
    
    def Denergy(self,
                zt:Array,
                ut:Array,
                )->Array:
        
        return grad(self.energy, argnums=0, has_aux=True)(zt, ut)
    
    def inner_product(self,
                      zt:Array,
                      ut:Array,
                      batch:Array,
                      )->Array:
        
        SG = vmap(self.PM.SG, in_axes=(0,None))(zt, batch)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', ut, SG, ut)), SG
    
    def sgt(self,
            zt:Array,
            ut:Array,
            batch:Array,
            )->Array:
        
        return grad(self.inner_product, has_aux=True)(zt,ut,batch)
    
    def georce_step(self,
                    zt:Array,
                    ut:Array,
                    sgt:Array,
                    sgt_inv:Array,
                    )->Array:
        
        mut = self.update_scheme(sgt, sgt_inv)

        ut_hat = -0.5*jnp.einsum('tij,tj->ti', sgt_inv, mut)
        zt_hat = self.z0+jnp.cumsum(ut_hat[:-1], axis=0)
        
        return zt_hat-zt, ut_hat-ut
    
    def update_scheme(self, 
                      sgt:Array, 
                      sgt_inv:Array,
                      )->Array:
        
        sg_cumsum = jnp.cumsum(sgt[::-1], axis=0)[::-1]
        sginv_sum = jnp.sum(sgt_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', sgt_inv[:-1], sg_cumsum), axis=0)+2.0*self.diff

        muT = -jnp.linalg.solve(sginv_sum, rhs)
        mut = jnp.vstack((muT+sg_cumsum, muT))
        
        return mut
    
    def adaptive_default(self,
                         SG_k1:Array,
                         SG_k2:Array,
                         sgt_k1:Array,
                         sgt_k2:Array,
                         rg_k1:Array,
                         rg_k2:Array,
                         beta1:Array,
                         beta2:Array,
                         idx:int,
                         )->Tuple[Array, Array, Array, Array, Array, Array, Array,
                                  Array, Array, Array]:
    
        SG_k2 = (1.-self.beta1)*SG_k2+self.beta1*SG_k1
        sgt_k2 = (1.-self.beta1)*sgt_k2+self.beta1*sgt_k1
        rg_k2 = (1.-self.beta2)*rg_k2 +self.beta2*rg_k1
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2

        SG_hat = SG_k2/(1.-beta1)
        sgt_hat = sgt_k2/(1.-beta1)
        vt = rg_k2/(1.-beta2)
        
        kappa = jnp.min(jnp.array([self.lr_rate/(jnp.sqrt(1+vt)+self.eps),1]))
        
        return SG_k2, SG_hat, sgt_k2, sgt_hat, rg_k2, beta1, beta2, kappa
    
    def adaptive_convergence(self,
                             SG_k1:Array,
                             SG_k2:Array,
                             sgt_k1:Array,
                             sgt_k2:Array,
                             rg_k1:Array,
                             rg_k2:Array,
                             beta1:Array,
                             beta2:Array,
                             idx:int,
                             )->Tuple[Array, Array, Array, Array, Array, Array, Array,
                                      Array, Array, Array]:

        SG_k2 = SG_k2/(idx+1.)+SG_k1*idx/(idx+1.)
        sgt_k2 = sgt_k2/(idx+1.)+sgt_k1*idx/(idx+1.)
        rg_k2 = rg_k2/(idx+1.)+rg_k1*idx/(idx+1.)
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2
        
        kappa = jnp.min(jnp.array([self.lr_rate/(jnp.sqrt(1+rg_k2)+self.eps),1]))
        
        return SG_k2, SG_k2, sgt_k2, sgt_k2, rg_k2, beta1, beta2, kappa
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, SG, SG_hat, sgt, sgt_hat, rg, grad, beta1, beta2, kappa, idx = carry
        
        norm_grad = jnp.linalg.norm(grad.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt_k1, ut_k1, SG_k1, SG_hat, sgt_k1, sgt_hat, rg_k1, grad, \
            beta1, beta2, kappa, idx = carry
        
        zt_sk, ut_sk = self.georce_step(zt_k1,
                                        ut_k1,
                                        sgt_hat,
                                        vmap(jnp.linalg.inv)(SG_hat))
        
        zt_k2 = zt_k1+kappa*zt_sk
        ut_k2 = ut_k1+kappa*ut_sk
        sk = jnp.vstack((zt_sk, ut_sk))
        
        grad, (SG_k2, sgt_k2, rg_k2) = self.Denergy(zt_k2, ut_k2)
        
        SG_k2, SG_hat, sgt_k2, sgt_hat, rg_k2, beta1, beta2, kappa \
            = lax.cond((jnp.linalg.norm(sk.reshape(-1))<self.eps_conv) & (kappa>self.kappa_conv),
                       self.adaptive_convergence,
                       self.adaptive_default,
                       SG_k1,
                       SG_k2,
                       sgt_k1,
                       sgt_k2,
                       rg_k1,
                       rg_k2,
                       beta1,
                       beta2,
                       idx,
                       )
        
        return (zt_k2, ut_k2, SG_k2, SG_hat, sgt_k2, sgt_hat, rg_k2, grad,
                beta1, beta2, kappa, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        zt_k1, ut_k1, SG_k1, SG_hat, sgt_k1, sgt_hat, rg_k1, grad, \
            beta1, beta2, kappa = carry
        
        zt_sk, ut_sk = self.georce_step(zt_k1,
                                        ut_k1,
                                        sgt_hat,
                                        vmap(jnp.linalg.inv)(SG_hat))
        
        zt_k2 = zt_k1+kappa*zt_sk
        ut_k2 = ut_k1+kappa*ut_sk
        sk = jnp.vstack((zt_sk, ut_sk))
        
        grad, (SG_k2, sgt_k2, rg_k2) = self.Denergy(zt_k2, ut_k2)
        
        SG_k2, SG_hat, sgt_k2, sgt_hat, rg_k2, beta1, beta2, kappa \
            = lax.cond((jnp.linalg.norm(sk.reshape(-1))<self.eps_conv) & (kappa>self.kappa_conv),
                       self.adaptive_convergence,
                       self.adaptive_default,
                       SG_k1,
                       SG_k2,
                       sgt_k1,
                       sgt_k2,
                       rg_k1,
                       rg_k2,
                       beta1,
                       beta2,
                       idx,
                       )
        
        return ((zt_k2, ut_k2, SG_k2, SG_hat, sgt_k2, sgt_hat, rg_k2, grad,
                 beta1, beta2, kappa),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        self.PM = IndicatorManifold(self.M, self.batch_size, self.seed)
        
        self.z0 = z0
        self.zT = zT
        self.diff = zT-z0
        
        zt = self.init_fun(z0,zT,self.T)
        ut = jnp.ones((self.T, self.M.dim), dtype=z0.dtype)*self.diff/self.T
        grad, (SG, sgt, rg) = self.Denergy(zt, ut)
        
        if step == "while":
            zt, ut, SG, SG_hat, gt, gt_hat, rg, grad, \
                beta1, beta2, kappa, idx \
                    = lax.while_loop(self.cond_fun,
                                     self.while_step,
                                     init_val=(zt, 
                                               ut, 
                                               SG, 
                                               SG,
                                               sgt,
                                               sgt,
                                               rg, 
                                               grad,
                                               self.beta1, 
                                               self.beta2, 
                                               self.lr_rate, 
                                               0),
                                 )
            
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
                
            _, val = lax.scan(self.for_step,
                              init=(zt, 
                                    ut, 
                                    SG, 
                                    SG, 
                                    sgt, 
                                    sgt,
                                    rg, 
                                    grad,
                                    self.beta1, 
                                    self.beta2, 
                                    self.lr_rate),
                              xs=jnp.ones(self.max_iter),
                              )
            
            zt, grad = val[0], val[7]
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return zt, grad, idx

        