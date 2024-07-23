#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

from geometry.manifolds.riemannian.manifold import RiemannianManifold

#%% Gradient Descent Estimation of Geodesics

class GEORCE_SA(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 batch_size:int=None,
                 lr_rate:float=0.001,
                 beta1:float=0.9,
                 beta2:float=0.999,
                 eps:float=1e-8,
                 init_fun:Callable[[Array, Array, int], Array]=None,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 )->None:
        
        self.M = M
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        
        if batch_size is None:
            self.batch_size = max(M.emb_dim//10, 1)
        else:
            self.batch_size = batch_size
            
        self.scaling = self.M.emb_dim/self.batch_size
        self.lr_rate = lr_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*jnp.linspace(0.0,
                                                                   1.0,
                                                                   self.T,
                                                                   endpoint=False,
                                                                   dtype=z0.dtype)[1:].reshape(-1,1)+z0
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem"
    
    def inner_product(self,
                      zt:Array,
                      ut:Array,
                      batch:Array,
                      )->Array:
        
        Gt = vmap(self.M.G, in_axes=(0,None))(zt, self.batch_size)
        
        return jnp.sum(jnp.einsum('ti,tij,tj->t', ut, Gt, ut))
    
    def gt(self,
           zt:Array,
           ut:Array,
           batch:Array,
           )->Array:
        
        return grad(self.inner_product)(zt,ut,batch)
    
    def sample_estimates(self,
                         zt:Array,
                         ut:Array,
                         )->Tuple[Array]:
        
        SG0 = self.M.G(self.z0, self.batch_size)
        SG = vmap(self.M.G, in_axes=(0,None))(zt, self.batch_size)
        gt = self.gt(zt, ut[1:], batch)
        rg = jnp.sum(gt**2)
        
        return SG0, SG, gt, rg
    
    def cond_fun(self, 
                 carry:Tuple[Array,Array,Array, Array, int],
                 )->Array:
        
        zt, ut, SG, SG0, SGinv, gt, rg, sk, beta1, beta2, kappa, idx = carry

        norm_grad = jnp.linalg.norm(sk.reshape(-1))

        return (norm_grad>self.tol) & (idx < self.max_iter)
    
    def unconstrained_opt(self, gt:Array, gt_inv:Array)->Array:
        
        g_cumsum = jnp.cumsum(gt[::-1], axis=0)[::-1]
        ginv_sum = jnp.sum(gt_inv, axis=0)
        rhs = jnp.sum(jnp.einsum('tij,tj->ti', gt_inv[:-1], g_cumsum), axis=0)+2.0*self.diff
        #lhs = -jnp.linalg.inv(ginv_sum)
        #muT = jnp.einsum('ij,j->i', lhs, rhs)
        muT = -jnp.linalg.solve(ginv_sum, rhs)
        mut = jnp.vstack((muT+g_cumsum, muT))
        
        return mut
    
    def georce_step(self,
                    zt:Array,
                    ut:Array,
                    gt:Array,
                    gt_inv:Array,
                    )->Array:
        
        mut = self.unconstrained_opt(gt, gt_inv)
        
        #ut = -0.5*vmap(jnp.linalg.lstsq)(gt,mut)[0]

        ut = -0.5*jnp.einsum('tij,tj->ti', gt_inv, mut)
        zt = self.z0+jnp.cumsum(ut[:-1], axis=0)
        
        return zt, ut
    
    def while_step(self,
                     carry:Tuple[Array,Array,Array, Array, int],
                     )->Array:
        
        zt, ut, SG, SG0, SGinv, gt, rg, sk, beta1, beta2, kappa, idx = carry
        
        zt_hat, ut_hat = self.georce_step(zt, ut, gt, SGinv)
        
        zt_sk = zt_hat-zt
        ut_sk = ut_hat-ut
        
        sk = jnp.vstack((zt_sk, ut_sk))
        
        zt_hat = zt+kappa*zt_sk
        ut_hat = ut+kappa*ut_sk
        
        SG0_hat, SG_hat, gt_hat, rg_hat = self.sample_estimates(zt_hat, ut_hat)

        SG0_hat = (1.-self.beta1)*SG0_hat+self.beta1*SG0
        SG_hat = (1.-self.beta1)*SG_hat+self.beta1*SG
        SGinv_hat = jnp.vstack((jnp.linalg.inv(SG0_hat).reshape(1,self.M.dim,self.M.dim), 
                                vmap(lambda g: jnp.linalg.inv(g))(SG_hat)))
        gt_hat = (1.-self.beta1)*gt_hat+self.beta1*gt
        rg_hat = (1.-self.beta2)*rg_hat-self.beta2*rg
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2
        
        SG_hat /= (1.-beta1)
        gt_hat /= (1.-beta1)
        rg_hat /= (1.-beta2)
        
        kappa = jnp.min(jnp.array([self.lr_rate/(jnp.sqrt(1+rg_hat)+self.eps),
                                   1]))
        
        return (zt_hat, ut_hat, SG_hat, SG0_hat, SGinv_hat, gt_hat, rg_hat, sk, 
                beta1, beta2, kappa, idx+1)
    
    def for_step(self,
                 carry:Tuple[Array,Array],
                 idx:int,
                 )->Array:
        
        zt, ut, SG, SG0, SGinv, gt, rg, beta1, beta2, kappa = carry
        batch = self.batch_generator()
 
        zt_hat, ut_hat = self.georce_step(zt, ut, gt, SGinv, batch)
        
        zt_sk = zt_hat-zt
        ut_sk = ut_hat-ut
        
        zt_hat = zt+kappa*zt_sk
        ut_hat = ut+kappa*ut_sk
        
        SG0_hat, SG_hat, gt_hat, rg_hat = self.sample_estimates(zt_hat, ut_hat)

        SG0_hat = (1.-self.beta1)*SG0_hat+self.beta1*SG0
        SG_hat = (1.-self.beta1)*SG_hat+self.beta1*SG
        SGinv_hat = jnp.vstack((jnp.linalg.inv(SG0_hat).reshape(1,self.M.dim,self.M.dim), 
                                vmap(lambda g: jnp.linalg.inv(g))(SG_hat)))
        gt_hat = (1.-self.beta1)*gt_hat+self.beta1*gt
        rg_hat = (1.-self.beta2)*rg_hat-self.beta2*rg
        
        beta1 = beta1*self.beta1
        beta2 = beta2*self.beta2
        
        SG_hat /= (1.-beta1)
        gt_hat /= (1.-beta1)
        rg_hat /= (1.-beta2)
        
        kappa = jnp.min(jnp.array([self.lr_rate/(jnp.sqrt(1+rg_hat)+self.eps),
                                   1]))
        
        return ((zt_hat, ut_hat, SG_hat, SG0_hat, SGinv_hat, gt_hat, rg_hat, 
                beta1, beta2, kappa),)*2
    
    def __call__(self, 
                 z0:Array,
                 zT:Array,
                 step:str="while",
                 )->Array:
        
        dtype = z0.dtype
        
        zt = self.init_fun(z0,zT,self.T)

        self.z0 = z0
        self.zT = zT
        self.diff = zT-z0
        
        ut = jnp.ones((self.T, self.M.dim), dtype=dtype)*self.diff/self.T
        SG0, SG, gt, rg = self.sample_estimates(zt, ut)

        SGinv = jnp.vstack((jnp.linalg.inv(SG0).reshape(1,self.M.dim,self.M.dim), 
                            vmap(lambda g: jnp.linalg.inv(g))(SG)))
        
        if step == "while":
            sk = jnp.ones((2*self.T-1, self.M.dim))+self.tol
            zt, ut, SG, SG0, SGinv, gt, rg, sk, beta1, beta2, kappa, idx = lax.while_loop(self.cond_fun, 
                                                                                     self.while_step, 
                                                                                     init_val=(zt, ut, SG, SG0, SGinv, gt, rg, sk, self.beta1, self.beta2, self.lr_rate, 0))
            
            zt = jnp.vstack((z0, zt, zT))
        elif step == "for":
                
            _, val = lax.scan(self.for_step,
                              init=(zt, ut, SG, SG0, SGinv, gt, rg, self.beta1, self.beta2, self.lr_rate),
                              xs=jnp.ones(self.max_iter),
                              )
            
            zt = val[0]
            zt = vmap(lambda z: jnp.vstack((z0, z, zT)))(zt)
            sk = None
            idx = self.max_iter
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
        return zt, sk, idx

        