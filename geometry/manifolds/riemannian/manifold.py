#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

from geometry.setup import *

#%% Riemannian Manifold

class RiemannianManifold(ABC):
    def __init__(self,
                 G:Callable[[Array], Array]=None,
                 f:Callable[[Array], Array]=None,
                 invf:Callable[[Array],Array]=None,
                 )->None:
        
        self.f = f
        self.invf = invf
        if ((G is None) and (f is None)):
            raise ValueError("Both the metric, g, and chart, f, is not defined")
        elif (G is None):
            self.G = lambda z: self.pull_back_metric(z)
        else:
            self.G = G
            
        return
        
    def __str__(self)->str:
        
        return "Riemannian Manifold base object"
    
    def Jf(self,
           z:Array
           )->Array:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            return jacfwd(self.f)(z)
        
    def pull_back_metric(self,
                         z:Array
                         )->Array:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            Jf = self.Jf(z)
            return jnp.einsum('ik,il->kl', Jf, Jf)
    
    def DG(self,
           z:Array
           )->Array:

        return jacfwd(self.G)(z)
    
    def Ginv(self,
             z:Array
             )->Array:
        
        return jnp.linalg.inv(self.G(z))
    
    def christoffel_symbols(self,
                            z:Array
                            )->Array:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def geodesic_equation(self,
                          z:Array,
                          v:Array
                          )->Array:
        
        Gamma = self.Chris(z)

        dx1t = v
        dx2t = -jnp.einsum('ikl,k,l->i',Gamma,v,v)
        
        return jnp.hstack((dx1t,dx2t))
    
    def energy(self, 
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.G(g))(gamma)
        integrand = jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma)
        
        return jnp.trapz(integrand, dx=dt)
    
    def length(self,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.G(g))(gamma)
        integrand = jnp.sqrt(jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma))
            
        return jnp.trapz(integrand, dx=dt)
    
#%% Stochastic Riemannian Manifold

class StochasticRiemannianManifold(ABC):
    def __init__(self,
                 G:Callable[[Array], Array]=None,
                 f:Callable[[Array], Array]=None,
                 invf:Callable[[Array],Array]=None,
                 seed:int=2712,
                 )->None:
        
        self.f = f
        self.invf = invf
        if ((G is None) and (f is None)):
            raise ValueError("Both the metric, g, and chart, f, is not defined")
        elif (G is None):
            self.G = lambda z, N_samples: self.pull_back_metric(z, N_samples)
        else:
            self.G = G
            
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
    
    def DG(self,
           z:Array,
           N_samples:int=1,
           )->Array:

        return jacfwd(self.G, argnums=1)(z,N_samples)
    
    def Ginv(self,
             z:Array,
             N_samples:int=1,
             )->Array:
        
        return vmap(jnp.linalg.inv, in_axes=(0,None))(self.G(z,N_samples))
    
    def christoffel_symbols(self,
                            z:Array,
                            N_samples:int=1,
                            )->Array:
        
        Dgx = self.DG(z, N_samples)
        gsharpx = self.Ginv(z, N_samples)
        
        return 0.5*(jnp.einsum('tim,tkml->tikl',gsharpx,Dgx)
                   +jnp.einsum('tim,tlmk->tikl',gsharpx,Dgx)
                   -jnp.einsum('tim,tklm->tikl',gsharpx,Dgx))
    
    def geodesic_equation(self,
                          z:Array,
                          v:Array,
                          N_samples:int=1,
                          )->Array:
        
        Gamma = self.Chris(z, N_samples)

        dx1t = v
        dx2t = -jnp.einsum('ikl,k,l->i',Gamma,v,v)
        
        return jnp.hstack((dx1t,dx2t))
    
    def energy(self, 
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.G(g))(gamma)
        integrand = jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma)
        
        return jnp.trapz(integrand, dx=dt)
    
    def length(self,
               gamma:Array,
               )->Array:
        
        T = len(gamma)-1
        dt = 1.0/T
        
        dgamma = (gamma[1:]-gamma[:-1])*T
        
        g = vmap(lambda g: self.G(g))(gamma)
        integrand = jnp.sqrt(jnp.einsum('ti,tij,tj->t', dgamma, g[:-1], dgamma))
            
        return jnp.trapz(integrand, dx=dt)
    