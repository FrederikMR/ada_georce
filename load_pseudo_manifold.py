#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:42 2024

@author: fmry
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
from jax import jit

import haiku as hk

from geometry.pseudo_riemannian.manifolds import LatentSpaceManifold

from vae.model_loader import mnist_generator, svhn_generator, celeba_generator, load_model

from vae.models import mnist_encoder
from vae.models import mnist_decoder
from vae.models import mnist_vae

from vae.models import svhn_encoder
from vae.models import svhn_decoder
from vae.models import svhn_vae

from vae.models import celeba_encoder
from vae.models import celeba_decoder
from vae.models import celeba_vae

#%% Load manifolds

def load_manifold(manifold:str="Euclidean",
                  dim:int = 2,
                  svhn_path:str = "../../../Data/SVHN/",
                  celeba_path:str = "../../../Data/CelebA/",
                  ):
   
    if manifold == "celeba":
        celeba_state = load_model(''.join(('models/', f'celeba_{dim}/')))
        celeba_dataloader = celeba_generator(data_dir=celeba_path,
                                             batch_size=64,
                                             seed=2712,
                                             split=0.8,
                                             )
        @hk.transform
        def celeba_tvae(x):

            vae = celeba_vae(
                        encoder=celeba_encoder(latent_dim=dim),
                        decoder=celeba_decoder(),
            )
         
            return vae(x)
       
        @hk.transform
        def celeba_tencoder(x):
       
            encoder = celeba_encoder(latent_dim=32)
       
            return encoder(x)[0]
       
        @hk.transform
        def celeba_tdecoder(x):
       
            decoder = celeba_decoder()
       
            return decoder(x)
       
        celeba_encoder_fun = jit(lambda x: celeba_tencoder.apply(celeba_state.params,
                                                                 None,
                                                                 x.reshape(-1,64,64,3)
                                                                 )[0].reshape(-1,dim).squeeze())
        celeba_decoder_fun = jit(lambda x: celeba_tdecoder.apply(celeba_state.params,
                                                                 None,
                                                                 x.reshape(-1,dim)
                                                                 ).reshape(-1,64*64*3).squeeze())
       
        M = LatentSpaceManifold(dim=64*64*3,
                                emb_dim=dim,
                                encoder=celeba_encoder_fun,
                                decoder=celeba_decoder_fun,
                                )
       
       
        celeba_data = next(celeba_dataloader).x
       
        z0, zT = celeba_data[0], celeba_data[1]
       
        return jnp.array(z0).reshape(-1), jnp.array(zT).reshape(-1), M
   
    elif manifold == "svhn":
        svhn_state = load_model(''.join(('models/', f'svhn_{dim}/')))
        svhn_dataloader = svhn_generator(data_dir=svhn_path,
                                         batch_size=64,
                                         seed=2712,
                                         split='train[:80%]',
                                         )
        @hk.transform
        def svhn_tvae(x):

            vae = svhn_vae(
                        encoder=svhn_encoder(latent_dim=dim),
                        decoder=svhn_decoder(),
            )
         
            return vae(x)
       
        @hk.transform
        def svhn_tencoder(x):
       
            encoder = svhn_encoder(latent_dim=dim)
       
            return encoder(x)[0]
       
        @hk.transform
        def svhn_tdecoder(x):
       
            decoder = svhn_decoder()
       
            return decoder(x)
       
        svhn_encoder_fun = jit(lambda x: svhn_tencoder.apply(svhn_state.params,
                                                             None,
                                                             x.reshape(-1,32,32,3)
                                                             )[0].reshape(-1,dim).squeeze())
        svhn_decoder_fun = jit(lambda x: svhn_tdecoder.apply(svhn_state.params,
                                                             None,
                                                             x.reshape(-1,dim)
                                                             ).reshape(-1,32*32*3).squeeze())
       
        M = LatentSpaceManifold(dim=32*32*3,
                                emb_dim=dim,
                                encoder=svhn_encoder_fun,
                                decoder=svhn_decoder_fun,
                                )
       
       
        svhn_data = next(svhn_dataloader).x
       
        z0, zT = svhn_data[0], svhn_data[1]
       
        return jnp.array(z0).reshape(-1), jnp.array(zT).reshape(-1), M

    elif manifold == "mnist":
        mnist_state = load_model(''.join(('models/', f'mnist_{dim}/')))
        mnist_dataloader = mnist_generator(seed=2712,
                                           batch_size=64,
                                           split='train[:80%]')
       
        @hk.transform
        def mnist_tvae(x):
       
            vae = mnist_vae(
                        encoder=mnist_encoder(latent_dim=dim),
                        decoder=mnist_decoder(),
            )
       
            return vae(x)
       
        @hk.transform
        def mnist_tencoder(x):
       
            encoder = mnist_encoder(latent_dim=dim)
       
            return encoder(x)[0]
       
        @hk.transform
        def mnist_tdecoder(x):
       
            decoder = mnist_decoder()
       
            return decoder(x)
       
        mnist_encoder_fun = jit(lambda x: mnist_tencoder.apply(mnist_state.params,
                                                               None,
                                                               x.reshape(-1,28,28,1)
                                                               )[0].reshape(-1,dim).squeeze())
        mnist_decoder_fun = jit(lambda x: mnist_tdecoder.apply(mnist_state.params,
                                                               None,
                                                               x.reshape(-1,dim)
                                                               ).reshape(-1,28*28).squeeze())
       
        M = LatentSpaceManifold(dim=28*28,
                                emb_dim=dim,
                                encoder=mnist_encoder_fun,
                                decoder=mnist_decoder_fun,
                                )
       
       
        mnist_data = next(mnist_dataloader).x
       
        z0, zT = mnist_data[0], mnist_data[1]
       
        return jnp.array(z0).reshape(-1), jnp.array(zT).reshape(-1), M
       
    else:
        raise ValueError(f"Manifold, {manifold}, is not defined. Only suported is: \n\t-Euclidean\n\t-Paraboloid\n\t-Sphere")
       
    return z0, zT, M