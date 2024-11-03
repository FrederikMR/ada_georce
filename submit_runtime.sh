    #! /bin/bash
    #BSUB -q gpua100
    #BSUB -J SGD_REllipsoid10000_10001
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
    
    python3 runtime.py \
        --manifold Ellipsoid \
        --geometry Riemannian \
        --dim 10000 \
        --T 100 \
        --method SGD \
        --batch_size 10001
        --tol 1e-4 \
        --max_iter 1000 \
        --number_repeats 5 \
        --timing_repeats 5 \
        --seed 2712 \
        --save_path timing_gpu/
    