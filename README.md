## IterativeVAE: Non-Autoregressive Neural Sequence Modeling by Iterative Refinement from Latent Space

```
# ours: 
python train_vae.py --h

python train_vae.py --gpuid 1 --batchsize 1024 --realbatch 64 --trainsamples 100000 --validsamples 5000 --niter 500 --weight_decay 0.0001

python train_vae.py --realbatch 32 --trainsamples 50000 --validsamples 1000 --gpuid 3 --embedsize 256 --niter 500 --lr 0.001 --weight_decay 0.0001 --batchsize 256 --logstep 200

# NAT:
python train_natransformer.py --gpuid 2 --batchsize 1024 --realbatch 128 --vocabattn --posattn --niter 100 --warmup --trainsamples 100000 --validsamples 5000

# CMLM:
python train_cmlm.py --gpuid 2 --batchsize 1024 --realbatch 128 --vocabattn --posattn --niter 100 --warmup --trainsamples 100000 --validsamples 5000

# AR:
python train_transformer.py --gpuid 2 --batchsize 1024 --realbatch 128 --vocabattn --posattn --niter 100 --warmup --trainsamples 100000 --validsamples 5000
```
