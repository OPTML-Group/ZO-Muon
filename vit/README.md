# ViT fine-tuning experiments
Our code is primarily based on [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch.git).


## Usage
### 1. Download Pre-trained model (Google's Official Checkpoint)
We use ViT-B_16(**85.8M**) and  ViT-L_16(**303.4M**) pretrained on imagenet21k. Download the models and put them in `./checkpoint` via:
```
mkdir checkpoint
cd checkpoint
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz
```

### 2. Zeroth-Order Training
For **ZO-Muon**, we set `--zo_trainer` to `lowdim` and `--zo_optimizer` to `muon`. Run on CIFAR-10 via:
```
export CUDA_VISIBLE_DEVICES=0
python3 zo_train.py --name cifar10 \
        --dataset cifar10 \
        --model_type ViT-L_16 \
        --pretrained_dir checkpoint/ViT-L_16.npz \
        --learning_rate 1e-2 \
        --zo_trainer lowdim \
        --rank_r 64 \
        --zo_optimizer muon \
        --step_interval 100 \
        --multiple_sample \
        --num_samples 16 \
        --train_batch_size 64 \
        --num_steps 2000 \
        --eval_every 100 \
```
Run on CIFAR-100 via:
```
export CUDA_VISIBLE_DEVICES=0
python3 zo_train.py --name cifar100 \
        --dataset cifar100 \
        --model_type ViT-L_16 \
        --pretrained_dir checkpoint/ViT-L_16.npz \
        --learning_rate 1e-2 \
        --zo_trainer lowdim \
        --rank_r 64 \
        --zo_optimizer muon \
        --step_interval 100 \
        --multiple_sample \
        --num_samples 16 \
        --train_batch_size 256 \
        --num_steps 2000 \
        --eval_every 100 \
```


For Subspace-MeZO, we set `--zo_trainer` to `lowdim` and `--zo_optimizer` to `sgd`.
See an example code below:
```
export CUDA_VISIBLE_DEVICES=0
python3 zo_train.py --name cifar10 \
        --dataset cifar10 \
        --model_type ViT-L_16 \
        --pretrained_dir checkpoint/ViT-L_16.npz \
        --learning_rate 1e-4 \
        --zo_trainer lowdim \
        --rank_r 64 \
        --zo_optimizer sgd \
        --step_interval 100 \
        --train_batch_size 64 \
        --num_steps 20000 \
        --eval_every 100 \
```

Additional baselines include [MeZO](https://arxiv.org/pdf/2305.17333), [SparseMeZO](https://neurips.cc/virtual/2025/loc/san-diego/poster/117825), [HiZOO](https://arxiv.org/pdf/2402.15173), [LOZO](https://arxiv.org/pdf/2410.07698) and [SubZero](https://arxiv.org/pdf/2410.08989).
Run MeZO via:
```
export CUDA_VISIBLE_DEVICES=0
python3 zo_train.py --name cifar10 \
        --dataset cifar10 \
        --model_type ViT-L_16 \
        --pretrained_dir checkpoint/ViT-L_16.npz \
        --learning_rate 1e-4 \
        --zo_trainer mezo \
        --train_batch_size 64 \
        --num_steps 20000 \
        --eval_every 100 \
```
Run SparseMeZO via:
```
export CUDA_VISIBLE_DEVICES=0
python3 zo_train.py --name cifar10 \
        --dataset cifar10 \
        --model_type ViT-L_16 \
        --pretrained_dir checkpoint/ViT-L_16.npz \
        --learning_rate 1e-4 \
        --zo_trainer sparse \
        --train_batch_size 64 \
        --num_steps 20000 \
        --eval_every 100 \
```
Run HiZOO via:
```
export CUDA_VISIBLE_DEVICES=0
python3 zo_train.py --name cifar10 \
        --dataset cifar10 \
        --model_type ViT-L_16 \
        --pretrained_dir checkpoint/ViT-L_16.npz \
        --learning_rate 1e-4 \
        --zo_trainer hizoo \
        --train_batch_size 64 \
        --num_steps 2000 \
        --eval_every 100 \
```
Run LOZO via:
```
export CUDA_VISIBLE_DEVICES=0
python3 zo_train.py --name cifar10 \
        --dataset cifar10 \
        --model_type ViT-L_16 \
        --pretrained_dir checkpoint/ViT-L_16.npz \
        --learning_rate 1e-4 \
        --zo_trainer lozo \
        --rank_r 4 \
        --step_interval 100 \
        --train_batch_size 64 \
        --num_steps 20000 \
        --eval_every 100 \
```
Run SubZero via:
```
export CUDA_VISIBLE_DEVICES=0
python3 zo_train.py --name cifar10 \
        --dataset cifar10 \
        --model_type ViT-L_16 \
        --pretrained_dir checkpoint/ViT-L_16.npz \
        --learning_rate 1e-4 \
        --zo_trainer subzero \
        --rank_r 48 \
        --step_interval 1000 \
        --train_batch_size 64 \
        --num_steps 20000 \
        --eval_every 100 \
```


### 3. First-Order Training
Adam fine-tuning:
```
python3 train.py --name cifar10 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz
```

LoRA fine-tuning:
```
python3 train.py --name cifar10 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --lora
```