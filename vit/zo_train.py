# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)
from zo_optimizer import MeZOOptimizer,LOZOOptimizer,LowDimOptimizer,SubZeroOptimizer,HessianOptimizer, SparseOptimizer

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model 
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


import os
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm  # Ensure tqdm is imported
import time
# Assuming these are defined elsewhere
# from utils import AverageMeter, get_loader, set_seed, save_model, valid
# from mezo import MeZOOptimizer

logger = logging.getLogger(__name__)

def train(args, model):
    """ Train the model using Zeroth-Order Optimization with Time Limits and Time-Based Eval """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log File Initialization
    if hasattr(args, 'log_path') and args.log_path:
        log_dir = os.path.dirname(args.log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(args.log_path, "w") as f:
            f.write("") 

    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    train_loader, test_loader = get_loader(args)
    steps_per_epoch = len(train_loader) if len(train_loader) > 0 else 1

    # Optimizer Initialization (Keeping your existing logic)
    if args.zo_trainer == 'mezo':
        zo_optimizer = MeZOOptimizer(model, args)
    elif args.zo_trainer == 'sparse':
        zo_optimizer = SparseOptimizer(model, args)
    elif args.zo_trainer == 'hessian':
        zo_optimizer = HessianOptimizer(model, args)
    elif args.zo_trainer == 'lozo':
        zo_optimizer = LOZOOptimizer(model, args)
    elif args.zo_trainer == 'subzero':
        zo_optimizer = SubZeroOptimizer(model, args)
    elif args.zo_trainer == 'lowdim':
        zo_optimizer = LowDimOptimizer(model, args)
    else:
        raise ValueError(f"Unknown zo_trainer type: {args.zo_trainer}")

    t_total = args.num_steps
    logger.info("***** Running training (Zeroth-Order) *****")
    
    model.zero_grad()
    set_seed(args)
    
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    
    start_time = time.time()
    last_eval_time = start_time 
    
    pbar = tqdm(total=t_total, desc="Training", dynamic_ncols=True)
    stop_training = False 

    while True:
        for step, batch in enumerate(train_loader):
            if isinstance(batch, (tuple, list)):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch
            else:
                x = batch
                y = None 

            loss_val = zo_optimizer.zo_step(x, y)
            zo_optimizer.zo_update()

            losses.update(loss_val.item())
            global_step += 1
            pbar.update(1)
            
            # Logging (Every 10 steps)
            if global_step % 10 == 0:
                current_loss = losses.avg
                tqdm.write(f"Step: {global_step} | Loss: {current_loss:.4f}")
                
                try:
                    if hasattr(zo_optimizer, 'optimizer'):
                        current_lr = zo_optimizer.optimizer.param_groups[0]['lr']
                    elif hasattr(zo_optimizer, 'param_groups'):
                        current_lr = zo_optimizer.param_groups[0]['lr']
                    else:
                        current_lr = args.learning_rate
                except:
                    current_lr = args.learning_rate

                current_epoch = global_step / steps_per_epoch
                log_entry = f"{{'loss': {current_loss:.4f}, 'learning_rate': {current_lr}, 'epoch': {current_epoch:.2f}}}\n"

                if hasattr(args, 'log_path') and args.log_path:
                    with open(args.log_path, "a") as f:
                        f.write(log_entry)

                losses.reset()

            writer.add_scalar("train/loss", scalar_value=loss_val.item(), global_step=global_step)
            
            # -------------------------------------------------------
            # Standard Evaluation Logic
            # -------------------------------------------------------
            current_time = time.time()
            perform_eval = False
            eval_reason = ""

            if args.eval_every > 0 and global_step % args.eval_every == 0:
                perform_eval = True
                eval_reason = f"step {global_step}"
            elif hasattr(args, 'eval_time') and args.eval_time and args.eval_time > 0:
                time_since_last = current_time - last_eval_time
                if time_since_last >= args.eval_time:
                    perform_eval = True
                    eval_reason = f"{time_since_last:.0f}s since last eval"

            if perform_eval:
                tqdm.write(f"Running Validation ({eval_reason})...") 
                accuracy = valid(args, model, writer, test_loader, global_step)
                if best_acc < accuracy:
                    save_model(args, model)
                    best_acc = accuracy
                model.train()
                last_eval_time = time.time()

            if hasattr(args, 'max_time') and args.max_time and args.max_time > 0:
                elapsed_time = time.time() - start_time
                
                if elapsed_time > args.max_time:
                    tqdm.write(f"\n[Time Limit Reached] Stopping after {elapsed_time:.2f}s")
                    logger.info(f"Time limit reached: {elapsed_time:.2f}s > {args.max_time}s")
                    
                    # Check if we need a final evaluation.
                    # We skip this only if we JUST evaluated (e.g., < 10 seconds ago)
                    time_since_last_eval = time.time() - last_eval_time
                    
                    if time_since_last_eval > 10:  # Threshold to avoid double-eval
                        tqdm.write("Running Final Validation before exit...")
                        accuracy = valid(args, model, writer, test_loader, global_step)
                        if best_acc < accuracy:
                            save_model(args, model)
                            best_acc = accuracy
                    
                    stop_training = True
                    break

            if global_step >= t_total:
                stop_training = True
                break
        
        if stop_training:
            break

    pbar.close()
    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")

def main():
    parser = argparse.ArgumentParser()
    
    # ======================================================
    # General Training Arguments
    # ======================================================
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int, help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps.")

    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The constant learning rate for ZO.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup.")
    
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate (Set to 1 for ZO).")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")

    # ======================================================
    # Zeroth-Order (ZO) General Arguments
    # ======================================================
    parser.add_argument("--zo_trainer", choices=["mezo", "lozo", "lowdim","subzero", "sparse", "hessian"], default="mezo",
                        help="Which ZO optimizer class to use.")
    
    parser.add_argument("--zo_eps", default=1e-3, type=float,
                        help="Epsilon value for Zeroth-Order perturbation.")
    parser.add_argument("--sparsity", default=0.8, type=float,
                        help="sparsity")
    # In your argument parsing section
    parser.add_argument("--max_time", type=float, default=None, help="Maximum training time in seconds.")
    parser.add_argument("--eval_time", type=float, default=None, help="Maximum training time in seconds.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="loss path")
    # ======================================================
    # LowDim Optimizer Specific Arguments
    # ======================================================
    parser.add_argument("--zo_optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "muon", "muon_m", "polar", "stiefel", "sgd_var", "sign", "muon_svd"],
                        help="Internal update rule for LowDim optimizer (e.g., 'muon', 'polar').")
    
    parser.add_argument("--rank_r", type=int, default=16,
                        help="Rank of the subspace for LowDim optimization.")
    
    parser.add_argument("--step_interval", type=int, default=100,
                        help="Interval (in steps) to resample the projection matrix P (switch_interval).")
    
    parser.add_argument("--multiple_sample", action="store_true",
                        help="If True, use multiple samples (center-based) instead of single antithetic sample.")
    
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of random directions to sample per step (used if multiple_sample=True).")

    args = parser.parse_args()

    # Enforce gradient accumulation is 1 for this ZO implementation
    if args.gradient_accumulation_steps != 1:
        logger.warning("Gradient accumulation > 1 is not supported in this ZO implementation. Resetting to 1.")
        args.gradient_accumulation_steps = 1

    # Setup CUDA, GPU (Single GPU or CPU only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Process rank: 0, device: %s, n_gpu: %s, distributed training: False, 16-bits training: False" %
                   (args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    # Assuming setup returns (args, model)
    args, model = setup(args)

    # Training
    train(args, model)

if __name__ == "__main__":
    main()