# # coding=utf-8
# from __future__ import absolute_import, division, print_function

# import logging
# import argparse
# import os
# import random
# import numpy as np

# from datetime import timedelta

# import torch
# # import torch.distributed as dist # Removed as DDP is not needed

# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter

# from models.modeling import VisionTransformer, CONFIGS
# from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
# from utils.data_utils import get_loader
# # from utils.dist_util import get_world_size # Removed as DDP is not needed


# logger = logging.getLogger(__name__)


# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count


# def simple_accuracy(preds, labels):
#     return (preds == labels).mean()


# def save_model(args, model):
#     # No need to check for .module since DDP is removed
#     model_to_save = model 
#     model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
#     torch.save(model_to_save.state_dict(), model_checkpoint)
#     logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


# def setup(args):
#     # Prepare model
#     config = CONFIGS[args.model_type]

#     num_classes = 10 if args.dataset == "cifar10" else 100

#     model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
#     model.load_from(np.load(args.pretrained_dir))
#     model.to(args.device)
#     num_params = count_parameters(model)

#     logger.info("{}".format(config))
#     logger.info("Training parameters %s", args)
#     logger.info("Total Parameter: \t%2.1fM" % num_params)
#     print(num_params)
#     return args, model


# def count_parameters(model):
#     params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return params/1000000


# def set_seed(args):
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.n_gpu > 0:
#         torch.cuda.manual_seed_all(args.seed)


# def valid(args, model, writer, test_loader, global_step):
#     # Validation!
#     eval_losses = AverageMeter()

#     logger.info("***** Running Validation *****")
#     logger.info("  Num steps = %d", len(test_loader))
#     logger.info("  Batch size = %d", args.eval_batch_size)

#     model.eval()
#     all_preds, all_label = [], []
#     epoch_iterator = tqdm(test_loader,
#                           desc="Validating... (loss=X.X)",
#                           bar_format="{l_bar}{r_bar}",
#                           dynamic_ncols=True)
#     loss_fct = torch.nn.CrossEntropyLoss()
#     for step, batch in enumerate(epoch_iterator):
#         batch = tuple(t.to(args.device) for t in batch)
#         x, y = batch
#         with torch.no_grad():
#             logits = model(x)[0]

#             eval_loss = loss_fct(logits, y)
#             eval_losses.update(eval_loss.item())

#             preds = torch.argmax(logits, dim=-1)

#         if len(all_preds) == 0:
#             all_preds.append(preds.detach().cpu().numpy())
#             all_label.append(y.detach().cpu().numpy())
#         else:
#             all_preds[0] = np.append(
#                 all_preds[0], preds.detach().cpu().numpy(), axis=0
#             )
#             all_label[0] = np.append(
#                 all_label[0], y.detach().cpu().numpy(), axis=0
#             )
#         epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

#     all_preds, all_label = all_preds[0], all_label[0]
#     accuracy = simple_accuracy(all_preds, all_label)

#     logger.info("\n")
#     logger.info("Validation Results")
#     logger.info("Global Steps: %d" % global_step)
#     logger.info("Valid Loss: %2.5f" % eval_losses.avg)
#     logger.info("Valid Accuracy: %2.5f" % accuracy)

#     writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
#     return accuracy


# def train(args, model):
#     """ Train the model """
#     os.makedirs(args.output_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

#     args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

#     # Prepare dataset
#     train_loader, test_loader = get_loader(args)

#     # Prepare optimizer and scheduler
#     optimizer = torch.optim.SGD(model.parameters(),
#                                 lr=args.learning_rate,
#                                 momentum=0.9,
#                                 weight_decay=args.weight_decay)
#     t_total = args.num_steps
#     if args.decay_type == "cosine":
#         scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
#     else:
#         scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

#     # Train!
#     logger.info("***** Running training *****")
#     logger.info("  Total optimization steps = %d", args.num_steps)
#     logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
#     logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

#     model.zero_grad()
#     set_seed(args)
#     losses = AverageMeter()
#     global_step, best_acc = 0, 0
#     while True:
#         model.train()
#         epoch_iterator = tqdm(train_loader,
#                               desc="Training (X / X Steps) (loss=X.X)",
#                               bar_format="{l_bar}{r_bar}",
#                               dynamic_ncols=True)
#         for step, batch in enumerate(epoch_iterator):
#             batch = tuple(t.to(args.device) for t in batch)
#             x, y = batch
#             loss = model(x, y)

#             if args.gradient_accumulation_steps > 1:
#                 loss = loss / args.gradient_accumulation_steps
            
#             loss.backward()

#             if (step + 1) % args.gradient_accumulation_steps == 0:
#                 losses.update(loss.item()*args.gradient_accumulation_steps)
                
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#                 scheduler.step()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 global_step += 1

#                 epoch_iterator.set_description(
#                     "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
#                 )
                
#                 writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
#                 writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                
#                 if global_step % args.eval_every == 0:
#                     accuracy = valid(args, model, writer, test_loader, global_step)
#                     if best_acc < accuracy:
#                         save_model(args, model)
#                         best_acc = accuracy
#                     model.train()

#                 if global_step % t_total == 0:
#                     break
#         losses.reset()
#         if global_step % t_total == 0:
#             break

#     writer.close()
#     logger.info("Best Accuracy: \t%f" % best_acc)
#     logger.info("End Training!")


# def main():
#     parser = argparse.ArgumentParser()
#     # Required parameters
#     parser.add_argument("--name", required=True,
#                         help="Name of this run. Used for monitoring.")
#     parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
#                         help="Which downstream task.")
#     parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
#                                                  "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
#                         default="ViT-B_16",
#                         help="Which variant to use.")
#     parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
#                         help="Where to search for pretrained ViT models.")
#     parser.add_argument("--output_dir", default="output", type=str,
#                         help="The output directory where checkpoints will be written.")

#     parser.add_argument("--img_size", default=224, type=int,
#                         help="Resolution size")
#     parser.add_argument("--train_batch_size", default=64, type=int,
#                         help="Total batch size for training.")
#     parser.add_argument("--eval_batch_size", default=64, type=int,
#                         help="Total batch size for eval.")
#     parser.add_argument("--eval_every", default=100, type=int,
#                         help="Run prediction on validation set every so many steps."
#                              "Will always run one evaluation at the end of training.")

#     parser.add_argument("--learning_rate", default=3e-2, type=float,
#                         help="The initial learning rate for SGD.")
#     parser.add_argument("--weight_decay", default=0, type=float,
#                         help="Weight deay if we apply some.")
#     parser.add_argument("--num_steps", default=10000, type=int,
#                         help="Total number of training epochs to perform.")
#     parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
#                         help="How to decay the learning rate.")
#     parser.add_argument("--warmup_steps", default=500, type=int,
#                         help="Step of training to perform learning rate warmup for.")
#     parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                         help="Max gradient norm.")

#     parser.add_argument('--seed', type=int, default=42,
#                         help="random seed for initialization")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#     parser.add_argument("--local_rank", type=int, default=-1,
#                         help="local_rank for distributed training on gpus")
#     # Removed fp16, local_rank, and loss_scale arguments
    
#     args = parser.parse_args()

#     # Setup CUDA, GPU (Single GPU or CPU only)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     args.n_gpu = torch.cuda.device_count()
#     args.device = device

#     # Setup logging
#     logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
#                         datefmt='%m/%d/%Y %H:%M:%S',
#                         level=logging.INFO)
#     logger.warning("Process rank: 0, device: %s, n_gpu: %s, distributed training: False, 16-bits training: False" %
#                    (args.device, args.n_gpu))

#     # Set seed
#     set_seed(args)

#     # Model & Tokenizer Setup
#     args, model = setup(args)

#     # Training
#     train(args, model)


# if __name__ == "__main__":
#     main()


# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import math
import numpy as np

from datetime import timedelta

import torch
import torch.nn as nn
# import torch.distributed as dist # Removed as DDP is not needed

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
# from utils.dist_util import get_world_size # Removed as DDP is not needed


logger = logging.getLogger(__name__)


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


class LoRALinear(nn.Module):
    """
    Wraps a standard nn.Linear layer with Low-Rank Adaptation.
    """
    def __init__(self, original_layer, rank=8, alpha=16):
        super(LoRALinear, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Ensure original weights are frozen
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False

        # LoRA Matrices
        # A: (in_features, rank)
        # B: (rank, out_features)
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with Kaiming, B with zeros (so starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original output
        original_out = self.original_layer(x)
        
        # LoRA path: (x @ A @ B) * scaling
        # Note: PyTorch Linear is x @ W.T. 
        # Here we do manual matmul: [Batch, Seq, In] @ [In, Rank] -> [Batch, Seq, Rank] @ [Rank, Out]
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return original_out + lora_out


def apply_lora(model, rank, alpha):
    """
    Applies LoRA to the model.
    1. Freezes all parameters.
    2. Replaces Linear layers in the transformer body with LoRALinear.
    3. Unfreezes the Classification Head.
    """
    # 1. Freeze entire model first
    for param in model.parameters():
        param.requires_grad = False

    # 2. Identify Linear layers to replace
    # We collect them in a list first to avoid modifying the module tree while iterating over it,
    # which causes RecursionError.
    
    layers_to_replace = []
    
    for name, module in model.named_modules():
        # Skip the head for LoRA replacement (we will unfreeze it later)
        if "head" in name:
            continue

        # We look for immediate children that are nn.Linear
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear):
                layers_to_replace.append((module, child_name, child))

    # 3. Perform the replacement
    for module, child_name, child in layers_to_replace:
        lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
        setattr(module, child_name, lora_layer)
                
    # 4. Unfreeze the head explicitly (we always want to train the classifier)
    # Assuming the standard ViT structure where model.head is the classifier
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        logger.warning("Could not find 'head' attribute. Please check model structure to ensure classifier is trainable.")

    return model


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    # No need to check for .module since DDP is removed
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
    
    # Apply LoRA if requested
    if args.lora:
        logger.info(f"Applying LoRA with Rank={args.lora_rank}, Alpha={args.lora_alpha}")
        model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    model.to(args.device)
    
    total_params, trainable_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameters: \t%2.1fM" % total_params)
    logger.info("Trainable Parameters: \t%2.1fM" % trainable_params)
    logger.info("Trainable Ratio: \t%.2f%%" % (trainable_params/total_params * 100))
    
    return args, model


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params/1000000, trainable_params/1000000


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


def train(args, model):
    """ Train the model """
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    # IMPORTANT: Only pass trainable parameters to the optimizer
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                
                if global_step % args.eval_every == 0:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    
    # LoRA arguments
    parser.add_argument("--lora", action='store_true',
                        help="Use LoRA for training.")
    parser.add_argument("--lora_rank", default=8, type=int,
                        help="LoRA rank parameter.")
    parser.add_argument("--lora_alpha", default=16, type=int,
                        help="LoRA alpha parameter.")
    
    args = parser.parse_args()

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
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()