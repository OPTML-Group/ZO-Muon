
import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import get_loader

class MeZOOptimizer(object):
    """
    Zeroth-Order Optimizer based on MeZO (Memory-efficient Zeroth Order).
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.zo_eps = args.zo_eps
        
        # Cache parameters requiring grad to avoid iterating all modules every step
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        self.zo_random_seed = None
        self.projected_grad = None

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        theta = theta + scaling_factor * z * eps
        """
        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.zo_eps

    def zo_forward(self, inputs, labels):
        """
        Get (no gradient) loss from the model.
        """
        self.model.eval()
        with torch.inference_mode():
            # In the original script, model(x, y) returns the loss
            loss = self.model(inputs, labels)
        return loss.detach()

    # def zo_step(self, inputs, labels):
    #     """
    #     Estimate gradient by MeZO. Return the loss from f(theta + z)
    #     """
    #     # Sample the random seed for sampling z
    #     self.zo_random_seed = np.random.randint(1000000000)

    #     # 1. First function evaluation: f(theta + z*eps)
    #     self.zo_perturb_parameters(scaling_factor=1)
    #     loss1 = self.zo_forward(inputs, labels)

    #     # 2. Second function evaluation: f(theta - z*eps)
    #     # We move from (theta + z*eps) to (theta - z*eps), so we subtract 2*z*eps
    #     self.zo_perturb_parameters(scaling_factor=-2)
    #     loss2 = self.zo_forward(inputs, labels)

    #     # Calculate projected gradient: (f(theta+z*eps) - f(theta-z*eps)) / (2*eps)
    #     self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()

    #     # 3. Reset model back to its parameters at start of step: theta
    #     # We move from (theta - z*eps) to theta, so we add 1*z*eps
    #     self.zo_perturb_parameters(scaling_factor=1)
        
    #     return loss1
    
    def zo_step(self, inputs, labels):
        """
        Estimate gradient by MeZO. 
        Supports:
        1. Multiple Samples (Center-based estimation)
        2. Single Sample (Antithetic estimation)
        """
        args = self.args
        model = self.model

        # Storage for seeds and scalar gradients for aggregation in zo_update
        self.zo_random_seeds = []
        self.projected_grads_list = []

        # Base random seed generator
        base_seed = np.random.randint(1000000000)

        # --- Strategy 1: Multiple Sampling (Center-based) ---
        # Formula: (f(theta + z) - f(theta)) / eps
        if getattr(args, 'multiple_sample', False):
            num_samples = args.num_samples
            
            # 1. Baseline function evaluation f(theta)
            # Note: We need to ensure no gradients are tracked during ZO
            with torch.no_grad():
                loss_baseline = self.zo_forward(inputs, labels)

            for i in range(num_samples):
                current_seed = base_seed + i
                self.zo_random_seeds.append(current_seed)
                
                # Set seed for this specific sample
                self.zo_random_seed = current_seed
                
                # 2. Perturb: theta + z*eps
                self.zo_perturb_parameters(scaling_factor=1)
                
                # 3. Forward: f(theta + z*eps)
                with torch.no_grad():
                    loss_perturbed = self.zo_forward(inputs, labels)
                
                # 4. Estimate Gradient Scalar
                grad_est = ((loss_perturbed - loss_baseline) / args.zo_eps).item()
                self.projected_grads_list.append(grad_est)
                
                # 5. Reset: theta + z*eps -> theta
                self.zo_perturb_parameters(scaling_factor=-1)
            
            return loss_baseline

        # --- Strategy 2: Single Sample (Antithetic) ---
        # Formula: (f(theta + z) - f(theta - z)) / (2 * eps)
        else:
            # Use one seed
            self.zo_random_seed = base_seed
            self.zo_random_seeds.append(base_seed)

            # 1. First function evaluation: f(theta + z*eps)
            self.zo_perturb_parameters(scaling_factor=1)
            with torch.no_grad():
                loss1 = self.zo_forward(inputs, labels)

            # 2. Second function evaluation: f(theta - z*eps)
            # Move from (theta + z*eps) to (theta - z*eps) requires subtracting 2*z*eps
            self.zo_perturb_parameters(scaling_factor=-2)
            with torch.no_grad():
                loss2 = self.zo_forward(inputs, labels)

            # 3. Estimate Gradient Scalar
            projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
            self.projected_grads_list.append(projected_grad)

            # 4. Reset: theta - z*eps -> theta
            # Add 1*z*eps
            self.zo_perturb_parameters(scaling_factor=1)
            
            return loss1

    def zo_update(self):
        """
        Update the parameters with the estimated gradients.
        Handles aggregation of multiple samples and Muon optimization.
        """
        args = self.args
        num_samples = args.num_samples
        lr = args.learning_rate

        for name, param in self.named_parameters_to_optim:
            
            # 1. Aggregate Gradient Estimate
            # We reconstruct the gradient: G = (1/N) * sum(scalar_i * z_i)
            grad_est = torch.zeros_like(param.data)
            
            for i in range(num_samples):
                # Reset seed to reconstruct the specific z_i used in forward pass
                torch.manual_seed(self.zo_random_seeds[i])
                
                # Re-sample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), 
                               device=param.data.device, dtype=param.data.dtype)
                
                # Accumulate
                grad_est += self.projected_grads_list[i] * z
            
            # Average the gradient
            if num_samples > 1:
                grad_est /= num_samples

            # 2. Apply Optimization Logic

            # --- Option A: Low-Rank Muon (Randomized Newton-Schulz) ---
            if getattr(args, 'zo_optimizer', 'sgd') == "lr_muon":
                if grad_est.ndim >= 2:
                    # Save original shape to restore later
                    orig_shape = grad_est.shape
                    
                    # Flatten to 2D: [d_out, d_in_flattened]
                    # For a Conv2d [C_out, C_in, H, W], this becomes [C_out, C_in*H*W]
                    grad_2d = grad_est.view(orig_shape[0], -1)
                    
                    m, n = grad_2d.shape
                    rank_r = 64 # Hyperparameter
                    curr_r = min(rank_r, m, n)
                    
                    # 1. Generate Random Projection Matrix
                    Q_rand = torch.randn((n, curr_r), device=param.device, dtype=torch.float32)
                    
                    # 2. Compute the Sketch Y = G @ Q_rand -> (m x r)
                    Y = grad_2d.to(torch.float32) @ Q_rand
                    
                    # 3. Orthogonalize Y to get P (m x r)
                    P, _ = torch.linalg.qr(Y)
                    P = P.to(dtype=param.dtype)
                    
                    # 4. Project Gradient: P_T (r x m) @ G (m x n) -> (r x n)
                    projected_grad = P.T @ grad_2d
                    
                    # 5. Apply Newton-Schulz to the subspace
                    update_sub = zeropower_via_newtonschulz5(projected_grad, steps=5)
                    
                    # 6. Project back: P (m x r) @ update_sub (r x n) -> (m x n)
                    full_update_2d = P @ update_sub.to(dtype=P.dtype)
                    
                    # 7. Reshape back to original dimensions and apply
                    full_update = full_update_2d.view(orig_shape)
                    param.data.add_(full_update, alpha=-lr)
                else:
                    # Fallback for 1D tensors (biases, layernorm weights)
                    param.data.add_(grad_est, alpha=-lr)

            # --- Option B: Full Muon (Standard Newton-Schulz) ---
            elif getattr(args, 'zo_optimizer', 'sgd') == "muon":
                if grad_est.ndim >= 2:
                    # Save original shape
                    orig_shape = grad_est.shape
                    
                    # Flatten to 2D: [d_out, d_in_flattened]
                    grad_2d = grad_est.view(orig_shape[0], -1)
                    
                    # Apply Newton-Schulz directly to the full gradient matrix
                    orthogonal_grad_2d = zeropower_via_newtonschulz5(grad_2d, steps=5)
                    
                    # Reshape back and apply
                    orthogonal_grad = orthogonal_grad_2d.view(orig_shape)
                    param.data.add_(orthogonal_grad, alpha=-lr)
                else:
                    param.data.add_(grad_est, alpha=-lr)

            # --- Option C: Standard SGD ---
            else:
                # Add Weight Decay (excluding bias/layernorm)
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        grad_est += args.weight_decay * param.data
                
                # Apply update
                param.data.add_(grad_est, alpha=-lr)

        #self.lr_scheduler.step()


    # def zo_update(self):
    #     """
    #     Update the parameters with the estimated gradients.
    #     """
    #     args = self.args

    #     # Reset the random seed to regenerate the exact same z used in perturbation
    #     torch.manual_seed(self.zo_random_seed)     

    #     for name, param in self.named_parameters_to_optim:
    #         # Resample z
    #         z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
    #         # Calculate update: projected_grad * z
    #         update = self.projected_grad * z
            
    #         # Add weight decay if applicable
    #         # (Excluding bias and layernorm from weight decay is standard practice)
    #         if args.weight_decay > 0:
    #             if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
    #                 update += args.weight_decay * param.data

    #         # Apply Update: theta = theta - lr * update
    #         param.data = param.data - args.learning_rate * update

    


class SparseOptimizer(object):
    """
    Zeroth-Order Optimizer with Sparse Perturbation.
    Based on MeZO, but the perturbation vector z is masked by a random sparsity mask.
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.zo_eps = args.zo_eps
        
        # Default sparsity to 0.5 if not provided in args
        self.sparsity = getattr(args, 'sparsity', 0.5)
        
        # Cache parameters requiring grad to avoid iterating all modules every step
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        self.zo_random_seed = None
        self.projected_grad = None

    def _sample_z_and_mask(self, param):
        """
        Helper function to sample z and apply the sparsity mask.
        Returns: masked_z
        """
        # 1. Sample standard Gaussian noise
        z = torch.normal(mean=0, std=1, size=param.data.size(), 
                         device=param.data.device, dtype=param.data.dtype)
        
        # 2. Create a random mask (Bernoulli distribution)
        # Probability of 1 is (1 - sparsity). e.g., if sparsity is 0.9, keep prob is 0.1.
        # We generate a float tensor of 0s and 1s.
        keep_prob = 1.0 - self.sparsity
        mask = torch.bernoulli(torch.full_like(z, keep_prob))
        
        # 3. Apply mask
        return z * mask

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with sparse random vector z.
        theta = theta + scaling_factor * (mask * z) * eps
        """
        # Set the random seed to ensure that we sample the same z and mask
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            # Generate the sparse perturbation vector
            masked_z = self._sample_z_and_mask(param)
            
            # Apply perturbation
            param.data = param.data + scaling_factor * masked_z * self.zo_eps

    def zo_forward(self, inputs, labels):
        """
        Get (no gradient) loss from the model.
        """
        self.model.eval()
        with torch.inference_mode():
            loss = self.model(inputs, labels)
        return loss.detach()

    def zo_step(self, inputs, labels):
        """
        Estimate gradient by Sparse MeZO. 
        """
        # Sample the random seed for sampling z and the mask
        self.zo_random_seed = np.random.randint(1000000000)

        # 1. First function evaluation: f(theta + masked_z*eps)
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(inputs, labels)

        # 2. Second function evaluation: f(theta - masked_z*eps)
        # We move from (theta + masked_z*eps) to (theta - masked_z*eps), subtract 2*masked_z*eps
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(inputs, labels)

        # Calculate projected gradient
        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()

        # 3. Reset model back to its parameters at start of step: theta
        # We move from (theta - masked_z*eps) to theta, add 1*masked_z*eps
        self.zo_perturb_parameters(scaling_factor=1)
        
        return loss1

    def zo_update(self):
        """
        Update the parameters with the estimated gradients using the sparse z.
        """
        args = self.args

        # Reset the random seed to regenerate the exact same z and mask used in perturbation
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            # Resample z with the exact same mask as used in zo_step
            masked_z = self._sample_z_and_mask(param)
            
            # Calculate update: projected_grad * (mask * z)
            update = self.projected_grad * masked_z
            
            # Add weight decay if applicable
            if args.weight_decay > 0:
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    update += args.weight_decay * param.data

            # Apply Update: theta = theta - lr * update
            param.data = param.data - args.learning_rate * update



class HessianOptimizer(object):
    """
    Zeroth-Order Optimizer with Hessian-aware adaptive scaling.
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.zo_eps = args.zo_eps
        self.hessian_smooth = getattr(args, 'hessian_smooth', 1e-8) # Default smoothing factor if not in args
        
        # Cache parameters requiring grad
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        # Initialize Hessian Matrix (diagonal approximation) for each parameter
        # Initialized to 1.0 to avoid division by zero initially
        self.hessian_matrix = {}
        for name, param in self.named_parameters_to_optim:
            self.hessian_matrix[name] = torch.ones_like(param.data)

        self.zo_random_seed = None
        self.loss1 = None
        self.loss2 = None
        self.loss_original = None

    def zo_forward(self, inputs, labels):
        """
        Get (no gradient) loss from the model.
        """
        self.model.eval()
        with torch.inference_mode():
            # Assuming standard call signature model(input, labels) -> loss
            # Adjust if your model returns a tuple or dict
            loss = self.model(inputs, labels)
            if isinstance(loss, dict):
                loss = loss['loss']
            elif isinstance(loss, tuple):
                loss = loss[0]
        return loss.detach()

    def _perturb_parameters(self, random_seed, scaling_factor):
        """
        Perturb parameters using the Hessian-scaled noise.
        theta = theta + scaling_factor * (z / sqrt(H)) * eps
        """
        torch.manual_seed(random_seed)
        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), 
                             device=param.data.device, dtype=param.data.dtype)
            
            # Apply perturbation scaled by inverse sqrt of Hessian
            # Added epsilon to sqrt to prevent division by zero for numerical stability if needed, 
            # though initialization is 1.0.
            h_scale = torch.sqrt(self.hessian_matrix[name])
            param.data = param.data + (scaling_factor / h_scale) * z * self.zo_eps

    def zo_step(self, inputs, labels):
        """
        Perform the function evaluations required to estimate Gradient and Hessian.
        Returns the loss from the perturbed state (loss1).
        """
        self.zo_random_seed = np.random.randint(1000000000)

        # 0. Baseline evaluation (f(theta)) - Required for Hessian estimation
        # Note: Standard MeZO doesn't need this, but Hessian-aware version does
        # to calculate the curvature (loss1 + loss2 - 2*loss_original).
        self.loss_original = self.zo_forward(inputs, labels)

        # 1. First function evaluation: f(theta + z_scaled)
        self._perturb_parameters(self.zo_random_seed, scaling_factor=1)
        self.loss1 = self.zo_forward(inputs, labels)

        # 2. Second function evaluation: f(theta - z_scaled)
        # Move from +1 state to -1 state, so scaling factor is -2
        self._perturb_parameters(self.zo_random_seed, scaling_factor=-2)
        self.loss2 = self.zo_forward(inputs, labels)

        # 3. Reset model back to original parameters: theta
        # Move from -1 state to 0 state, so scaling factor is +1
        self._perturb_parameters(self.zo_random_seed, scaling_factor=1)

        return self.loss1

    def zo_update(self):
        """
        Update the Hessian Matrix and the Parameters.
        """
        # Regenerate the exact noise z used in step
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), 
                             device=param.data.device, dtype=param.data.dtype)

            # --- 1. Update Hessian Matrix ---
            # H_temp = H_curr * z * z
            # Note: In the original code provided, it used self.Hessian_matrix[name] * z * z.
            # This implies we are estimating curvature along direction z relative to current scale.
            hessian_temp = self.hessian_matrix[name] * z * z
            
            # Finite difference for second derivative: (f(x+h) + f(x-h) - 2f(x)) / h^2
            # Here h (step size) is effectively (eps / sqrt(H))
            curvature_diff = torch.abs(self.loss1 + self.loss2 - 2 * self.loss_original)
            hessian_estimator = (curvature_diff * hessian_temp * self.hessian_smooth) / (2 * self.args.zo_eps ** 2)
            
            # Moving average update
            self.hessian_matrix[name] = ((1 - self.hessian_smooth) * self.hessian_matrix[name] + hessian_estimator)

            # --- 2. Update Parameters ---
            # Gradient estimate: (f(x+h) - f(x-h)) / (2h)
            # grad = (loss1 - loss2) / (2 * eps) * (z / sqrt(H))
            grad_scalar = (self.loss1 - self.loss2) / (2 * self.args.zo_eps)
            grad_vector = grad_scalar * z / torch.sqrt(self.hessian_matrix[name])
            
            # Apply Weight Decay
            if self.args.weight_decay > 0:
                 # Standard exclusion of bias/layernorm
                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    grad_vector += self.args.weight_decay * param.data

            # Update weights
            param.data = param.data - self.args.learning_rate * grad_vector

import torch
import numpy as np
class LOZOOptimizer(object):
    """
    Low-Rank Zeroth-Order Optimizer (LOZO).
    Updated to handle high-dimensional tensors (3D/4D) via flattening.
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.zo_eps = args.zo_eps
        self.rank_r = args.rank_r
        self.step_interval = args.step_interval
        
        # Cache parameters requiring grad
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        self.zo_random_seed = None
        self.projected_grad = None
        
        # LOZO specific state
        self.step = 0
        self.v = {} # Cache for matrix v

    def random_gaussian_matrix(self, m, n, device, dtype, random_seed=None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
        return torch.randn(m, n, device=device, dtype=dtype)

    def lowrank_zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector uv^t for weights (>=2D), 
        and standard z for 1D weights (bias/layernorm).
        """
        # Set the random seed to ensure consistency between perturb and update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            # Handle >= 2D weights (Linear, Conv2d, Embeddings)
            if param.data.ndim >= 2:
                # 1. Flatten dimensions: [d0, d1, d2...] -> [m, n]
                orig_shape = param.data.shape
                m = orig_shape[0]
                n = param.data.numel() // m
                
                # Update v only at specific intervals
                if self.step % self.step_interval == 0:
                    # v shape is [n, r]
                    v = torch.randn(n, self.rank_r, device=param.data.device, dtype=param.data.dtype)
                    self.v[name] = v
                else:
                    # Use cached v
                    if name not in self.v:
                        # Fallback initialization
                        self.v[name] = torch.randn(n, self.rank_r, device=param.data.device, dtype=param.data.dtype)
                    v = self.v[name]
                
                # Generate u [m, r]
                u = self.random_gaussian_matrix(m=m, n=self.rank_r, device=param.data.device, dtype=param.data.dtype)
                
                # Calculate perturbation: (u @ v.t()) results in [m, n]
                perturbation = u @ v.t()
                
                # Apply perturbation: Reshape [m, n] back to orig_shape
                param.data = param.data + scaling_factor * perturbation.view(orig_shape) * self.zo_eps
            
            # Handle 1D weights (Standard MeZO perturbation)
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.zo_eps

    def zo_forward(self, inputs, labels):
        """
        Get (no gradient) loss from the model.
        """
        self.model.eval()
        with torch.inference_mode():
            loss = self.model(inputs, labels)
            # Handle distributed training loss averaging if necessary
            if hasattr(loss, "mean"): 
                loss = loss.mean()
        return loss.detach()

    def zo_step(self, inputs, labels):
        """
        Estimate gradient by LOZO. 
        """
        # Increment step counter for v-matrix update logic
        self.step += 1

        # Sample the random seed for sampling u (and v if interval met)
        self.zo_random_seed = np.random.randint(1000000000)

        # 1. First function evaluation: f(theta + noise)
        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(inputs, labels)

        # 2. Second function evaluation: f(theta - noise)
        # Move from (theta + noise) to (theta - noise) -> subtract 2*noise
        self.lowrank_zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(inputs, labels)

        # Calculate projected gradient
        # LOZO scaling: (loss1 - loss2) / (2 * eps * rank)
        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps * self.rank_r)).item()

        # 3. Reset model back to start: theta
        # Move from (theta - noise) to theta -> add 1*noise
        self.lowrank_zo_perturb_parameters(scaling_factor=1)
        
        return loss1

    def zo_update(self):
        """
        Update the parameters with the estimated gradients.
        """
        args = self.args

        # Reset the random seed to regenerate the exact same u/z used in perturbation
        torch.manual_seed(self.zo_random_seed)     

        for name, param in self.named_parameters_to_optim:
            
            # Calculate Gradient Estimate (G)
            if param.data.ndim >= 2:
                # 1. Flatten dimensions again to match perturbation logic
                orig_shape = param.data.shape
                m = orig_shape[0]
                n = param.data.numel() // m

                # Reconstruct u and retrieve v
                v = self.v[name] # [n, r]
                u = self.random_gaussian_matrix(m=m, n=self.rank_r, device=param.data.device, dtype=param.data.dtype) # [m, r]
                
                # G is [m, n]
                G = self.projected_grad * (u @ v.t())
                
                # Apply Weight Decay
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        # Add weight decay to the flat gradient (view param as [m, n])
                        G += args.weight_decay * param.data.view(m, n)

                # Apply Update: theta = theta - lr * G
                # Reshape G back to orig_shape
                param.data = param.data - args.learning_rate * G.view(orig_shape)

            else:
                # Reconstruct z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                G = self.projected_grad * z

                # Apply Weight Decay
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        G += args.weight_decay * param.data

                # Apply Update
                param.data = param.data - args.learning_rate * G

import torch
import numpy as np

import torch
import numpy as np
import math

def get_approximate_svd(matrix, rank):
    """
    Computes truncated SVD.
    matrix: [m, n]
    returns: U [m, r], V [r, n]
    """
    # Ensure float32 for stability
    orig_dtype = matrix.dtype
    mat_float = matrix.float()
    
    # Full SVD (or use randomized SVD for speed in production)
    U, S, Vh = torch.linalg.svd(mat_float, full_matrices=False)
    
    # Determine effective rank (cannot be larger than matrix dimensions)
    r = min(rank, U.shape[1])
    
    U_r = U[:, :r].to(orig_dtype)
    V_r = Vh[:r, :].to(orig_dtype) # Vh is already V^T
    
    return U_r, V_r

class SubZeroOptimizer(object):
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.zo_eps = args.zo_eps
        self.rank_r = args.rank_r
        self.step_interval = args.step_interval
        
        # Collect parameters
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        # State storage
        self.subspace_state = {} # Stores U and V
        self.zo_random_seed = None
        self.projected_grad = 0.0
        self.step_curr = 0

    def zo_subspace_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturbs parameters. 
        - 2D+ tensors: Projected noise (U @ z0 @ V)
        - 1D tensors: Standard Gaussian noise
        """
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            
            # --- Case 1: High-Dimensional Parameters (Weights) ---
            if param.data.ndim >= 2:
                # Flatten logic: [d0, d1, d2...] -> [m, n]
                orig_shape = param.data.shape
                m = orig_shape[0]
                n = param.data.numel() // m
                
                # Initialize state if needed
                if name not in self.subspace_state:
                    self.subspace_state[name] = {'U': None, 'V': None}
                
                # Update SVD bases periodically
                if self.subspace_state[name]['U'] is None or self.step_curr % self.step_interval == 0:
                    # View as matrix for SVD
                    param_view = param.data.view(m, n)
                    U, V = get_approximate_svd(param_view, self.rank_r)
                    self.subspace_state[name]['U'] = U
                    self.subspace_state[name]['V'] = V
                
                U = self.subspace_state[name]['U'] # [m, r]
                V = self.subspace_state[name]['V'] # [r, n]
                
                # Dynamic Rank Safety: Use the actual rank of U, not self.rank_r
                # This prevents shape mismatch if the layer is smaller than rank_r
                actual_rank = U.shape[1] 
                
                # Sample z0 in the reduced space [r, r]
                z0 = torch.normal(mean=0, std=1, size=(actual_rank, actual_rank), 
                                  device=param.data.device, dtype=param.data.dtype)
                
                # Project: U [m, r] @ z0 [r, r] @ V [r, n] -> [m, n]
                # Scale correction ensures variance is maintained after projection
                scale_correction = math.sqrt(param.data.numel() / z0.numel())
                perturbation = (U @ z0 @ V) * scale_correction
                
                # Apply perturbation
                param.data = param.data + scaling_factor * perturbation.view(orig_shape) * self.zo_eps

            # --- Case 2: 1D Parameters (Bias, LayerNorm) ---
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), 
                                 device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + scaling_factor * z * self.zo_eps

    def zo_forward(self, inputs, labels):
        self.model.eval()
        with torch.inference_mode():
            # Adjust this call based on your model's specific forward signature
            loss = self.model(inputs, labels) 
            if hasattr(loss, "mean"):
                loss = loss.mean()
        return loss.detach()

    def zo_step(self, inputs, labels):
        """
        Estimate gradient via Antithetic Sampling (f(x+z) - f(x-z)) / 2eps
        """
        self.step_curr += 1
        
        # Generate a new seed for this step
        self.zo_random_seed = np.random.randint(1000000000)

        # 1. Perturb (+)
        self.zo_subspace_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(inputs, labels)

        # 2. Perturb (-)
        # Move from +1 to -1 requires subtracting 2
        self.zo_subspace_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(inputs, labels)

        # 3. Calculate Projected Gradient Scalar
        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()

        # 4. Reset Parameters to original state
        self.zo_subspace_perturb_parameters(scaling_factor=1)

        return (loss1 + loss2) / 2

    def zo_update(self):
        """
        Apply the update using the estimated projected gradient.
        """
        args = self.args
        lr = args.learning_rate # Or self.optimizer.param_groups[0]['lr']

        # Re-seed to regenerate the exact same z/z0 used in zo_step
        torch.manual_seed(self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            
            # --- High-Dim Update ---
            if param.data.ndim >= 2:
                orig_shape = param.data.shape
                m = orig_shape[0]
                n = param.data.numel() // m
                
                U = self.subspace_state[name]['U']
                V = self.subspace_state[name]['V']
                actual_rank = U.shape[1]

                # Reconstruct z0
                z0 = torch.normal(mean=0, std=1, size=(actual_rank, actual_rank), 
                                  device=param.data.device, dtype=param.data.dtype)
                
                # Reconstruct full gradient estimate
                scale_correction = math.sqrt(param.data.numel() / z0.numel())
                grad_est = (U @ z0 @ V) * scale_correction * self.projected_grad
                
                # Weight Decay
                if args.weight_decay > 0:
                     # Exclude bias/layernorm from weight decay if needed, 
                     # though usually 2D params are weights.
                    grad_est += args.weight_decay * param.data.view(m, n)

                # Update
                param.data = param.data - lr * grad_est.view(orig_shape)

            # --- 1D Update ---
            else:
                z = torch.normal(mean=0, std=1, size=param.data.size(), 
                                 device=param.data.device, dtype=param.data.dtype)
                
                grad_est = z * self.projected_grad
                
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name:
                        grad_est += args.weight_decay * param.data

                param.data = param.data - lr * grad_est


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix (whitening).
    Used for Muon/Stiefel updates.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    
    # Ensure spectral norm < sqrt(3) for convergence
    X /= (X.norm() + eps) 
    
    if G.size(0) > G.size(1):
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X.to(G.dtype)


def zeropower_via_svd(G):
    """
    Orthogonalize G directly using SVD instead of Newton–Schulz iteration.
    Works on float16, bfloat16, float32, or float64.
    For half-precision types, we upcast to float32 before doing SVD.
    """
    assert G.ndim >= 2, "Input must be at least 2D (batched matrices allowed)"

    # make sure smaller dimension is last for SVD efficiency
    transpose_needed = G.size(-2) > G.size(-1)
    if transpose_needed:
        G = G.mT

    orig_dtype = G.dtype
    if G.dtype in (torch.float16, torch.bfloat16):
        G32 = G.float()  # upcast for numerical stability and CUDA kernel support
        U, S, Vh = torch.linalg.svd(G32, full_matrices=False)
        X = (U @ Vh).to(orig_dtype)
    else:
        U, S, Vh = torch.linalg.svd(G, full_matrices=False)
        X = U @ Vh

    if transpose_needed:
        X = X.mT
    return X


class LowDimOptimizer(object):
    """
    Low-Dimensional Zeroth-Order Optimizer (LowDim).
    Supports multiple subspace strategies (Gaussian, Polar) and update rules (SGD, Momentum, Muon).
    Handles both 2D (Linear) and 4D (Conv2d) parameter tensors.
    """
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.zo_eps = args.zo_eps
        
        # Internal State
        self.step_curr = 0
        self.P_matrices = {}
        self.momentum_RGE = {}
        self.saved_u = {}
        self.projected_grads_list = []
        
        # Cache parameters requiring grad
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        
        self.zo_random_seed = None

    def _sample_orthogonal_P(self, m, r, device, dtype):
        """
        Samples an orthogonal projection matrix P using QR decomposition.
        """
        # If the dimension m is smaller than the requested rank r, 
        # we can't have r orthogonal columns of length m. 
        # We clamp r to be at most m.
        effective_r = min(m, r)
        
        A = torch.randn(m, effective_r, device=device, dtype=torch.float32)
        Q, R = torch.linalg.qr(A, mode="reduced")

        # Randomize signs to make it Haar distributed
        signs = torch.sign(torch.diagonal(R))
        signs[signs == 0] = 1.0
        Q = Q * signs

        # Optional: downcast
        if dtype in (torch.float16, torch.bfloat16):
            return Q.to(dtype=dtype)
        elif dtype in (torch.float32, torch.float64):
            return Q.to(dtype)
        else:
            raise TypeError(f"Unsupported dtype for P matrix: {dtype}")

    def lowdim_zo_perturb_parameters(self, random_seed=None, scaling_factor=1, sample_idx=0):
        """
        Perform parameter perturbation for low-dimensional ZO gradient estimation.
        Stores the random direction u for later use in the update step.
        """
        args = self.args
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
        
        for name, param in self.named_parameters_to_optim:
            if name not in self.saved_u:
                self.saved_u[name] = {}

            # Treat anything with >= 2 dimensions as a matrix for subspace optimization
            if param.data.ndim >= 2:
                orig_shape = param.data.shape
                
                # Flatten: [d_out, d_in_flattened]
                m = orig_shape[0]
                n = param.data.numel() // m
                
                # Determine effective rank for this specific parameter
                current_rank_r = min(m, args.rank_r)

                # Check if P needs resampling
                wrong_rank_shape = False
                if name in self.P_matrices:
                    # Check if stored P matches current requirements (m, current_rank_r)
                    if self.P_matrices[name].shape != (m, current_rank_r):
                        wrong_rank_shape = True

                # Resample P if interval met, not initialized, or shape mismatch
                resample = (self.step_curr % args.step_interval == 0) or \
                           (name not in self.P_matrices) or \
                           wrong_rank_shape

                if resample:
                    P = self._sample_orthogonal_P(m, args.rank_r, param.data.device, param.data.dtype)
                    self.P_matrices[name] = P
                else:
                    P = self.P_matrices[name]

                # Ensure we use the actual rank of P (in case m < rank_r)
                actual_rank = P.shape[1]

                # Standard Gaussian sampling: u is [actual_rank, n]
                u = torch.randn(actual_rank, n, device=param.data.device, dtype=param.data.dtype)

                self.saved_u[name][sample_idx] = u

                # Perturb parameters: W' = W + P @ u * scaling
                # P is [m, r], u is [r, n] -> result is [m, n]
                perturb = (P @ u) * (scaling_factor * args.zo_eps)
                
                # Reshape perturbation back to original shape and add
                param.data.add_(perturb.view(orig_shape))

            else:
                # 1D case (no subspace)
                z = torch.randn_like(param.data)
                self.saved_u[name][sample_idx] = z 
                param.data.add_(z * (scaling_factor * args.zo_eps))

    def zo_forward(self, inputs, labels):
        """
        Get (no gradient) loss from the model.
        """
        self.model.eval()
        with torch.inference_mode():
            loss = self.model(inputs, labels)
            if self.args.n_gpu > 1:
                 loss = loss.mean()
        return loss.detach()

    def zo_step(self, inputs, labels):
        """
        Low-dimensional random gradient estimate step.
        """
        args = self.args

        self.step_curr += 1

        base_seed = np.random.randint(1000000000)
        self.zo_random_seed = base_seed 
        
        self.projected_grads_list = []
        self.saved_u = {} 

        # --- Strategy 1: Multiple Sampling ---
        if getattr(args, 'multiple_sample', False):
            num_samples = args.num_samples
            loss_baseline = self.zo_forward(inputs, labels)
            
            for i in range(num_samples):
                current_seed = base_seed + i
                self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=1, sample_idx=i)
                loss_perturbed = self.zo_forward(inputs, labels)
                grad_est = ((loss_perturbed - loss_baseline) / self.args.zo_eps).item()
                self.projected_grads_list.append(grad_est)
                self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=-1, sample_idx=i)

            return loss_baseline

        # --- Strategy 2: Single Sampling (Antithetic) ---
        else:
            num_samples = 1
            current_seed = base_seed 
            i = 0
            self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=1, sample_idx=i)
            loss1 = self.zo_forward(inputs, labels)
            self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=-2, sample_idx=i)
            loss2 = self.zo_forward(inputs, labels)
            grad_est = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()
            self.projected_grads_list.append(grad_est)
            self.lowdim_zo_perturb_parameters(random_seed=current_seed, scaling_factor=1, sample_idx=i)

            return (loss1 + loss2) / 2

    def zo_update(self):
        """
        Update parameters based on the estimated gradients.
        Supports: sgd, muon, muon_svd.
        Removed: momentum, sign, polar.
        """
        args = self.args
        lr = args.learning_rate
        opt_type = args.zo_optimizer.lower()
        num_samples = len(self.projected_grads_list)

        for name, param in self.named_parameters_to_optim:
            
            # --- 2D Parameters (Subspace Update) ---
            if param.data.ndim >= 2:
                orig_shape = param.data.shape
                m = orig_shape[0]
                n = param.data.numel() // m

                P = self.P_matrices[name]
                actual_rank = P.shape[1] 

                # lowdim_rge shape is [actual_rank, n]
                lowdim_rge = torch.zeros(actual_rank, n, device=param.data.device, dtype=param.data.dtype)
                
                # Accumulate gradients
                for i in range(num_samples):
                    u_i = self.saved_u[name][i] 
                    grad_scalar_i = self.projected_grads_list[i] 
                    # Standard accumulation (removed sign/stiefel logic)
                    lowdim_rge.add_(u_i * grad_scalar_i)
                
                lowdim_rge.div_(num_samples)

                # Apply Optimizer Logic
                if opt_type == "muon":
                    M_sign = zeropower_via_newtonschulz5(lowdim_rge)
                    G = P @ M_sign

                elif opt_type == "muon_svd":
                    M_sign = zeropower_via_svd(lowdim_rge)
                    G = P @ M_sign

                elif opt_type == "sgd":
                    G = P @ lowdim_rge
                
                else:
                    raise ValueError(f"Unsupported optimizer_type: {opt_type}. Only sgd, muon, muon_svd are supported.")

                # Weight Decay
                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        G.add_(args.weight_decay, param.data.view(m, n))

                # Update
                param.data.add_(-lr * G.view(orig_shape))

            # --- 1D Parameters (Standard Update) ---
            else:
                grad_est = torch.zeros_like(param.data)
                
                for i in range(num_samples):
                    z_i = self.saved_u[name][i]
                    grad_scalar_i = self.projected_grads_list[i]
                    grad_est.add_(z_i * grad_scalar_i)
                grad_est.div_(num_samples)

                # Removed momentum logic for 1D

                if args.weight_decay > 0:
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        grad_est.add_(args.weight_decay, param.data)

                # Standard SGD update using the actual learning rate
                param.data.add_(-1e-4* grad_est)