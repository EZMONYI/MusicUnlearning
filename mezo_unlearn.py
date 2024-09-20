import argparse
import random
import numpy as np
from argparse import Namespace
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zo_perturb_parameters(zo_random_seed, named_parameters_to_optim, scaling_factor=1):
    """
    Perturb the parameters with random vector z.
    Input: 
    - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use zo_random_seed)
    - scaling_factor: theta = theta + scaling_factor * z * eps
    """

    # Set the random seed to ensure that we sample the same z for perturbation/update
    torch.manual_seed(zo_random_seed)
    
    for name, param in named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data + scaling_factor * z * args.zo_eps


def zo_forward(self, model, inputs):
    """
    Get (no gradient) loss from the model. Dropout is turned off too.
    """
    model.eval()
    with torch.inference_mode():
        loss = compute_loss(model, inputs)

    return loss.detach()


def zo_step(args, model, inputs):
    """
    Estimate gradient by MeZO. Return the loss from f(theta + z)
    """
    args = args

    # What parameters to optimize 
    named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            named_parameters_to_optim.append((name, param))

    # Sample the random seed for sampling z
    zo_random_seed = np.random.randint(1000000000)

    # First function evaluation
    zo_perturb_parameters(zo_random_seed, named_parameters_to_optim, scaling_factor=1)
    loss1 = zo_forward(model, inputs)

    # Second function evaluation
    zo_perturb_parameters(zo_random_seed, named_parameters_to_optim, scaling_factor=-2)
    loss2 = zo_forward(model, inputs)

    projected_grad = ((loss1 - loss2) / (2 * args.zo_eps)).item()

    # No gradient accumulation support
    assert args.gradient_accumulation_steps == 1

    # Reset model back to its parameters at start of step
    zo_perturb_parameters(scaling_factor=1)
    
    return projected_grad


def zo_update(self, model):
    """
    Update the parameters with the estimated gradients.
    """
    args = self.args

    # Reset the random seed for sampling zs
    torch.manual_seed(self.zo_random_seed)     

    for name, param in self.named_parameters_to_optim:
        # Resample z
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.data = param.data - self._get_learning_rate() * (self.projected_grad * z + args.weight_decay * param.data)
        else:
            param.data = param.data - self._get_learning_rate() * (self.projected_grad * z)

    self.lr_scheduler.step()


def train():
    pass

def main():
    args = Namespace()
    set_seed(8888)

    for train_set_id, train_samples in enumerate(train_sets):
        train()


if __name__ == "__main__": 
    main()