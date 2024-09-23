import argparse
import random
import numpy as np
from argparse import Namespace
import torch
from .model.songmass import build_songmass

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zo_perturb_parameters(zo_random_seed, named_parameters_to_optim, args, scaling_factor=1):

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


def zo_step(model, inputs, args):

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

    # Reset model back to its parameters at start of step
    zo_perturb_parameters(scaling_factor=1)

    # unlearn
    projected_grad = -projected_grad
    
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


def main():

    ckpt_path = './songmass.pth'
    model = build_songmass()

    args = Namespace(num_train_epochs)
    set_seed(8888)
    
    for epoch in range(num_train_epochs):
        for idx, sample in enumerate(train_loader)
            zo_step(model, sample)

        zo_update(model)

    return TrainOutput(self.state.global_step, train_loss, metrics)


if __name__ == "__main__": 
    main()