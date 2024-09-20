import argparse
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
    """
    Perturb the parameters with random vector z.
    Input: 
    - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
    - scaling_factor: theta = theta + scaling_factor * z * eps
    """

    # Set the random seed to ensure that we sample the same z for perturbation/update
    torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)
    
    for name, param in self.named_parameters_to_optim:
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        param.data = param.data + scaling_factor * z * self.args.zo_eps


def zo_forward(self, model, inputs):
    """
    Get (no gradient) loss from the model. Dropout is turned off too.
    """
    model.eval()
    if self.args.non_diff:
        # Non-differentiable objective (may require autoregressive generation)
        return self.zo_forward_nondiff(model, inputs)

    with torch.inference_mode():
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            # Warning: this is copied from the original Huggingface Trainer. Untested.
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
    return loss.detach()


def zo_step(self, model, inputs):
    """
    Estimate gradient by MeZO. Return the loss from f(theta + z)
    """
    args = self.args

    # What parameters to optimize 
    self.named_parameters_to_optim = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            self.named_parameters_to_optim.append((name, param))

    # Sample the random seed for sampling z
    self.zo_random_seed = np.random.randint(1000000000)

    # First function evaluation
    self.zo_perturb_parameters(scaling_factor=1)
    loss1 = self.zo_forward(model, inputs)

    # Second function evaluation
    self.zo_perturb_parameters(scaling_factor=-2)
    loss2 = self.zo_forward(model, inputs)

    self.projected_grad = ((loss1 - loss2) / (2 * self.args.zo_eps)).item()

    # No gradient accumulation support
    assert self.args.gradient_accumulation_steps == 1

    # Reset model back to its parameters at start of step
    self.zo_perturb_parameters(scaling_factor=1)
    
    return loss1


def main():
    args = 1 # TODO

    set_seed(args.seed)
# train
    for train_set_id, train_samples in enumerate(train_sets):
        train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

        # Sample eval samples
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
        else:
            eval_samples = task.valid_samples

        if args.trainer != "none":
            if args.num_dev is not None:
                # Dev samples
                dev_samples = train_samples[-args.num_dev:] 
                train_samples = train_samples[:-args.num_dev]
            else:
                dev_samples = None

            # Training
            framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples)
            

            if not args.no_eval:
                metrics = framework.evaluate([], eval_samples) # No in-context learning if there is training
                if dev_samples is not None:
                    dev_metrics = framework.evaluate([], dev_samples) 
                    for m in dev_metrics:
                        metrics["dev_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                metrics = framework.evaluate(train_samples, eval_samples)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" +  result_file_tag(args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)



if __name__ == "__main__": 
    main()