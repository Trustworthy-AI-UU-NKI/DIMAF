import torch.optim as optim
from transformers import (get_constant_schedule_with_warmup, 
                         get_linear_schedule_with_warmup, 
                         get_cosine_schedule_with_warmup)

def get_optim(args, model):
    """Get the specified optimizer, adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/utils.py """
    def exclude(
        n, p): return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n

    def include(n, p): return not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(
        n, p) and p.requires_grad]
    parameters = [
        {"params": gain_or_bias_params, "weight_decay": 0.},
        {"params": rest_params, "weight_decay": args.wd},
    ]
    optimizer = optim.AdamW(parameters, lr=args.lr)
    
    return optimizer

def get_lr_scheduler(args, optimizer, n_data):
    """Get the specified learning rate scheduler, adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/utils.py"""
    scheduler_name = args.lr_scheduler
    warmup_steps = args.warmup_steps
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs
    assert not (warmup_steps > 0 and warmup_epochs > 0), "Cannot have both warmup steps and epochs"
 
    if warmup_steps > 0:
        warmup_steps = warmup_steps
    elif warmup_epochs > 0:
        warmup_steps = warmup_epochs * n_data
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps)
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_data * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=n_data * epochs,
        )
    return lr_scheduler

def list_to_device(list, device):
    """ Put all items in a list to the device (cpu or gpu). """
    return [item.to(device) for item in list]

class LoggingMeter(object):
    """Computes and stores the average and current value, adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/utils.py"""

    def __init__(self, name):
        self.name = name
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
    
def log_results(writer, results, epoch, mode='train'):
    for item, value in results.items():
        writer.add_scalar(f'{mode}_{item}', value, epoch)