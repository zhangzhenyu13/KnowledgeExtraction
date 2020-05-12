import math
def get_lr(fstep, opt_config):
    if opt_config['learning_rate_schedule'] == 'static':
        lr = opt_config['static_lr']
    else:
        lr = opt_config['lr_constant'] \
            * min(1.0, (fstep / opt_config['warmup_steps'])) \
            * (1 / math.sqrt(max(fstep, opt_config['warmup_steps'])))
    return lr
    