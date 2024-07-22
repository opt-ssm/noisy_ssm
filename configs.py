import random

import numpy as np
import peft
import torch
import transformers


class Constants:
    # general
    lr = 3e-4
    batch_size = 64
    # Mamba 1 models
    mamba_1_130m = "state-spaces/mamba-130m-hf"
    mamba_1_370m = "state-spaces/mamba-370m-hf"
    mamba_1_790m = "state-spaces/mamba-790m-hf"
    mamba_1_4b = "state-spaces/mamba-1.4b-hf"
    # Mamba 2 models
    mamba_2_4b = "state-spaces/mamba-1.4b-hf"
    mamba_2_8b = "state-spaces/mamba-2.8b-hf"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_one_device(device_no: int):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_no}")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(device_no))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
