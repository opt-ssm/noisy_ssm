from transformers import Trainer, TrainerCallback
import torch
import os
from mamba_trainer.preprocess import preprocess


class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        return lm_loss


    def save_model(self, output_dir, _internal_call):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.tokenizer.save_pretrained(output_dir)



class GradientCallback(TrainerCallback):
    def __init__(self):
        self.gradient_norm = 0

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        gradient_vector = []

        for param in model.parameters():
            if param.grad is not None:
                gradient_vector.append(param.grad.view(-1))
        
        gradient_vector = torch.cat(gradient_vector)
        self.gradient_norms = gradient_vector.norm().item()