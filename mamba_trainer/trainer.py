from transformers import Trainer, TrainerCallback
import torch
import os
import json
from mamba_trainer.preprocess import preprocess
import logging
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


    def save_model(self, output_dir, _internal_call=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
        self.model.config.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_model(self, output_dir):
        model_load_path = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(model_load_path):
            self.model.load_state_dict(torch.load(model_load_path))
            self.tokenizer.from_pretrained(output_dir)
            print(f"Loaded model from {model_load_path}")
        else:
            print(f"No checkpoint found at {model_load_path}")



class GradientCallback(TrainerCallback):
    def __init__(self, norm_file="gradient_norms.json"):
        self.norm_file = norm_file

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if 'loss' in state.log_history[-1]:
            train_loss = state.log_history[-1]['loss']
            logger.info(f"Training Loss at step {state.global_step}: {train_loss}")
            
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if 'eval_loss' in state.log_history[-1]:
            val_loss = state.log_history[-1]['eval_loss']
            logger.info(f"Validation Loss at step {state.global_step}: {val_loss}")
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if model:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # Read existing norms from the file, handling empty or malformed files
            if os.path.exists(self.norm_file):
                try:
                    with open(self.norm_file, "r") as f:
                        norm_data = json.load(f)
                except json.JSONDecodeError:
                    norm_data = {"norm": []}
            else:
                norm_data = {"norm": []}
            
            # Append the new norm
            norm_data["norm"].append(total_norm)
            
            # Write updated norms back to the file
            with open(self.norm_file, "w") as f:
                json.dump(norm_data, f)
            
            logger.info(f"Gradient Norm at step {state.global_step}: {total_norm}")