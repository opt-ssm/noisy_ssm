from transformers import Trainer, TrainerCallback
import torch
import os
import json
from mamba_trainer.preprocess import preprocess
import logging
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Ensure the inputs are handled correctly
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        if return_outputs:
            return (loss, logits)
        return loss


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
        self.total_norm = 0.0

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if 'loss' in state.log_history[-1]:
            train_loss = state.log_history[-1]['loss']
            logger.info(f"Training Loss at step {state.global_step}: {train_loss}")
            
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if 'eval_loss' in state.log_history[-1]:
            val_loss = state.log_history[-1]['eval_loss']
            logger.info(f"Validation Loss at step {state.global_step}: {val_loss}")
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        # Register hook for gradient calculation
        if model:
            for p in model.parameters():
                if p.requires_grad:
                    p.register_hook(self.get_grad_hook())

    def get_grad_hook(self):
        def hook(grad):
            grad_norm = grad.data.norm(2)
            self.total_norm += grad_norm.item() ** 2
        return hook
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self.total_norm = self.total_norm ** 0.5
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
        norm_data["norm"].append(self.total_norm)
        
        # Write updated norms back to the file
        with open(self.norm_file, "w") as f:
            json.dump(norm_data, f)
        
        logger.info(f"Gradient Norm at step {state.global_step}: {self.total_norm}")
        # Reset total norm for the next step
        self.total_norm = 0.0