from transformers import Trainer, TrainerCallback
import torch
import os
import json
from mamba_trainer.preprocess import preprocess
import matplotlib.pyplot as plt
import logging
from IPython import display
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
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
    def __init__(self, norm_file: str = "gradient_norms.json"):
        self.norm_file = norm_file
        self.step = 0
        self.total_norm = 0.0
        self.gradients = []
        self.losses = []
        self.norms = []
        self.steps = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if 'loss' in state.log_history[-1]:
            train_loss = state.log_history[-1]['loss']
            logger.info(f"Training Loss at step {self.step}: {train_loss}")
            self.losses.append(train_loss)
        self.update_plot()
            
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self.step += 1
        self.steps.append(self.step)
        self.gradients = []
        if model:
            for p in model.parameters():
                if p.requires_grad:
                    p.register_hook(self.get_grad_hook())

    def get_grad_hook(self):
        def hook(grad):
            grad_norm = grad.data.norm(2)
            print(f"Type of self.gradients before appending: {type(self.gradients)}")
            self.gradients.append(grad.detach().reshape(-1).clone())
            self.total_norm += grad_norm.item() ** 2
        return hook
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        self.gradients = torch.cat(self.gradients)
        self.total_norm = self.total_norm ** 0.5
        if os.path.exists(self.norm_file):
            try:
                with open(self.norm_file, "r") as f:
                    norm_data = json.load(f)
            except json.JSONDecodeError:
                norm_data = {"norm": []}
        else:
            norm_data = {"norm": []}
        
        norm_data["norm"].append(self.total_norm)

        with open(self.norm_file, "w") as f:
            json.dump(norm_data, f)
        
        logger.info(f"Gradient Norm at step {self.step}: {self.total_norm}")
        self.norms.append(self.total_norm)
        self.total_norm = 0.0

    def update_plot(self):
        display.clear_output(wait=True)
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.steps, self.losses, label='Training Loss', color='blue')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Steps')
        plt.legend()
        plt.grid()

        plt.show()
