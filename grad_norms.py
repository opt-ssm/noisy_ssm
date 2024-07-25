import torch
import argparse
import os
from numba import cuda
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from mamba_trainer.data import DataModule
from mamba_trainer.trainer import MambaTrainer, GradientCallback


def print_selected_gpu(device_id):
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    print(f"Selected GPU {device_id}: {gpu_name}, Memory: {gpu_memory:.2f} GB")

def run(args):
    device_id=0
    cuda.select_device(device_id)
    print_selected_gpu(device_id)
    cuda.close()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(DEVICE)

    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    data_module = DataModule(
        tokenizer=tokenizer,
        data_path=args.train_data,
        batch_size=64
    )

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        logging_dir="./logs", 
        logging_steps=1,
        save_steps=1,
        report_to="none", 
    )

    gradient_norms = []

    for step in range(1000):
        grad_callback = GradientCallback()

        trainer = MambaTrainer(
            model=model,
            train_dataset=data_module.dataset,
            tokenizer=tokenizer,
            args=training_args,
            optimizers=(optimizer, None),
            data_collator=data_module.data_collator,
            callbacks=[grad_callback]
        )

        trainer.train()
        gradient_norms.append(grad_callback.gradient_norm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-1.4b")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_data", type=str, default="./data/listops_500-1000/train.tsv")
    parser.add_argument("--output_dir", type=str, default="model")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)
