import torch
import argparse
import os
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from mamba_trainer.data import DataModule
from mamba_trainer.trainer import MambaTrainer, GradientCallback

def run(args):
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
        batch_size=args.batch_size
    )

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=optimizer,
        output_dir=args.output_dir,
        logging_dir="./logs", 
        logging_steps=1,
        save_steps=1,
        report_to="none", 
    )

    gradient_norms = []

    for step in range(t_total):
        grad_callback = GradientCaptureCallback()

        trainer = MambaTrainer(
            model=model,
            train_dataset=data_module.dataset,
            tokenizer=tokenizer,
            args=training_args,
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
