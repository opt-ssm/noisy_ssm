import torch
import argparse
import os
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from trainer.data import DataModule
from mamba_trainer.trainer import MambaTrainer, GradientCaptureCallback

def run(args):

    model_directory = args.output_dir
    model_file_path = os.path.join(model_directory, 'pytorch_model.bin')
    if os.path.exists(model_file_path):
        model = MambaLMHeadModel.from_pretrained(model_file_path, dtype=torch.bfloat16, device="cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_file_path)
    else:
        model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model.save_pretrained(model_directory)
        tokenizer.save_pretrained(model_directory)

    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True)

    data_module = DataModule(
        tokenizer=tokenizer,
        data_path=args.data_path,
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
    parser.add_argument("--train_data", type=str, default="./data/listops_500-1000/train_data.tsv")
    parser.add_argument("--output_dir", type=str, default="./model")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    run(args)
