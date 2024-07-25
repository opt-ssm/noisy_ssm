from mamba_trainer.preprocess import preprocess
import random
from typing import Dict, Sequence
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, Subset
import transformers

def get_random_subset(dataset: Dataset, n: int) -> Subset:
    if n > len(dataset):
        raise ValueError("Requested subset size is larger than the dataset.")
    
    indices = random.sample(range(len(dataset)), n)
    return Subset(dataset, indices)

class LongRangeDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.AutoTokenizer):
        super(LongRangeDataset, self).__init__()
        data_dict = preprocess(data_path, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollator(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "input_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

class DataModule():
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str, batch_size: int):
        self.dataset = LongRangeDataset(tokenizer=tokenizer, data_path=data_path)
        if batch_size is not None:
            self.dataset = get_random_subset(self.dataset, batch_size)
        self.data_collator = DataCollator(tokenizer=tokenizer)