class ChatDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: transformers.AutoTokenizer):
        super(ChatDataset, self).__init__()
        data_dict = preprocess_dataset(file_path, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForChatDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    

class ChatDataModule():
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):

        self.dataset = ChatDataset(tokenizer=tokenizer, data_path=data_path)
        self.data_collator = DataCollatorForChatDataset(tokenizer=tokenizer)