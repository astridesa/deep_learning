from tokenizer import Tokenizer
from transformers import *
from torch.utils.data import Dataset
import torch
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class WikiTextDataset(Dataset):
    def __init__(self, data_path, max_length=512):
        with open(data_path, 'r') as f:
            data = f.readlines()
        self.tokenizer = BertTokenizer.from_pretrained('model/checkpoint-3500/vocab.txt')
        self.data = data
        self.max_length = max_length

    def __getitem__(self, index):
        sentence = self.data[index]
        token = self.tokenizer(sentence, add_special_tokens=True,
                               padding='max_length', truncation=True, max_length=self.max_length)
        return {
            'input_ids': torch.tensor(token['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(token['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(token['attention_mask'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


def main():
    tokenizer = BertTokenizer.from_pretrained('model/checkpoint-3500/vocab.txt')
    max_length = 300
    model_config = BertConfig(vocab_size=30133, max_position_embeddings=max_length)
    model = BertForMaskedLM(config=model_config)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.2
    )
    model_path = './model'
    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="steps",
        overwrite_output_dir=True,
        num_train_epochs=20,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=64,
        logging_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
        # save_total_limit=3
    )

    train_set = WikiTextDataset('./data/wikitext-2/train.txt', max_length)
    valid_set = WikiTextDataset('./data/wikitext-2/valid.txt', max_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        eval_dataset=valid_set
    )
    trainer.train()


if __name__ == '__main__':
    main()




