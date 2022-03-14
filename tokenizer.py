from transformers import *
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import json

# choose full sentence
def clean_dataset(in_path, out_path):
    with open(in_path, 'r') as f:
        lines = f.readlines()
    paragraphs = [line.strip() for line in lines if '.' in line]
    with open(out_path, 'w') as w:
        for para in paragraphs:
            w.write(para + '\n')


# train a new vocabulary for wikitext-2
def get_tokenizer(data_path):
    special_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]"
    ]
    files = ['./data/wikitext-2/train.txt', './data/wikitext-2/valid.txt', './data/wikitext-2/test.txt']
    max_length = 512
    max_vocab_size = 30522
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizers = Whitespace()
    trainer = BpeTrainer(special_tokens=special_tokens)
    tokenizer.train(files=files, trainer=trainer)
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.save(data_path)


if __name__ == '__main__':
    # clean_dataset('./wikitext-2/wiki.train.tokens', './wikitext-2/train.txt')
    # get_tokenizer('./wikitext-2/wikitext-2.json')
    writer = open('model/checkpoint-3500/vocab.txt', 'w')
    with open('./data/wikitext-2/wikitext-2.json', 'r') as f:
        data = json.load(f)
        for token in list(data['model']['vocab'].keys()):
            writer.write(token)
            writer.write('\n')








