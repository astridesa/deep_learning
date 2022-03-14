from transformers import *
from datasets import *
import numpy as np



def encode(example):
    tokenizer = BertTokenizer.from_pretrained('./model/checkpoint-3500/vocab.txt')
    max_length = 300
    return tokenizer(example['tweet'], add_special_tokens=True,
                     padding='max_length', truncation=True, max_length=max_length)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    metric = load_metric('accuracy', 'f1')
    return metric.compute(predictions=predictions, references=labels)


def main():
    dataset = load_dataset('tweets_hate_speech_detection', split='train')
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train'].shuffle(seed=21).select(range(100))
    test_dataset = dataset['test'].shuffle(seed=21).select(range(1000))
    tokenized_train_dataset = train_dataset.map(encode, batched=True)
    tokenized_test_dataset = test_dataset.map(encode, batched=True)
    model_path = './model'
    model = BertForSequenceClassification.from_pretrained('./model/checkpoint-3500', num_labels=2)
    training_args = TrainingArguments(
        output_dir=model_path,
        evaluation_strategy="steps",
        num_train_epochs=100,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        load_best_model_at_end=True,
        save_total_limit=3,
        gradient_accumulation_steps=8,
        save_steps=1000,
        logging_steps=1000
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()


if __name__ == "__main__":
    main()










