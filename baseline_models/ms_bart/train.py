import torch
import argparse
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from dataloader import SmilesSet
def main(args):
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    print(f"raw tokenizer size: {len(tokenizer)}")
    model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path)
    
    dataset = SmilesSet(tokenizer, split="all")
    data = dataset.__getitem__(0)
    print(data)
    tokenizer = dataset.get_tokenizer()
    print(f"after add tokenizer size: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = SmilesSet(tokenizer, split="train")
    val_dataset = SmilesSet(tokenizer, split="val")

    # configure train parameters
    training_args = TrainingArguments(
        output_dir='./results',              # output directory
        learning_rate=5e-5,                  # learning rate
        per_device_train_batch_size=16,       # train batch size
        gradient_accumulation_steps=8,
        fp16=True,
        num_train_epochs=20,                  # train epochs
        logging_dir='./logs/pfas',                # log directory
        logging_steps=1,
        eval_steps=1,
        evaluation_strategy="epoch",
    )

    # use Trainer to train
    trainer = Trainer(
        model=model,                         # fine-tuned model
        args=training_args,                  # train parameters
        train_dataset=train_dataset,  # train dataset
        eval_dataset=val_dataset
    )
    trainer.train()
    model.save_pretrained('./fine_tuned_bart_pfas')
    tokenizer.save_pretrained('./fine_tuned_bart_pfas')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="facebook/bart-base")
    
    args = parser.parse_args()
    main(args)
