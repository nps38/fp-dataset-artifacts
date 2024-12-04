from transformers import Trainer
from collections import defaultdict
import torch

def train_model(model, tokenizer, train_dataset, eval_dataset, training_args, compute_metrics=None):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    return trainer


def train_model_with_cartography(model, tokenizer, train_dataset, eval_dataset, training_args, compute_metrics=None):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions, labels = [], []
    for batch in trainer.get_train_dataloader():
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions.append(logits.softmax(dim=-1))
        labels.append(batch["labels"])

    # Compute confidence, variability, and correctness
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    scores = defaultdict(list)
    for i in range(len(predictions)):
        confidence = predictions[i, labels[i]].item()
        variability = predictions[i].std().item()
        correctness = int(predictions[i].argmax() == labels[i])
        scores["confidence"].append(confidence)
        scores["variability"].append(variability)
        scores["correctness"].append(correctness)

    # Categorize samples
    easy_indices = [i for i, score in enumerate(scores["correctness"]) if score == 1 and scores["confidence"][i] > 0.9]
    hard_indices = [i for i, score in enumerate(scores["correctness"]) if score == 0]
    ambiguous_indices = list(set(range(len(predictions))) - set(easy_indices) - set(hard_indices))

    # Re-filter the dataset
    def filter_dataset(dataset, indices):
        return dataset.select(indices)

    refined_train_dataset = filter_dataset(train_dataset, ambiguous_indices + hard_indices)

    # Retrain the model with the refined dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=refined_train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer
