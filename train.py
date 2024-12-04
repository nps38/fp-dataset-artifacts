from transformers import Trainer, TrainingArguments
from collections import defaultdict
import torch
import torch.nn as nn

def train_model(model, tokenizer, train_dataset, eval_dataset, training_args, compute_metrics=None):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.1

    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        evaluation_strategy=training_args.evaluation_strategy,
        learning_rate=training_args.learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        num_train_epochs=training_args.num_train_epochs,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_dir=training_args.logging_dir,
        logging_steps=training_args.logging_steps,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        eval_steps=training_args.eval_steps,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Perform initial training pass
    trainer.train()

    # Dataset cartography: collect metrics
    predictions = []
    labels = []
    for batch in trainer.get_train_dataloader():
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions.append(logits.softmax(dim=-1))
        labels.append(batch['labels'])

    # Concatenate predictions and labels
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    # Compute dataset cartography metrics
    scores = defaultdict(list)
    for i in range(len(predictions)):
        confidence = predictions[i, labels[i]].item()
        variability = predictions[i].std().item()
        correctness = int(predictions[i].argmax() == labels[i])
        scores['confidence'].append(confidence)
        scores['variability'].append(variability)
        scores['correctness'].append(correctness)

    # Categorize samples
    easy_indices = [i for i, score in enumerate(scores['correctness']) if score == 1 and scores['confidence'][i] > 0.9]
    hard_indices = [i for i, score in enumerate(scores['correctness']) if score == 0]
    ambiguous_indices = list(set(range(len(predictions))) - set(easy_indices) - set(hard_indices))

    # Filter dataset by indices
    def filter_dataset(dataset, indices):
        return dataset.select(indices)

    # Optionally, re-filter train_dataset (e.g., keep easy samples only)
    # train_dataset = filter_dataset(train_dataset, easy_indices)

    # Re-initialize Trainer with filtered dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
    trainer.save_model(training_args.output_dir)
    return trainer