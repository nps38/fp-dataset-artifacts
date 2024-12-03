from transformers import Trainer

def train_model(model, tokenizer, train_dataset, eval_dataset, training_args, compute_metrics=None):
    from collections import defaultdict
    import numpy as np
    import torch

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Perform initial training pass to collect predictions
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

    # Save the final model
    trainer.save_model(training_args.output_dir)

    return trainer


