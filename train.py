from transformers import Trainer, TrainingArguments
from collections import defaultdict
import torch
from transformers import pipeline
from datasets import Dataset
import numpy as np


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


class CustomTrainer(Trainer):
    def __init__(self, *args, sample_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_weights = sample_weights  # Add sample weights to the trainer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute the custom loss with support for unexpected arguments.
        """
        labels = inputs.get("labels")
        
        # Create the sample weights tensor from the sample_weights list provided
        sample_weights = torch.tensor(
            [self.sample_weights[i] for i in range(len(labels))],
            device=model.device,
            dtype=torch.float32  # Ensure the type matches
        )
        # print(f"Sample Weights: {sample_weights[:10]}")  # Print the first 10 weights to check

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Custom loss: Weighted cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_fct(logits, labels)

        # Weighted loss
        weighted_loss = torch.mean(losses * sample_weights)

        return (weighted_loss, outputs) if return_outputs else weighted_loss


# Training with Hard Example Mining
def train_model_cartography(model, tokenizer, train_dataset, eval_dataset, training_args, compute_metrics=None):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        learning_rate=1e-4,  # Keep learning rate as 1e-4
        lr_scheduler_type="cosine_with_restarts",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        num_train_epochs=training_args.num_train_epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        warmup_steps=500,
        gradient_accumulation_steps=2,
    )

    # Step 1: Train the initial model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Step 2: Generate predictions to categorize samples
    predictions, labels = [], []
    for batch in trainer.get_train_dataloader():
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        # print(f"Logits: {logits[:10]}")  # Print the first 10 logits
        predictions.append(logits.softmax(dim=-1))
        labels.append(batch["labels"])

    # Categorize samples
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    scores = defaultdict(list)
    for i in range(len(predictions)):
        confidence = predictions[i, labels[i]].item()
        correctness = int(predictions[i].argmax() == labels[i])
        scores["confidence"].append(confidence)
        scores["correctness"].append(correctness)

    # Define weights for ambiguous and hard samples
    hard_indices = [i for i, score in enumerate(scores["correctness"]) if score == 0]
    ambiguous_indices = [
        i for i, score in enumerate(scores["correctness"]) if score == 1 and scores["confidence"][i] < 0.6
    ]

    # Assign weights to the training dataset
    weights = np.ones(len(train_dataset))
    for idx in hard_indices:
        weights[idx] = 2.0  # High priority for hard examples
    for idx in ambiguous_indices:
        weights[idx] = 1.5  # Medium priority for ambiguous examples
    weights /= weights.sum()  # Normalize weights to sum to 1

    # Step 3: Use CustomTrainer with sample weights
    custom_trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        sample_weights=weights,  # Pass the weights here
    )

    custom_trainer.train()
    return custom_trainer