from transformers import Trainer

def train_model(model, tokenizer, train_dataset, eval_dataset, training_args, compute_metrics=None):
    # Initialize Trainer
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

