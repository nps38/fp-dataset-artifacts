import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained model and tokenizer
model_name = "./trained_model_cartography"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
error_count = 0
total_count = 0

# Load contrast set examples
with open("contrast_set.json", "r") as f:
    contrast_set = json.load(f)

# Define label mapping
label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}

# Process examples
for example in contrast_set:
    total_count += 1
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    true_label = example["label"]

    # Tokenize inputs
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=128)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label_id = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = label_mapping[predicted_label_id]

    # Print results
    if true_label.lower() != predicted_label.lower():
        error_count += 1
        print(f"Premise: {premise}")
        print(f"Hypothesis: {hypothesis}")
        print(f"True Label: {true_label}")
        print(f"Predicted Label: {predicted_label}")
        print("-" * 50)
        
print(f"Error Count: {error_count}")
print(f"Total Count: {total_count}")
