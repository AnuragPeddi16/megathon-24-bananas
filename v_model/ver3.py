import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from nlpaug.augmenter.word import SynonymAug
import re  # Import regex module

""" # Check for available device: MPS or CUDA
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: CUDA")
else:
    device = torch.device("cpu")
    print("Using device: CPU")
 """

device = torch.device("cpu")
print("Using device: CPU")

# Load the CSV data
data = pd.read_csv('dataset.csv')
print("Data loaded successfully.")
print(f"Number of records: {len(data)}")

# Define text preprocessing function
def preprocess_text(text):
    text = str(text).lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Apply preprocessing
data['statement'] = data['statement'].fillna('').apply(preprocess_text)
print("Text preprocessing completed.")

# Data augmentation (optional)
augmenter = SynonymAug(aug_src='wordnet')
data['statement_aug'] = data['statement'].apply(lambda x: augmenter.augment(x) if x else [x])
print("Data augmentation completed.")

# Ensure augmented statements are strings
data['statement_aug'] = data['statement_aug'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
print("Augmented statements processed.")

# Split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['statement_aug'], data['status'], test_size=0.2, random_state=42
)
print(f"Data split into training and validation sets: {len(train_texts)} training samples, {len(val_texts)} validation samples.")

# Load tokenizer and model
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(data['status'].unique()))
model.to(device)  # Move model to the appropriate device
print(f"Model '{model_name}' loaded successfully.")

# Tokenize the data
try:
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)
    print("Tokenization completed successfully.")
except Exception as e:
    print(f"Error during tokenization: {e}")

# Convert labels to integers
label_dict = {label: i for i, label in enumerate(data['status'].unique())}
train_labels = [label_dict[label] for label in train_labels]
val_labels = [label_dict[label] for label in val_labels]
print("Labels converted to integers.")

# Create torch datasets
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(device) for key, val in self.encodings.items()}  # Move to the appropriate device
        item['labels'] = torch.tensor(self.labels[idx]).to(device)  # Move to the appropriate device
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)
print("Torch datasets created successfully.")

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)   

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
print("Starting training...")
trainer.train()
print("Training completed.")

# Save model and tokenizer
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_tokenizer')
print("Model and tokenizer saved!")
