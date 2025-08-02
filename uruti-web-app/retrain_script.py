import os
import pandas as pd
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset
import psycopg2
from dotenv import load_dotenv # Assuming you'll use a .env file for database credentials

# Load environment variables from a .env file
load_dotenv()

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://user:password@localhost/uruti_db')
MODEL_PATH = './models/distilbert-base-uncased' # Path to the currently deployed model
NEW_MODEL_PATH = './models/distilbert-base-uncased_retrained' # Path to save the retrained model

# Define label mapping (should match the one used during initial training)
label_map = {
    'Mentorship Needed': 0,
    'Investment Ready': 1,
    'Needs Refinement': 2
}
# Invert the map for evaluation reporting
id_to_label = {v: k for k, v in label_map.items()}


def fetch_data_from_db():
    """
    Fetches user requests from the database.
    """
    conn = None
    df = pd.DataFrame(columns=['input_text', 'predicted_label', 'timestamp'])
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT input_text, predicted_label, timestamp FROM user_request;") # Assuming table name is user_request
        data = cur.fetchall()
        df = pd.DataFrame(data, columns=['input_text', 'predicted_label', 'timestamp'])
        cur.close()
    except Exception as e:
        print(f"Error fetching data from database: {e}")
    finally:
        if conn is not None:
            conn.close()
    return df

def retrain_model():
    """
    Fetches data, retrains the model, evaluates, and saves if better.
    """
    print("Starting retraining process...")

    # 1. Fetch data
    df = fetch_data_from_db()
    if df.empty:
        print("No data fetched from database. Retraining aborted.")
        return

    # 2. Preprocess data
    df['input_text'] = df['input_text'].str.strip().fillna('').astype(str)
    # Use only the labels that are relevant for retraining
    relevant_labels = label_map.keys()
    df = df[df['predicted_label'].isin(relevant_labels)]
    df['label_id'] = df['predicted_label'].map(label_map)
    df.dropna(subset=['label_id'], inplace=True)
    df['label_id'] = df['label_id'].astype(int)

    if df.empty:
        print("No relevant data after preprocessing. Retraining aborted.")
        return

    # 3. Split data
    # Ensure enough samples per class for stratification, or handle small classes
    if len(df['label_id'].unique()) > 1 and min(df['label_id'].value_counts()) >= 2:
         train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['label_id']
        )
    else:
        print("Not enough samples for stratification. Using simple split.")
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 4. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(label_map))

    def tokenize_function(examples):
        transcripts = [str(t) if t is not None else "" for t in examples['input_text']]
        return tokenizer(transcripts, padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    tokenized_train_dataset = tokenized_train_dataset.rename_column("label_id", "labels")
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label_id", "labels")
    tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])


    # 5. Retrain model
    print("Starting model training...")
    training_args = TrainingArguments(
        output_dir='./retraining_results',
        num_train_epochs=1, # Adjust epochs as needed
        per_device_train_batch_size=8, # Adjust batch size
        per_device_eval_batch_size=8,
        logging_dir='./retraining_logs',
        logging_steps=10,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="no",
        report_to="none" # Disable reporting to external services
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics, # Need to define this function
    )

    trainer.train()
    print("Model training finished.")

    # 6. Evaluate model
    print("Evaluating retrained model...")
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # 7. Compare performance and deploy
    # This is a simplified comparison. In a real scenario, you'd compare
    # against a stored metric of the currently deployed model.
    # For this example, we'll just check if accuracy improved (if a previous metric exists)
    # or if it meets a minimum threshold.
    retrained_accuracy = eval_results.get('eval_accuracy', 0)

    # Placeholder for loading previous best metric - replace with actual logic
    previous_best_accuracy = 0.0 # Load this from a file or database

    if retrained_accuracy > previous_best_accuracy: # Simple comparison
        print(f"Retrained model performance improved (Accuracy: {retrained_accuracy:.4f} > Previous: {previous_best_accuracy:.4f}). Saving model.")
        model.save_pretrained(NEW_MODEL_PATH)
        tokenizer.save_pretrained(NEW_MODEL_PATH)
        print(f"New model saved to {NEW_MODEL_PATH}")
        # Update the MODEL_PATH variable in the main app or a config file
        # to point to the new model path for future predictions.
        # You might need a more robust deployment strategy here.
    else:
        print(f"Retrained model performance did not improve significantly (Accuracy: {retrained_accuracy:.4f} <= Previous: {previous_best_accuracy:.4f}). Not saving.")

    print("Retraining process finished.")


def compute_metrics(p):
    """
    Computes evaluation metrics.
    """
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    # Handle cases where a class might not be present in true labels or predictions
    # Use zero_division=0 to avoid errors for missing classes
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted', zero_division=0)
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted', zero_division=0)
    return {"eval_accuracy": accuracy, "eval_precision": precision, "eval_recall": recall, "eval_f1": f1}


if __name__ == "__main__":
    retrain_model()
