# Copyright © 2023-2024 Apple Inc.

import json
import argparse
from pathlib import Path
from datasets import load_dataset
import random

def prepare_legal_dataset(output_dir: str = "data", train_split: float = 0.8, val_split: float = 0.1):
    """
    Prepare the legal case document summarization dataset for LoRA fine-tuning.
    
    Args:
        output_dir: Directory to save the processed data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation (test will be the remainder)
    """
    
    print("Loading legal case document summarization dataset...")
    dataset = load_dataset("joelniklaus/legal_case_document_summarization")
    
    # Convert to list for easier processing
    data = []
    for split in dataset:
        for item in dataset[split]:
            # Format the text for summarization task
            # Using the format: "Document: {text}\n\nSummary: {summary}"
            formatted_text = f"Document: {item['text']}\n\nSummary: {item['summary']}"
            data.append({"text": formatted_text})
    
    # Shuffle the data
    random.shuffle(data)
    
    # Split the data
    total_size = len(data)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save the splits
    with open(output_path / "train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open(output_path / "valid.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    
    with open(output_path / "test.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Dataset prepared successfully!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Total samples: {total_size}")
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare legal dataset for LoRA fine-tuning")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save the processed data"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data for validation"
    )
    
    args = parser.parse_args()
    prepare_legal_dataset(args.output_dir, args.train_split, args.val_split) 