# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import utils as lora_utils
from mlx.utils import tree_flatten
from models import LoRALinear

# Disable output buffering to see print statements in real-time
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)


def build_parser():
    parser = argparse.ArgumentParser(description="Legal LoRA or QLoRA finetuning for document summarization.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    # Generation args
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="The maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        help="The prompt for generation",
        default=None,
    )

    # Training args
    parser.add_argument(
        "--train",
        action="store_true",
        help="Do training",
    )
    parser.add_argument(
        "--add-eos-token",
        type=int,
        default=1,
        help="Enable add_eos_token for tokenizer",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="Directory with {train, valid, test}.jsonl files",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument(
        "--iters", type=int, default=1000, help="Iterations to train for."
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=25,
        help="Number of validation batches, -1 uses the entire validation set.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Adam learning rate."
    )
    parser.add_argument(
        "--steps-per-report",
        type=int,
        default=10,
        help="Number of training steps between loss reporting.",
    )
    parser.add_argument(
        "--steps-per-eval",
        type=int,
        default=200,
        help="Number of training steps between validations.",
    )
    parser.add_argument(
        "--resume-adapter-file",
        type=str,
        default=None,
        help="Load path to resume training with the given adapter weights.",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        default="legal_adapters.npz",
        help="Save/load path for the trained adapter weights.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save the model every N iterations.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on the test set after training",
    )
    parser.add_argument(
        "--test-batches",
        type=int,
        default=500,
        help="Number of test set batches, -1 uses the entire test set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    return parser


class Dataset:
    """
    Light-weight wrapper to hold lines from a jsonl file
    """

    def __init__(self, path: Path, key: str = "text"):
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(l) for l in fid]
        self._key = key

    def __getitem__(self, idx: int):
        return self._data[idx][self._key]

    def __len__(self):
        return len(self._data)


def load(args):
    def load_and_check(name):
        dataset_path = Path(args.data) / f"{name}.jsonl"
        try:
            return Dataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset {dataset_path} ({e})")
            raise

    names = ("train", "valid", "test")
    train, valid, test = (load_and_check(n) for n in names)

    if args.train and len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if args.train and len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if args.test and len(test) == 0:
        raise ValueError(
            "Test set not found or empty. Must provide test set for evaluation."
        )
    return train, valid, test


def loss(model, inputs, targets, lengths):
    # Run model on inputs
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    # Mask padding tokens
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def iterate_batches(dset, tokenizer, batch_size, train=False):
    # Shuffle indices
    while True:
        indices = np.arange(len(dset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices), batch_size):
            # Encode batch
            batch_indices = indices[i : i + batch_size]
            batch = [dset[j] for j in batch_indices]
            encoded = tokenizer(
                batch,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            inputs = encoded["input_ids"]
            targets = encoded["input_ids"].copy()

            # Remove padding tokens from targets
            for j, seq_len in enumerate(encoded["attention_mask"].sum(axis=1)):
                targets[j, seq_len:] = -100

            yield inputs, targets, encoded["attention_mask"].sum(axis=1)


def evaluate(model, dataset, loss, tokenizer, batch_size, num_batches):
    all_losses = []
    ntokens = 0
    for i, (inputs, targets, lengths) in enumerate(iterate_batches(dataset, tokenizer, batch_size)):
        if num_batches > 0 and i >= num_batches:
            break
        batch_loss, batch_ntokens = loss(model, inputs, targets, lengths)
        all_losses.append(batch_loss)
        ntokens += batch_ntokens

    return mx.mean(mx.array(all_losses)), ntokens


def train(model, train_set, val_set, optimizer, loss, tokenizer, args):
    # Create value and grad function for loss
    loss_value_and_grad = mx.value_and_grad(loss)

    losses = []
    tic = time.perf_counter()

    for iteration, (inputs, targets, lengths) in enumerate(
        iterate_batches(train_set, tokenizer, args.batch_size, train=True)
    ):
        if iteration >= args.iters:
            break

        # Forward pass and gradient computation
        (lvalue, ntokens), grads = loss_value_and_grad(model, inputs, targets, lengths)

        # Update the model
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Record loss
        losses.append(lvalue)

        # Report training loss if needed
        if (iteration + 1) % args.steps_per_report == 0:
            train_loss = mx.mean(mx.array(losses))
            print(
                f"Iter {iteration + 1}: Train loss: {train_loss:.3f}, "
                f"It/sec: {args.steps_per_report / (time.perf_counter() - tic):.3f}"
            )
            losses = []
            tic = time.perf_counter()

        # Evaluate validation loss if needed
        if (iteration + 1) % args.steps_per_eval == 0:
            val_loss, val_ntokens = evaluate(
                model, val_set, loss, tokenizer, args.batch_size, args.val_batches
            )
            print(
                f"Iter {iteration + 1}: "
                f"Val loss: {val_loss:.3f}, "
                f"Val ppl: {math.exp(val_loss):.3f}"
            )

        # Save adapter weights if needed
        if (iteration + 1) % args.save_every == 0:
            mx.savez(
                args.adapter_file,
                **dict(tree_flatten(model.parameters())),
            )


def generate(model, prompt, tokenizer, args):
    print(args.prompt, end="", flush=True)
    prompt_tokens = tokenizer.encode(args.prompt)
    prompt = mx.array(prompt_tokens)

    def generate_step():
        temp = args.temp

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, cache = model(prompt[None, ...])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = model(y[None, ...], cache)
            y = sample(logits[:, -1, :])
            yield y

    tokens_generated = 0
    for token in generate_step():
        print(tokenizer.decode([token.item()]), end="", flush=True)
        tokens_generated += 1
        if tokens_generated >= args.max_tokens:
            break
    print()


def main():
    seed = 0
    np.random.seed(seed)
    mx.random.seed(seed)

    parser = build_parser()
    args = parser.parse_args()

    # Load the tokenizer and model
    model, tokenizer, config = lora_utils.load(args.model)
    
    if args.add_eos_token:
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loaded pretrained model and tokenizer")

    # Load datasets
    train_set, val_set, test_set = load(args)

    if args.train:
        print("Training ...")
        # Freeze all layers other than LORA linears
        model.freeze()
        for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
            l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)

        # Resume training if requested
        if args.resume_adapter_file is not None:
            print(f"Loading pretrained adapters from {args.resume_adapter_file}")
            model.load_weights(args.resume_adapter_file)

        p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
        print(f"Total parameters {p:.3f}M")
        p = sum(v.size for _, v in tree_flatten(model.parameters()) if "lora" in str(type(v))) / 10**6
        print(f"Trainable parameters {p:.3f}M")

        # Training
        optimizer = optim.Adam(learning_rate=args.learning_rate)
        train(model, train_set, val_set, optimizer, loss, tokenizer, args)

        # Save the final adapters
        mx.savez(
            args.adapter_file,
            **dict(tree_flatten(model.parameters())),
        )

    if args.test:
        print("Testing ...")
        # Load the trained adapters
        model.load_weights(args.adapter_file)
        test_loss, test_ntokens = evaluate(
            model, test_set, loss, tokenizer, args.batch_size, args.test_batches
        )
        test_ppl = math.exp(test_loss)
        print(f"Test loss: {test_loss:.3f}")
        print(f"Test perplexity: {test_ppl:.3f}")

    if args.prompt is not None:
        print("Generating ...")
        # Load the trained adapters
        model.load_weights(args.adapter_file)
        generate(model, args.prompt, tokenizer, args)


if __name__ == "__main__":
    main() 