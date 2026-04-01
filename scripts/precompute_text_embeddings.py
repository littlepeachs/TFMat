import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    hidden = last_hidden_state * mask
    denom = mask.sum(dim=1).clamp_min(1.0)
    return hidden.sum(dim=1) / denom


def pool_hidden(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "cls":
        return last_hidden_state[:, 0]
    if pooling == "mean":
        return mean_pool(last_hidden_state, attention_mask)
    raise ValueError(f"Unknown pooling mode: {pooling}")


def build_default_output_path(output_dir: Path, input_csv: Path, text_column: str, model_tag: str, pooling: str) -> Path:
    stem = input_csv.stem
    return output_dir / f"{stem}_{text_column}_{model_tag}_{pooling}.pt"


def infer_model_tag(model_name: str) -> str:
    return model_name.rstrip("/").split("/")[-1]


def main(args):
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.text_column not in df.columns:
        raise KeyError(f"Column {args.text_column} not found in {input_csv}")
    if "material_id" not in df.columns:
        raise KeyError(f"Column material_id not found in {input_csv}")

    texts = df[args.text_column].fillna("").astype(str).tolist()
    material_ids = df["material_id"].astype(str).tolist()

    device = torch.device(args.device)
    model_tag = infer_model_tag(args.model_name)
    output_path = Path(args.output_path) if args.output_path else build_default_output_path(
        output_dir=output_dir,
        input_csv=input_csv,
        text_column=args.text_column,
        model_tag=model_tag,
        pooling=args.pooling,
    )

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    embeddings = []
    total = len(texts)
    print(f"Encoding {total} samples from {input_csv}")
    with torch.no_grad():
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)
            batch_texts = texts[start:end]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            pooled = pool_hidden(outputs.last_hidden_state, batch["attention_mask"], args.pooling)
            embeddings.append(pooled.detach().cpu())
            if (start // args.batch_size) % args.log_interval == 0:
                print(f"Processed {end}/{total}")

    embedding_tensor = torch.cat(embeddings, dim=0)
    result = {
        "material_id": material_ids,
        "text_column": args.text_column,
        "model_name": args.model_name,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "embedding_dim": int(embedding_tensor.shape[1]),
        "embeddings": embedding_tensor,
    }
    torch.save(result, output_path)
    print(f"Saved embeddings to {output_path}")
    print(f"Embedding tensor shape: {tuple(embedding_tensor.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_csv", help="Path to a CSV file containing material_id and text column")
    parser.add_argument("--text-column", default="text2", help="Text column to encode")
    parser.add_argument("--model-name", default="m3rg-iitd/matscibert", help="Hugging Face model id or local path")
    parser.add_argument("--pooling", choices=["cls", "mean"], default="mean")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:4, cpu")
    parser.add_argument("--output-dir", default="./data_text/precomputed_embeddings", help="Directory for saved pt files")
    parser.add_argument("--output-path", default=None, help="Optional explicit output path")
    parser.add_argument("--log-interval", type=int, default=10, help="Batch interval for progress logs")
    args = parser.parse_args()
    main(args)