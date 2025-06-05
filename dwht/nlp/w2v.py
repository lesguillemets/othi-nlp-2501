"""
word2vec で色々試す
Assume:
    - {basename}_words_{SOAPF}_{parts of speech}.txt にスペースで分かち書きされた一行一文の記載がある
    - 同じ stem の .sequence_ids.txt に，各行がどの sequence id を持っていたかが記録されている
"""

from pathlib import Path

import polars as pl
from gensim.models import word2vec


def load_corpus(corpus_file: Path) -> pl.DataFrame:
    with corpus_file.open("r") as f:
        corpus: list[list[str]] = [line.strip().split(" ") for line in f.readlines()]
    with corpus_file.with_suffix(".sequence_ids.txt").open("r") as g:
        seq_ids: list[str] = [line.strip() for line in g.readlines()]
    return pl.DataFrame({"words": corpus, "seq_ids": seq_ids})


def learn(
    df: pl.DataFrame, vector_size=100, min_count=2, window=5, epochs=100
) -> word2vec.Word2Vec:
    model = word2vec.Word2Vec(
        sentences=df["words"].to_list(),
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        epochs=epochs,
    )
    return model


def run(f: Path):
    df = load_corpus(f)
    learn(df)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=Path, help="newline-separated documents of space-separated words"
    )
    args = parser.parse_args()
    run(args.file)


if __name__ == "__main__":
    main()
