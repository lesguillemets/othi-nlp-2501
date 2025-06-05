"""
word2vec でクラスタリング試す
"""

from pathlib import Path
import typing
from sys import stdout

import dwht.nlp.w2v as w2v
from gensim.models.word2vec import Word2Vec


def run(f: Path, command: str):
    if command == "try_mincounts":
        try_mincounts(f)
    elif command == "try_conditions":
        try_conditions(f)
    else:
        save_model_with_defaults(f)


def save_model_with_defaults(f: Path):
    df = w2v.load_corpus(f)
    opt = {"vector_size": 200, "window": 4, "epochs": 150}
    model = w2v.learn(df, **opt)
    model.save(
        str(
            f.with_suffix(
                f".model.vs{opt['vector_size']:03d}_w{opt['window']}_e{opt['epochs']:03d}_minc2.w2v.model"
            ).resolve()
        )
    )
    model.wv.save(
        f".model.vs{opt['vector_size']:03d}_w{opt['window']}_e{opt['epochs']:03d}_minc2.w2v.kv"
    )
    try_model(model, stdout)


def try_mincounts(f: Path):
    df = w2v.load_corpus(f)
    for mincount in range(10):
        model = w2v.learn(df, vector_size=200, window=4, epochs=200, min_count=mincount)
        with f.with_suffix(f".result.vs_200_w4_e200_mincount{mincount:02d}.w2v").open(
            "a"
        ) as fl:
            try_model(model, fl)


def try_conditions(f: Path):
    df = w2v.load_corpus(f)
    for vector_size in [20, 30, 50, 100, 200]:
        for window in [2, 3, 4, 5, 6]:
            for epochs in [10, 20, 50, 100, 150, 200]:
                model = w2v.learn(
                    df, vector_size=vector_size, window=window, epochs=epochs
                )
                with f.with_suffix(
                    f".result.vs{vector_size:03d}_w{window}_e{epochs:03d}.w2v"
                ).open("a") as fl:
                    try_model(model, fl)


def try_model(model: Word2Vec, write_to: typing.IO):
    print(model.wv["笑顔"], file=write_to)
    for word in ["表情", "不安", "家族", "母親", "友人", "食事", "話す"]:
        ret = model.wv.most_similar(positive=[word])
        print(f"similar to {word}:", file=write_to)
        for item in ret:
            print(f"\t> {item[0]}, {item[1]}", file=write_to)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=Path, help="newline-separated documents of space-separated words"
    )
    parser.add_argument(
        "--command",
        choices=["try_conditions", "try_mincounts", "default"],
        default="default",
    )
    args = parser.parse_args()
    run(args.file, args.command)


if __name__ == "__main__":
    main()
