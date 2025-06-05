"""
解析前のいろんな前処理
"""

from dwht.base import timestamp
from dwht.carte.load import load_cartes
from dwht.nlp.preprocess import do_tokenize, save_tokenized, tokenize_and_save_excel

from pathlib import Path
import unicodedata
from typing import Optional
import argparse

import polars as pl
import sudachipy.tokenizer
import sudachipy.dictionary
from sudachipy import SplitMode, Morpheme


COMMON_PREFIX = "OT_Brassica"
SUDACHI_POS_EN_JA = {"n": "名詞", "v": "動詞", "adj": "形容詞", "kj": "形状詞"}


def split_conference(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    摂食カンファレンスに関する記載とそうじゃないのに分ける
    """
    seq_ids_from_conference: pl.Series = df.filter(
        [pl.col("記事").str.contains("摂食カンファ")]
    )["シーケンスID"].unique()
    condition = pl.col("シーケンスID").is_in(seq_ids_from_conference)
    return (df.filter(condition), df.filter(condition.not_()))


def preprocess_raw_cartes(
    fs: list[Path],
    to_subfolder: Optional[str],
    stop_words: set[str],
    to_line_csvs: bool = True,
):
    """
    いろんな事前処理をしてファイルを作っておく
    """
    f = fs[0]
    time_stamp: str = to_subfolder or timestamp()
    name_base = f"{COMMON_PREFIX}_{f.stem}"
    base_path = f.parent / time_stamp / name_base
    if not base_path.parent.exists():
        base_path.parent.mkdir()
    df = load_cartes(fs)
    # 初手で normalize しておく
    df = df.with_columns(
        [
            pl.col("記事").map_elements(
                lambda text: unicodedata.normalize("NFKC", text), return_dtype=pl.String
            )
        ]
    )
    (conf_record, daily_record) = split_conference(df)
    # カンファレンス記録だけ
    conf_record.write_excel(base_path.with_suffix(".conf_record.xlsx"))
    # それ以外の記録だけ
    daily_record.write_excel(base_path.with_suffix(".daily_record.xlsx"))
    # SOAPF で分ける
    daily_soapf = {
        kind: daily_record.filter(pl.col("記事データ種別").eq(kind)) for kind in "SOAPF"
    }
    # 分かち書きの準備
    mode: SplitMode = sudachipy.tokenizer.Tokenizer.SplitMode.C
    suddic = sudachipy.dictionary.Dictionary()  # pos_matcher を作ろうとした跡
    tokenizer = suddic.create(mode=mode)
    # この stop_words を使いました
    with base_path.with_suffix(".stop_words.txt").open("w") as f:
        f.writelines((word + "\n" for word in stop_words))
    if to_line_csvs:
        for kind, df in daily_soapf.items():
            # SOAPF ごとの一般カルテ記録
            df.write_excel(base_path.with_suffix(f".daily_record_{kind}.xlsx"))
            # 記載されたテキスト全文
            # FIXME
            whole_text: pl.Series = df.with_columns(
                [pl.col("記事").str.strip_chars()]
            ).filter(pl.col("記事").ne(""))["記事"]
            with base_path.with_suffix(f".whole_text_{kind}.txt").open("w") as f:
                f.writelines((line + "\n" for line in whole_text.to_list()))
            df = do_tokenize(df, tokenizer)
            # 数字だけ除いて，えり好みせず全単語載せておく
            # 品詞は下記の通り選んでそれぞれ別にファイルを作ってみる
            for poss in [
                ["n"],
                ["v"],
                ["adj", "kj"],
                ["n", "v"],
                ["n", "v", "adj", "kj"],
            ]:
                save_tokenized(
                    df,
                    base_path,
                    kind=kind,
                    parts_of_speech=poss,
                    stop_words=stop_words,
                )
    else:
        tokenize_and_save_excel(
            daily_record,
            base_path,
            kind="SOAPF",
            these_parts_of_speech=[
                ["n"],
                ["v"],
                ["adj", "kj"],
                ["n", "v"],
                ["n", "v", "adj", "kj"],
            ],
            stop_words=stop_words,
        )


STOP_WORDS: set[str] = {
    "こと",
    "事",
    "言う",
    "いう",
    "もの",
    "物",
    "ところ",
    "所",
    "時",
    "とき",
    "よう",
    "為",
    "ため",
    "思う",
    "そう",
    "こう",
    "行う",
    "kg",
    "cm",
    "OT",
    "otr",
    "T",
    "I",
    "BMI",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path, help="Use these csvs", nargs="+")
    parser.add_argument(
        "--out-base",
        type=str,
        help="Use this as the name of the subdirectory to store data",
    )
    args = parser.parse_args()
    preprocess_raw_cartes(args.sources, args.out_base, STOP_WORDS)


if __name__ == "__main__":
    main()
