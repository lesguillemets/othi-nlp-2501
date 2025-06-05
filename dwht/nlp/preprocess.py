"""
自然言語処理関連の前処理全般にまつわるヘルパ関数たち

- do_tokenize が，形態素解析を済ませて新たに column を生やす
…ので，その結果を使い回すのが適切なつくり
"""

from pathlib import Path
from typing import Generator, Callable, Optional, Any

import polars as pl

from sudachipy import Tokenizer, Morpheme, MorphemeList
import sudachipy.tokenizer
import sudachipy.dictionary
from sudachipy import SplitMode

from dwht.morph.sudachi_interface import SUDACHI_POS_EN_JA as POS_EN_JA


def do_tokenize(
    df: pl.DataFrame,
    tokenizer: Tokenizer,
    morph_filter: Optional[Callable[[Morpheme], bool]] = None,
):
    """
    tokenize して morphemes という列を作る
    morpheme は list[MorphemeList]... が多分サポートされてないので，
    , normalised_form, dictionary_form, part_of_speech を含む
    struct にする．行ごとに分けてるからそれのリストになる
    """

    def tokenize_and_to_struct(article: str):
        tokenized_lines: Generator[MorphemeList] = (
            tokenizer.tokenize(line) for line in article.split("\n")
        )
        return [
            [
                {
                    "dictionary_form": morph.dictionary_form(),
                    "normalized_form": morph.normalized_form(),
                    "part_of_speech": morph.part_of_speech(),
                }
                for morph in line
                if morph_filter is None or morph_filter(morph)
            ]
            for line in tokenized_lines
        ]

    df = df.with_columns(
        [
            pl.col("記事")
            .map_elements(
                tokenize_and_to_struct,
                return_dtype=pl.List(pl.List(pl.Struct)),
            )
            .alias("morphemes")
        ]
    )
    return df


def default_filter_m(m: dict[str, Any]) -> bool:
    """
    morpheme のうち，上記で dictionary_form と normalized_form,
    part_of_speech を抜き出したデータに対して，
    それを token に含めるかどうかを判別するデフォルトの函数
    """
    mpos = m["part_of_speech"]
    return mpos[1] not in ["数詞", "非自立可能", "固有名詞"]


def add_tokenized_column(
    df: pl.DataFrame,
    kind: str,
    parts_of_speech: list[str],
    stop_words: set[str] = set(),
    filter_m: Callable[[dict[str, Any]], bool] = default_filter_m,
) -> pl.DataFrame:
    """
    input: df: do_tokenize を通って，'morphemes' にその結果が入ったdf
    所定の条件を満たす morpheme を並べた列を作る．
    列の名前はnorm_forms_{parts_of_speech}
    kind: soapf のどれかを表す文字列（ここでも一応フィルタしてる）
    parts_of_speech: SUDACHI_POS_EN_JA を参考に，含める品詞
    filter_m: morpheme に対して掛けられるフィルタ．上の default_filter_m を参照
    """
    pos_ja = [POS_EN_JA[pos] for pos in parts_of_speech]

    def should_accept(m) -> bool:
        """
        m は Morpheme のつもりだが，pl.Struct だ
        accept: 与えられた品詞のもの
        refuse: 数詞か非自立可能なもの，固有名詞か， stop_words に含まれるもの
        """
        if m["dictionary_form"] in stop_words or m["normalized_form"] in stop_words:
            return False
        return filter_m(m) and m["part_of_speech"][0] in pos_ja

    # TODO: ひょっとしたら with_columns で行けるかも
    df = df.filter(pl.col("記事データ種別").is_in(list(kind.upper()))).with_columns(
        [
            pl.col("morphemes")
            .map_elements(
                lambda lines: [
                    " ".join((m["normalized_form"] for m in line if should_accept(m)))
                    + "\n"
                    for line in lines
                ],
                return_dtype=pl.List(pl.String),
            )
            .alias(
                f"norm_forms_{'_'.join(parts_of_speech)}"
            )  # list[str] : 元の一行が要素1つになってる
        ]
    )
    return df


def tokenize_and_save_excel(
    df: pl.DataFrame,
    base_path: Path,
    kind: str,
    these_parts_of_speech: list[list[str]],
    stop_words: set[str] = set(),
    filter_m: Callable[[dict[str, Any]], bool] = default_filter_m,
) -> pl.DataFrame:
    """
    input: 素の（カルテ読み込んだ）DataFrame
    these_parts_of_speech: [ ['n','v'], ['n','v','adj'] ] みたいな
    それぞれ morpheme に分けて適宜フィルタして列として保存する
    """
    mode: SplitMode = sudachipy.tokenizer.Tokenizer.SplitMode.C
    suddic = sudachipy.dictionary.Dictionary()  # pos_matcher を作ろうとした跡
    tokenizer = suddic.create(mode=mode)
    df = do_tokenize(df, tokenizer)
    for parts_of_speech in these_parts_of_speech:
        df = add_tokenized_column(
            df,
            kind=kind,
            parts_of_speech=parts_of_speech,
            stop_words=stop_words,
            filter_m=filter_m,
        ).with_columns(
            pl.col(f"norm_forms_{'_'.join(parts_of_speech)}").list.join("\n")
        )
    df.write_excel(base_path.with_suffix(".tokenized.xlsx"))
    return df


def save_tokenized(
    df: pl.DataFrame,
    base_path: Path,
    kind: str,
    parts_of_speech: list[str],
    stop_words: set[str] = set(),
    filter_m: Callable[[dict[str, Any]], bool] = default_filter_m,
):
    """
    input: df: do_tokenize を通って，'morphemes' にその結果が入ったdf
    行ごとに空白で分かち書きした単語からなるテキストと，
    それぞれの行のシーケンスIDを書いたファイルを生成する
    kind: soapf のどれかを表す文字列（ここでも一応フィルタしてる）
    parts_of_speech: SUDACHI_POS_EN_JA を参考に，含める品詞
    filter_m: morpheme に対して掛けられるフィルタ．上の default_filter_m を参照

    かなり，nlp/gen_txt_for_doc2vec と被ってはいる
    """

    df_ = add_tokenized_column(df, kind, parts_of_speech, stop_words, filter_m)
    the_path = base_path.with_suffix(f".words_{kind}_{'_'.join(parts_of_speech)}.txt")
    with (
        the_path.open("a") as f_morphs,
        the_path.with_suffix(".sequence_ids.txt").open("a") as f_seq,
    ):
        for seq_id, words_lines in df_[
            "シーケンスID", f"norm_forms_{'_'.join(parts_of_speech)}"
        ].rows():
            # 空行はたんに除く
            wl = list(filter(lambda n: n.strip() != "", words_lines))
            f_morphs.writelines(wl)
            f_seq.writelines([f"{seq_id}\n" for _ in wl])
    return df_
