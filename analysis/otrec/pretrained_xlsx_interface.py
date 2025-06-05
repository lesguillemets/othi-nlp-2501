"""
クラスタわけの結果
"""

import re
from re import Pattern

from typing import Optional

import polars as pl
from pathlib import Path

from dwht.carte.load import load_carte

cluster_name: Pattern[str] = re.compile(r"^cluster\s+\d+:")
word_score_pattern: Pattern[str] = re.compile(r"(\S+)\(([-+]?\d+\.\d{4})\)")


def parse_clusters_from_txt(fp: Path) -> list[dict[str, float]]:
    clusters: list[dict[str, float]] = []
    words: dict[str, float] = {}
    """
    以下のフォーマットで記載されたテキストをパースして，
    クラスタごとに {単語: 値} の dict として返す
    cluster 5:
    太陽(0.2332) みかん(0.5829)
    """
    cluster_number = 0  # zero start
    with fp.open("r") as f:
        for line in f:
            line = line.strip()
            if cluster_name.match(line):
                cluster_number += 1
                clusters.append(words)
                words = {}
            for word, score in word_score_pattern.findall(line):
                words[word] = float(score)
    clusters.append(words)
    assert len(clusters) == cluster_number + 1
    return clusters[1:]


def count_cluster_occurences(
    clusters: list[dict[str, float]],
    sentence: list[str],
    similarity_threshold: float = 0.5,
    cluster_groups: Optional[list[list[int]]] = None,
) -> list[int]:
    """
    文章（正規化されて単語のリスト化された）に，cluster に含まれる単語が何回出てくるか
    clusters は上の parse_clusters_from_txt の結果を想定，i 番目のクラスタを含んだリストで，
    各クラスタは単語→値（中心からの similarity）の dict.

    クラスタ内の単語のうち，similarity_thresholdより高いもののみを採用する．
    cluster_groups が与えられるときは，そのリスト内の各リストに含まれるクラスタを同一視する
    ( [ [2,3], [0,4], [1] ] では，クラスタ2と3，クラスタ0と4 は同一視して，それぞれクラスタ0, 1 と読み替える)
    """
    if cluster_groups is None:
        # 本人だけと同一視
        cluster_groups = [[i] for i in range(len(clusters))]
    occurences: list[int] = [0 for _ in cluster_groups]
    for word in sentence:
        for i, cluster_indices in enumerate(cluster_groups):
            # この cluster_group に含まれるクラスタ
            clusters_in_group = [
                clusters[cluster_index] for cluster_index in cluster_indices
            ]
            # それぞれのクラスタにおける（中心との） similarity
            the_similarities: list[float] = [
                c[word] for c in clusters_in_group if word in c
            ]
            # 一番近いクラスタで閾値を超えてたらカウントする．
            # …のは，なさそうだけど，ダブルカウントを防ぐため．
            if len(the_similarities) > 0 and (
                max(the_similarities) > similarity_threshold
            ):
                occurences[i] += 1
    return occurences


def add_cluster_occurences(
    df: pl.DataFrame,
    clusters: list[dict[str, float]],
    similarity_threshold: float = 0.5,
    cluster_groups: Optional[list[list[int]]] = None,
    col_name: str = "norm_forms_n",
    result_col_name_prefix: str = "",
) -> pl.DataFrame:
    """
    カルテ記載を読み込んだ df から，
    上でパーズするような形式のクラスタの出現回数をカウントして関連情報を追記する
    複数列扱う想定はなく，col_name について処理する．
    クラスタの中で similarity_threshold 以上ののみを選択する．
    - f"clr{i:02}" : 割合 (cluster/ratio)
    - f"cl{i:02}" : 回数 (cluster)
    - f"最多クラスタ": 最も回数の多いクラスタ
    それぞれ result_col_name_prefix を列名の prefix として付ける
    """
    # norm_forms_{'_'.join(parts_of_speech)}
    # similarity_threshold を超えるものを使って，そのクラスタに属する単語の
    # 出現回数をカウントしてリストを入れとく
    df = df.with_columns(
        [
            pl.col(col_name)
            .map_elements(
                lambda row: count_cluster_occurences(
                    clusters,
                    row.strip().split(),
                    similarity_threshold=similarity_threshold,
                    cluster_groups=cluster_groups,
                ),
                return_dtype=pl.List(int),
            )
            .alias(f"clcounts_{col_name}")
        ]
    )
    if cluster_groups is None:
        n_clusters = len(clusters)
    else:
        n_clusters = len(cluster_groups)

    def calc_occ_ratio(counts: list[int], i: int) -> float:
        """
        クラスタごとのカウントのリストから，それぞれのクラスタの出現の割合
        """
        s = sum(counts)
        if s == 0:
            return 0
        else:
            return counts[i] / s

    for i in range(n_clusters):
        df = df.with_columns(
            [
                pl.col(f"clcounts_{col_name}")
                .map_elements(lambda d: calc_occ_ratio(d, i), return_dtype=pl.Float32)
                .alias(f"{result_col_name_prefix}clr{i:02}")
            ]
        )
    for i in range(n_clusters):
        df = df.with_columns(
            [
                pl.col(f"clcounts_{col_name}")
                .map_elements(lambda d: d[i], return_dtype=pl.Int32)
                .alias(f"{result_col_name_prefix}cl{i:02}")
            ]
        )

    def get_max_cluster(counts: list[int]) -> str:
        """
        クラスタごとのカウントのリストから，どのクラスタに属してるかの文字列を返す
        """
        max_count = max(counts)
        if max_count == 0:
            # どのクラスタも出現してない
            return ""
        else:
            return ",".join((f"{i}" for (i, c) in enumerate(counts) if c == max_count))

    df = df.with_columns(
        [
            pl.col(f"clcounts_{col_name}")
            .map_elements(get_max_cluster, return_dtype=pl.String)
            .alias(f"{result_col_name_prefix}最多クラスタ")
        ]
    )
    df.drop_in_place(f"clcounts_{col_name}")

    return df


def load_cluster_groups(p: Path) -> list[list[int]]:
    with p.open("r") as f:
        return [[int(d) for d in line.strip().split()] for line in f.readlines()]


def run(
    carte_file: Path,
    cluster_file: Path,
    cluster_groups_file: Optional[Path] = None,
    similarity_threshold: float = 0.5,
    col_name: str = "norm_forms_n",
    colname_prefix: str = "",
    outfile: Optional[Path] = None,
):
    df = load_carte(carte_file)
    cl = parse_clusters_from_txt(cluster_file)
    if cluster_groups_file is not None:
        cg = load_cluster_groups(cluster_groups_file)
    else:
        cg = None

    occ = add_cluster_occurences(
        df,
        cl,
        cluster_groups=cg,
        similarity_threshold=similarity_threshold,
        col_name=col_name,
        result_col_name_prefix=colname_prefix,
    )
    outfile = outfile or carte_file.with_stem(
        f"{carte_file.stem}_{cluster_file.stem}"
    ).with_suffix(".countadded.xlsx")
    occ.write_excel(outfile)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--carte", type=Path, help="carte data with morphemes listed", required=True
    )
    parser.add_argument(
        "--cluster", type=Path, help="Clustering Result to use", required=True
    )
    parser.add_argument(
        "--cluster-group",
        type=Path,
        help="newline-separated list of space-separated list of zero-based number. These clusters are treated as a same",
    )
    parser.add_argument(
        "--colname-prefix",
        type=str,
        default="",
        help="Use this string as the prefix for added columns",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(
        args.carte,
        args.cluster,
        cluster_groups_file=args.cluster_group,
        similarity_threshold=args.similarity_threshold,
        colname_prefix=args.colname_prefix,
    )


if __name__ == "__main__":
    main()
