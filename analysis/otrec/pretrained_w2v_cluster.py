"""
既存のモデルから単語をクラスタ化してみて，記載のグループを作ってみる
"""

from pathlib import Path
from typing import Optional
from collections import defaultdict


import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.preprocessing as sklp
from gensim.models import KeyedVectors, keyedvectors
import matplotlib.pyplot as plt

from dwht.carte.load import load_cartes


def cluster_words_in(
    wv: KeyedVectors, words: list[str], n_clusters: int = 12
) -> tuple[KMeans, dict[str, int]]:
    """
    学習済みの KeyedVectors を使って， list[str] をクラスタに分ける．
    多分順番が固定されてるのが大事なので list[str] で
    """
    km = KMeans(n_clusters=n_clusters, max_iter=1000)
    word_vectors = sklp.normalize([wv[w] for w in words if w in wv])
    km.fit(word_vectors)
    # 単語→クラスタの dict
    clusters: dict[str, int] = {}
    assert km.labels_ is not None
    for word, cluster_id in zip(words, km.labels_):
        clusters[word] = cluster_id
    return (km, clusters)


def pca_and_cluster_words_in(
    wv: KeyedVectors, words: list[str], n_clusters: int = 12, pca_dimensions: int = 20
) -> tuple[PCA, KMeans, dict[str, int]]:
    """
    学習済みの KeyedVectors を使って， list[str] をクラスタに分ける．
    """
    km = KMeans(n_clusters=n_clusters, max_iter=1000)
    pca = PCA(n_components=pca_dimensions)
    word_coords = pca.fit_transform(sklp.normalize([wv[w] for w in words if w in wv]))
    km.fit(word_coords)
    # 単語→クラスタの dict
    clusters: dict[str, int] = {}
    assert km.labels_ is not None
    for word, cluster_id in zip(words, km.labels_):
        clusters[word] = cluster_id
    return (pca, km, clusters)


def add_cluster_to_morphed_excel(
    df: pl.DataFrame,
    pca_result: tuple[PCA, KMeans, dict[str, int]],
    out_file: Path,
):
    pass


def report_cluster(
    wv: KeyedVectors,
    cluster_result: tuple[PCA, KMeans, dict[str, int]],
    report_similarities: bool = False,
):
    (pca, km, clusters) = cluster_result
    centres = km.cluster_centers_
    assert km.labels_ is not None
    # 単語の vector とクラスタの中心のとおさで
    # TODO: 一旦 inverse_transform してるが，PCAした後のやつで見るほうが自然かも
    for i in range(len(set(km.labels_))):
        the_cluster = [
            (w, cosine_similarity([centres[i]], pca.transform([wv[w]]))[0][0])
            for w in clusters
            if clusters[w] == i
        ]
        # sort by cosine similartities
        the_cluster.sort(key=lambda d: d[1], reverse=True)
        print(f"cluster {i}:")
        for w, simil in the_cluster:
            print(f"{w}({simil:.4f})", end="  ")

        print("")
        if report_similarities:
            print(
                wv.cosine_similarities(
                    centres[i], [wv[w] for w in the_cluster if w in wv]
                )
            )
            print("vs")
            print(
                wv.cosine_similarities(
                    centres[i - 1], [wv[w] for w in the_cluster if w in wv]
                )
            )


def load_corpus(f: Path, min_occ: int = 0, keep_duplicate: bool = True) -> list[str]:
    corpus: list[str] = []
    word_counts: defaultdict[str, int] = defaultdict(int)
    with f.open("r") as cpf:
        for line in cpf.readlines():
            for w in line.strip().split(" "):
                corpus.append(w)
                word_counts[w] += 1
    if keep_duplicate:
        corpus_n_occurences = [w for w in corpus if word_counts[w] >= min_occ]
    else:
        corpus_n_occurences = [w for (w, c) in word_counts.items() if c >= min_occ]
        # Dictionaries preserve insertion order.
    return corpus_n_occurences


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="load this model")
    parser.add_argument("source_corpus", type=Path, help="corpus used")
    # parser.add_argument("cartes", type=Path, help="The original data", nargs="+")
    parser.add_argument("--out-dir", type=Path, help="Dir to put the plots on")
    parser.add_argument(
        "--n-clusters", "-n", type=int, help="How many clusters to use?", default=8
    )
    parser.add_argument(
        "--pca-dimensions", "-p", type=int, help="PCA Dimensions", default=20
    )
    parser.add_argument(
        "--minimum-occurences",
        "-m",
        type=int,
        help="Minimum occurences for a word",
        default=4,
    )
    args = parser.parse_args()
    model = KeyedVectors.load(str(args.model.resolve()))
    # cartes = load_cartes(args.cartes)
    out_base_path = args.out_dir or args.source_corpus.parent
    corpus = load_corpus(args.source_corpus, min_occ=args.minimum_occurences)
    (pca, km, cl) = pca_and_cluster_words_in(
        wv=model,
        words=corpus,
        n_clusters=args.n_clusters,
        pca_dimensions=args.pca_dimensions,
    )
    report_cluster(model, (pca, km, cl))

    # clustered = clustered_records(cartes, cl)
    # clustered.write_excel(out_base_path / f"clustered_records_n{len(cl):02d}.xlsx")


if __name__ == "__main__":
    main()
