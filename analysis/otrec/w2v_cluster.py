"""
w2v_learn でできた w2v モデルに基づいて単語をクラスタ化する．
各カルテ記載について，そのクラスタに属する単語が何個出現してるかを併記するようにする．
"""

from pathlib import Path
from typing import Optional
from collections import defaultdict


import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.preprocessing as sklp
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

from wordcloud import WordCloud

from dwht.carte.load import load_cartes


def do_cluster(model: Word2Vec, n_clusters: int = 8) -> tuple[KMeans, dict[str, int]]:
    do_elbow(model)
    words: list[str] = model.wv.index_to_key
    vectors = model.wv.get_normed_vectors()
    km = KMeans(
        n_clusters=n_clusters,
    )
    km.fit(vectors)
    centres = km.cluster_centers_
    # 単語→クラスタの dict
    clusters: dict[str, int] = {}
    assert km.labels_ is not None
    for word, cluster_id in zip(words, km.labels_):
        clusters[word] = cluster_id
    return (km, clusters)


def report_cluster(model: Word2Vec, cluster_result: tuple[KMeans, dict[str, int]]):
    (km, clusters) = cluster_result
    normalized_centers = sklp.normalize(km.cluster_centers_)
    for center in normalized_centers:
        print(model.wv.similar_by_vector(center, topn=20))


def do_elbow(model, nmax=20, save_path: Path = Path("./data/elbow_w2v.png")):
    vectors = model.wv.vectors
    inertiae = []
    for i in range(1, nmax + 1):
        km = KMeans(
            n_clusters=i,
            max_iter=300,
        )
        km.fit(vectors)
        inertiae.append(km.inertia_)

    plt.plot(range(1, nmax + 1), inertiae, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("inertiae")
    plt.xticks(range(1, nmax + 1))
    plt.savefig(save_path)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="load this model")
    parser.add_argument("source_corpus", type=Path, help="corpus used")
    parser.add_argument("cartes", type=Path, help="The original data", nargs="+")
    parser.add_argument("--out-dir", type=Path, help="Dir to put the plots on")
    parser.add_argument(
        "--n-clusters", "-n", type=int, help="How many clusters to use?", default=8
    )
    args = parser.parse_args()
    model = Word2Vec.load(str(args.model.resolve()))
    cartes = load_cartes(args.cartes)
    out_base_path = args.out_dir or args.source_corpus.parent
    (km, cl) = do_cluster(model, n_clusters=args.n_clusters)
    report_cluster(model, (km, cl))

    # clustered = clustered_records(cartes, cl)
    # clustered.write_excel(out_base_path / f"clustered_records_n{len(cl):02d}.xlsx")


if __name__ == "__main__":
    main()
