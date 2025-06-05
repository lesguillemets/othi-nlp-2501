#!/usr/bin/env python3

from pathlib import Path

import polars as pl


def load_carte(f: Path) -> pl.DataFrame:
    if f.suffix == (".csv"):
        df = pl.read_csv(
            f,
            has_header=True,
            schema_overrides={
                "患者ID": pl.String,
                "記載日": pl.String,
                "記載時刻": pl.Time,
                "シーケンスID": pl.String,  # 元々 varchar なので
            },
        )
        df = df.with_columns([pl.col("記載日").str.to_date("%Y-%m-%d")])
    elif f.suffix == (".xlsx"):
        df = pl.read_excel(
            f,
            has_header=True,
            schema_overrides={
                "患者ID": pl.String,
                "記載日": pl.String,
                "記載時刻": pl.Time,
                "シーケンスID": pl.String,  # 元々 varchar なので
            },
            engine="openpyxl",
        )
        # excel には日付だけの欄が発生しないので多分こうする必要がある
        df = df.with_columns(
            [pl.col("記載日").str.slice(0, 10).str.to_date("%Y-%m-%d")]
        )
    else:
        raise ValueError(f"unsupported prefix: {f}")
    df = df.with_columns([pl.col("記事").fill_null("")])
    return df


def load_cartes(fs: list[Path]) -> pl.DataFrame:
    loaded = pl.concat([load_carte(f) for f in fs], how="diagonal", rechunk=False)
    loaded.rechunk()
    return loaded


def main() -> pl.DataFrame:
    from dwht.base import basic_parser_for_file, DEFAULT_CSV_PATH

    args = basic_parser_for_file().parse_args()
    p = args.file or DEFAULT_CSV_PATH
    loaded = load_carte(p)
    with pl.Config(tbl_cols=-1):
        print(loaded)
    return loaded


if __name__ == "__main__":
    main()
