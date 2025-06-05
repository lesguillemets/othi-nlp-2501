"""
basal functions that'd be useful everywhere
"""

from pathlib import Path
import argparse
from argparse import ArgumentParser
from datetime import datetime, timedelta

from typing import TypeVar

DEFAULT_CSV_PATH = Path("./data/sample.csv")


def basic_parser_for_file() -> ArgumentParser:
    """
    accepts:
    - file path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Path to the file",
    )
    return parser


def timestamp() -> str:
    now_jst = datetime.utcnow() + timedelta(hours=9)
    return now_jst.strftime("%Y%m%d_%H%M%S")


T = TypeVar("T")


def flatten_list(ls: list[list[T]]) -> list[T]:
    """
    リストを一段階 flatten
    """
    result = []
    for item in ls:
        result.append(item)
    return result
