"""CSV ingestion shared by API, training, evaluation and submission tools."""
from __future__ import annotations

import csv
import codecs
import itertools
from pathlib import Path

import pandas as pd


class CsvHeaderError(ValueError):
    """The original CSV header is ambiguous before pandas renames duplicates."""


def _check_header(stream, encoding, options):
    position = stream.tell()
    try:
        if isinstance(stream.read(0), bytes):
            lines = codecs.iterdecode(iter(stream.readline, b""), encoding)
        else:
            lines = iter(stream)
        first_line = next(lines, "")
        lines = itertools.chain([first_line.lstrip("\ufeff")], lines)
        if options.get("skip_blank_lines", True):
            # Match pandas' default without skipping quoted empty/space headers.
            lines = itertools.dropwhile(lambda line: not line.strip(), lines)
        delimiter = options.get("delimiter") or options.get("sep", ",")
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            raise CsvHeaderError("CSV header validation requires a one-character delimiter")
        reader = csv.reader(
            lines,
            delimiter=delimiter,
            quotechar=options.get("quotechar", '"'),
            doublequote=options.get("doublequote", True),
            escapechar=options.get("escapechar"),
            skipinitialspace=options.get("skipinitialspace", False),
            quoting=options.get("quoting", csv.QUOTE_MINIMAL),
        )
        header = next(reader, [])
        if header:
            header[0] = header[0].lstrip("\ufeff")
        if len(header) != len(set(header)):
            raise CsvHeaderError("Повторяющиеся колонки CSV")
    finally:
        stream.seek(position)


def read_csv_checked(source, **kwargs):
    """Check raw quoted headers and preserve identifiers such as ``001``."""
    encoding = kwargs.get("encoding") or "utf-8-sig"
    if isinstance(source, (str, Path)):
        with open(source, "rb") as stream:
            _check_header(stream, encoding, kwargs)
    else:
        _check_header(source, encoding, kwargs)
    dtype = kwargs.pop("dtype", None)
    if dtype is None or isinstance(dtype, dict):
        dtype = {"warehouse_name": "string", "article": "string", "char_article": "string", **(dtype or {})}
    return pd.read_csv(source, dtype=dtype, **kwargs)
