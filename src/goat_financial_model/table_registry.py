from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd


Normalizer = Callable[[pd.DataFrame], pd.DataFrame]
RowFactory = Callable[[pd.DataFrame], dict[str, Any]]


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    default: Any = None


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[ColumnSchema, ...]
    default_rows: tuple[dict[str, Any], ...] = ()
    normalizer: Optional[Normalizer] = None
    add_row_factory: Optional[RowFactory] = None
    required_order: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ordered_columns(self) -> list[str]:
        if self.required_order:
            return list(self.required_order)
        return [column.name for column in self.columns]

    @property
    def defaults(self) -> dict[str, Any]:
        return {column.name: column.default for column in self.columns}


def build_default_table(schema: TableSchema) -> pd.DataFrame:
    rows = [dict(row) for row in schema.default_rows] or [dict(schema.defaults)]
    table = pd.DataFrame(rows)
    for column in schema.columns:
        if column.name not in table.columns:
            table[column.name] = column.default
    return _finalize_table(schema, table)


def ensure_table(schema: TableSchema, table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if table is None or table.empty:
        work = build_default_table(schema)
    else:
        work = table.copy()
        for column in schema.columns:
            if column.name not in work.columns:
                work[column.name] = column.default
    return _finalize_table(schema, work)


def add_table_row(schema: TableSchema, table: pd.DataFrame) -> pd.DataFrame:
    work = ensure_table(schema, table)
    if schema.add_row_factory is not None:
        new_row = schema.add_row_factory(work)
    else:
        new_row = dict(schema.defaults)
    return _finalize_table(schema, pd.concat([work, pd.DataFrame([new_row])], ignore_index=True))


def remove_table_row(schema: TableSchema, table: pd.DataFrame, index: int) -> pd.DataFrame:
    work = ensure_table(schema, table)
    if 0 <= index < len(work):
        work = work.drop(index=index).reset_index(drop=True)
    return _finalize_table(schema, work)


def _finalize_table(schema: TableSchema, table: pd.DataFrame) -> pd.DataFrame:
    work = table.copy()
    if schema.normalizer is not None:
        work = schema.normalizer(work)
    ordered = schema.ordered_columns
    remainder = [column for column in work.columns if column not in ordered]
    return work[ordered + remainder].reset_index(drop=True)
