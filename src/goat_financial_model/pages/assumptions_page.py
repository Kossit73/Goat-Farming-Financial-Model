from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
import streamlit as st


EnsureFn = Callable[[Optional[pd.DataFrame]], pd.DataFrame]
RenderEditor = Callable[[str, pd.DataFrame, Callable[[pd.DataFrame], None]], None]


@dataclass(frozen=True)
class BiologicalEditorDefinition:
    name: str
    caption: str
    ensure_fn: EnsureFn
    editor_key: str


def render_biological_assumption_editor(
    *,
    definitions: list[BiologicalEditorDefinition],
    assumptions: dict[str, pd.DataFrame],
    render_row_editor: RenderEditor,
) -> dict[str, pd.DataFrame]:
    for definition in definitions:
        assumptions[definition.name] = definition.ensure_fn(assumptions.get(definition.name))

    selected_name = st.selectbox(
        "Biological assumption table",
        options=[definition.name for definition in definitions],
        key="biological_editor_name",
    )
    selected = next(definition for definition in definitions if definition.name == selected_name)
    st.markdown(f"#### {selected.name}")
    st.caption(selected.caption)
    render_row_editor(
        selected.editor_key,
        assumptions[selected.name],
        lambda updated, assumption_name=selected.name, ensure_fn=selected.ensure_fn: assumptions.__setitem__(
            assumption_name,
            ensure_fn(updated),
        ),
    )
    return assumptions
