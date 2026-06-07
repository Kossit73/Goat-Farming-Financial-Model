from __future__ import annotations

import pandas as pd
import streamlit as st


def render_named_table(title: str, table: pd.DataFrame) -> None:
    if isinstance(table, pd.DataFrame) and not table.empty:
        st.markdown(f"##### {title}")
        st.dataframe(table, use_container_width=True)
