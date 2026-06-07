from __future__ import annotations

import pandas as pd
import streamlit as st


def render_schedule_summary(title: str, table: pd.DataFrame) -> None:
    st.markdown(f"##### {title}")
    st.dataframe(table)
