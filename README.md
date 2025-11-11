# Goat Farming Financial Model

This repository provides a lightweight set of helpers for assembling a goat
farming financial model from manually entered schedules. Instead of relying on
an Excel workbook, you can capture the timeline directly in Python, via CSV, or
through the included Streamlit dashboard.

## Installation

Create a virtual environment and install the local package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Prepare a CSV file that contains a ``Period`` column (monthly dates) together
with financial metrics such as ``Revenue``, ``COGS``, ``EBITDA``, ``NPAT`` and
any other series you plan to analyse. Run the CLI to generate a tidy dataframe
and optional summary:

```bash
python -m goat_financial_model.cli path/to/schedule.csv --output tidy.csv
```

The command accepts optional valuation inputs, for example
``--wacc 12 --npv 750000``. You can also construct an
:class:`goat_financial_model.GoatModel` (or the accompanying
:class:`goat_financial_model.InputSchedule`) directly in Python for custom
analysis.

## Interactive dashboard

Install the optional dependencies and launch the Streamlit dashboard to explore
scenarios interactively:

```bash
pip install -e .[app]
streamlit run streamlit_app.py
```

The dashboard exposes editable schedules so you can enter revenue, cost and
expense assumptions, adjust milk price and feed cost shocks, view KPI trends,
and export the resulting time series for further analysis.
