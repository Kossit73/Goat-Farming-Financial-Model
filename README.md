# Goat Farming Financial Model

This repository provides a small helper around the goat farming Excel template
so that you can analyse the income statement, cash flow, and valuation data in
Python.

## Installation

Create a virtual environment and install the local package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Run the CLI, pointing it at the Excel workbook that follows the template
structure (``IS``, ``CF``, ``Valuation`` sheets). The script prints a short
summary and can optionally write the tidy data to disk.

```bash
python -m goat_financial_model.cli path/to/model.xlsx --output tidy.csv
```

You can also use the :class:`goat_financial_model.GoatModel` directly in Python
for custom analysis.

## Interactive dashboard

Install the optional dependencies and launch the Streamlit dashboard to explore
scenarios interactively:

```bash
pip install -e .[app]
streamlit run streamlit_app.py
```

The dashboard lets you flex the milk price and feed cost assumptions, monitor
how the adjusted scenario impacts profitability, and export the resulting time
series for further analysis.
