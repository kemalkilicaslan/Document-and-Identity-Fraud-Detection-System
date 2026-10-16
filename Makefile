.PHONY: install test lint reproduce reports dashboard api

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

reproduce:
	./run_all.sh

reports:
	open reports/management_report.html reports/case_study.html

dashboard:
	streamlit run src/docfraud/dashboard.py

api:
	uvicorn docfraud.api:app --reload
