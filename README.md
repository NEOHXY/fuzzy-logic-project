# Fuzzy Logic Project

This repository contains a fuzzy-logic loan approval dashboard project. The app combines a fuzzy risk engine, a Flask web interface, optional AI-assisted reasoning providers, retraining support, and an admin dashboard.

## Features

- Loan approval risk assessment using a fuzzy inference engine.
- Flask web app with applicant-facing and admin-facing pages.
- Static frontend assets for dashboard interactions and styling.
- Optional AI reasoning providers configured through environment variables.
- Retraining workflow support for model and data updates.
- Dockerfile and requirements file for deployment-oriented setup.

## Tech Stack

- Python
- Flask
- HTML, CSS, and JavaScript
- Fuzzy logic rule engine
- Google Cloud / Cloud Run deployment concepts
- Optional AI provider environment variables

## Installation

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r malaysian_loan_approval_v8_6\requirements.txt
```

Configure any required environment variables locally. Do not commit real API keys, admin tokens, webhook secrets, datasets, or production configuration files.

## How To Run

Run the Flask application from the extracted project folder:

```powershell
cd malaysian_loan_approval_v8_6
python app.py
```

Then open the local URL printed by the application.

## Folder Structure

```text
.
├── malaysian_loan_approval_v8_6/
│   ├── app.py
│   ├── fuzzy_engine.py
│   ├── retraining.py
│   ├── storage.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── static/
│   └── templates/
├── .gitignore
└── README.md
```

## Notes And Limitations

- The original zip archive and dashboard link file are kept locally but are intentionally ignored.
- The dashboard link file contains an admin-token URL and must not be committed.
- CSV datasets, generated JSON data, Python caches, archives, and credential files are excluded from Git.
- Rotate the existing admin token before making this repository public.
