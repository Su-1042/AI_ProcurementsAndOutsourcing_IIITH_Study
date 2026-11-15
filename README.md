# Research Document Word-Cloud Pipeline

This script builds unigram and bigram frequency tables and word-cloud images from PDF and HTML documents in the notice and specification folders. Text is cleaned and lemmatized with spaCy, filtered by document frequency, and cached to speed up later runs.

## Requirements

- Python 3.11 or newer
- Python packages listed in `requirements.txt`
- The spaCy English model `en_core_web_sm` (the script downloads it automatically if it is missing)
- Tesseract OCR installed on the system only if scanned PDFs need OCR

## Setup

Run these commands from this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first run also downloads the NLTK English stopwords and, if needed, the spaCy English model. To install the model explicitly instead, run:

```bash
python -m spacy download en_core_web_sm
```

## Input Folders

By default, the script searches for folders whose names contain `notice` and `spec` under the current folder or its `dataset/` subfolder. In this repository, the expected folders are `dataset/notice_docs/` and `dataset/specification_docs/`.

To select other input folders, set `NOTICE_DIR` and `SPEC_DIR` before running. The paths may be absolute or relative to the current working directory:

```bash
export NOTICE_DIR="dataset/notice_docs"
export SPEC_DIR="dataset/specification_docs"
```

The script reads `.pdf`, `.html`, and `.htm` files recursively. `OUTPUT_DIR` can optionally be set to change where output files and the text cache are written; by default they are written to the current working directory.

## Run

```bash
python generate_research_wordclouds.py
```

The run creates four CSV frequency tables and four PNG images:

- `unigram_notice.csv` and `unigram_notice_cloud.png`
- `bigram_notice.csv` and `bigram_notice_cloud.png`
- `unigram_specfion.csv` and `unigram_specfion_cloud.png`
- `bigram_specfion.csv` and `bigram_specfion_cloud.png`

The `specfion` spelling is the current filename convention in the script. Extracted document text is cached under `cache/pdf_text/`. Documents without enough readable text are skipped; scanned pages may use OCR when the optional Python OCR packages and system Tesseract are available.