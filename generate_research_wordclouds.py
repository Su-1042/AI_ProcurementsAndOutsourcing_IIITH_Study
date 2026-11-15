#!/usr/bin/env python3
"""Unigram and bigram word clouds for two folders of PDFs (and HTML pages) (notice, specfion).

Fully automatic, rule-based noise removal (no manual review step).
Outputs: unigram_{notice,specfion}.csv, bigram_{notice,specfion}.csv and the four
corresponding word cloud PNGs.

Usage:
    NOTICE_DIR=path/to/notice SPEC_DIR=path/to/specfion python make_wordclouds.py
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from tqdm import tqdm
from wordcloud import WordCloud

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
CACHE_DIR = OUTPUT_DIR / "cache" / "pdf_text"
MIN_DF = {"notice": 10, "specfion": 5}      # min documents a term must appear in
MIN_BIGRAM_COUNT = 5
RAW_MIN_CHARS = 200                          # below this the whole PDF is skipped
OCR_PAGE_MIN_CHARS = 30                      # pages with less text get OCR (if available)
RANDOM_STATE = 42

# "after_stopwords": bigrams built from the cleaned text after stopword removal, so
#   "reserves the right" -> "reserve right", "operation and maintenance" -> "operation
#   maintenance". This matches the methodology text and figures in the paper.
# "before_stopwords": a stopword breaks the pair (stricter, fewer artifacts, but
#   "reserve right" / "operation maintenance" / "supply installation" will not form).
BIGRAM_MODE = "after_stopwords"

# Genuine boilerplate only. Do NOT add substantive terms (condition, document,
# agreement, contract, bidder ...): the paper treats them as findings.
FILLER_TERMS = {
    "dated", "along", "hereby", "thereof", "herein", "aforesaid", "sl", "sr", "etc",
    "xxxx", "please", "refer", "page", "annexure", "sheet", "enclosed", "mentioned",
    "shall", "also",
}

# Short tokens that are real terms in this corpus but have no vowels / are short.
KEEP_TOKENS = {"rfp", "gst"}

# Never dropped by the NER (PERSON/GPE) filter, which misfires on tender text.
# This does not protect them from the DF threshold.
PROTECTED = {
    "contract", "document", "work", "submission", "bidder", "eoi", "signed", "condition",
    "agreement", "party", "contractor", "payment", "certificate", "deposit", "security",
    "process", "completion", "service", "data", "bank", "rfp", "support", "technical",
    "criterion", "chatbot", "video", "detection", "camera", "analytics", "platform",
    "provider", "compliance", "audit", "bias", "oversight", "fairness", "transparency",
    "accountability", "ethic", "ethical",
}


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PAGE_RE = re.compile(r"\b(?:page|pg|pp)\s*[:#-]?\s*\d+(?:\s*(?:of|/)\s*\d+)?\b", re.I)
REPEAT_RE = re.compile(r"(.)\1{2,}")
VOWEL_RE = re.compile(r"[aeiouy]")

# Lemmatizer misses seen in the first run (e.g. "data" -> "datum").
LEMMA_FIX = {"datum": "data", "specifications": "specification",
             "conditions": "condition", "terms": "term"}
ALLCAPS_RE = re.compile(r"\b[A-Z]{3,}\b")  # ALL-CAPS words defeat the lemmatizer

BREAK = None          # hard break (number, punctuation token, junk, entity)
SKIP = "<skip>"       # short alphabetic token (of, to, in ...): behaves like a stopword
CHUNK_CHARS = 100_000


# ----------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------
def resolve_folder(name: str) -> Path:
    env = os.environ.get("NOTICE_DIR" if name == "notice" else "SPEC_DIR")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser().resolve()
    direct = Path.cwd() / name
    if direct.exists():
        return direct.resolve()
    key = "notice" if name == "notice" else "spec"
    for root in (Path.cwd(), Path.cwd() / "dataset"):
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir() and key in child.name.lower():
                    return child.resolve()
    raise FileNotFoundError(f"Folder for '{name}' not found. Set NOTICE_DIR / SPEC_DIR.")


def get_stopwords() -> set:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def get_nlp() -> spacy.Language:
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = CHUNK_CHARS * 2
    return nlp


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------
DOC_SUFFIXES = {".pdf", ".html", ".htm"}


class _HTMLText(HTMLParser):
    """Collect visible text, ignoring script/style/noscript blocks."""

    def __init__(self):
        super().__init__()
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip += 1
        elif tag in {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def extract_html(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    parser = _HTMLText()
    parser.feed(html)
    return unescape("".join(parser.parts))


def extract_text(pdf_path: Path, folder: str) -> Tuple[str, str]:
    """Return (text, status). status in {cache, ok, ocr, skip}. Cached per folder+path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(str(pdf_path.resolve()).encode()).hexdigest()[:8]
    cache_file = CACHE_DIR / f"{folder}__{pdf_path.stem}__{key}.txt"
    if cache_file.exists():
        try:
            return cache_file.read_text(encoding="utf-8"), "cache"
        except Exception:
            pass

    if pdf_path.suffix.lower() in {".html", ".htm"}:
        try:
            text = extract_html(pdf_path)
        except Exception:
            text = ""
        if len(text.strip()) < RAW_MIN_CHARS:
            return "", "skip"
        try:
            cache_file.write_text(text, encoding="utf-8")
        except Exception:
            pass
        return text, "ok"

    text, status = "", "ok"
    if fitz is not None:
        try:
            parts = []
            with fitz.open(str(pdf_path)) as doc:
                for page in doc:
                    t = page.get_text("text") or ""
                    if len(t.strip()) < OCR_PAGE_MIN_CHARS and pytesseract and Image:
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            ocr = pytesseract.image_to_string(img)
                            if len(ocr.strip()) > len(t.strip()):
                                t, status = ocr, "ocr"
                        except Exception:
                            pass
                    parts.append(t)
            text = "\n".join(parts)
        except Exception:
            text = ""

    if len(text.strip()) < RAW_MIN_CHARS and pdfplumber is not None:
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            text = ""

    if len(text.strip()) < RAW_MIN_CHARS:
        return "", "skip"
    try:
        cache_file.write_text(text, encoding="utf-8")
    except Exception:
        pass
    return text, status


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def _chunks(text: str) -> List[str]:
    """Split on line boundaries into chunks under CHUNK_CHARS (spaCy length limit)."""
    out, cur, size = [], [], 0
    for line in text.split("\n"):
        if size + len(line) > CHUNK_CHARS and cur:
            out.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        out.append("\n".join(cur))
    return out


def clean_text(text: str, nlp: spacy.Language) -> List[List[Optional[str]]]:
    """Return sentences as lists of lemmas.

    BREAK (None) marks a hard break; SKIP marks a short alphabetic token. Stopword and
    filler handling happens later, so both bigram modes can be supported. NER runs on
    the original-case text and only removes the flagged token itself.
    """
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PAGE_RE.sub(" ", text)
    text = ALLCAPS_RE.sub(lambda m: m.group().lower(), text)
    text = re.sub(r"[ \t]+", " ", text)

    sentences: List[List[Optional[str]]] = []
    for doc in nlp.pipe(_chunks(text), batch_size=1):
        for sent in doc.sents:
            seq: List[Optional[str]] = []
            for tok in sent:
                if tok.is_space:
                    continue
                if not tok.is_alpha:
                    seq.append(BREAK)
                    continue
                lemma = LEMMA_FIX.get(tok.lemma_.lower(), tok.lemma_.lower())
                if len(lemma) < 3 and lemma not in KEEP_TOKENS:
                    seq.append(SKIP)
                    continue
                if REPEAT_RE.search(lemma):
                    seq.append(BREAK)
                    continue
                if lemma not in KEEP_TOKENS and not VOWEL_RE.search(lemma):
                    seq.append(BREAK)
                    continue
                if tok.ent_type_ in {"PERSON", "GPE"} and lemma not in PROTECTED:
                    seq.append(BREAK)
                    continue
                seq.append(lemma)
            sentences.append(seq)
    return sentences


def unigram_tokens(sentences, stop: set) -> List[str]:
    return [t for s in sentences for t in s
            if t not in (BREAK, SKIP) and t not in stop and t not in FILLER_TERMS]


def bigrams_from(sentences, stop: set, keep: set) -> List[Tuple[str, str]]:
    out = []
    for sent in sentences:
        prev = None
        for t in sent:
            if t is BREAK:
                prev = None
            elif t == SKIP or t in stop or t in FILLER_TERMS:
                if BIGRAM_MODE == "before_stopwords":
                    prev = None
                # after_stopwords: skip over it and keep the previous token
            elif t not in keep:
                prev = None
            else:
                if prev is not None:
                    out.append((prev, t))
                prev = t
    return out


# ----------------------------------------------------------------------------
# Filtering and counting
# ----------------------------------------------------------------------------
def build_vocab_filter(folder: str, doc_tokens: List[List[str]], min_df: int,
                       stop: set) -> Tuple[set, List[dict], Counter, Counter]:
    """DF-based filter. Returns keep set, removal rows, raw TF and raw DF (pre-filter)."""
    tf, df = Counter(), Counter()
    for toks in doc_tokens:
        tf.update(toks)
        df.update(set(toks))
    n = max(len(doc_tokens), 1)
    keep, removed = set(), []
    for term in sorted(tf):
        reasons = []
        if df[term] < min_df:
            reasons.append(f"df<{min_df}")
        if df[term] > 0.9 * n and (term in stop or term in FILLER_TERMS):
            reasons.append("ubiquitous_stopword_or_filler")
        if reasons:
            removed.append({"term": term, "folder": folder, "total_tf": tf[term],
                            "doc_freq": df[term], "reason": "; ".join(reasons)})
        else:
            keep.add(term)
    return keep, removed, tf, df


def count_unigrams(doc_tokens: List[List[str]], keep: set) -> pd.DataFrame:
    tf, df = Counter(), Counter()
    for toks in doc_tokens:
        kept = [t for t in toks if t in keep]
        tf.update(kept)
        df.update(set(kept))
    rows = [{"term": t, "count": c, "doc_freq": df[t]} for t, c in tf.items()]
    return _sorted(pd.DataFrame(rows, columns=["term", "count", "doc_freq"]))


def count_bigrams(doc_sentences, keep: set, min_df: int, stop: set) -> pd.DataFrame:
    tf, df = Counter(), Counter()
    for sents in doc_sentences:
        bgs = bigrams_from(sents, stop, keep)
        tf.update(bgs)
        df.update(set(bgs))
    rows = [{"term": f"{a} {b}", "count": c, "doc_freq": df[(a, b)]}
            for (a, b), c in tf.items() if c >= MIN_BIGRAM_COUNT and df[(a, b)] >= min_df]
    return _sorted(pd.DataFrame(rows, columns=["term", "count", "doc_freq"]))


def _sorted(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    return d.sort_values(["count", "term"], ascending=[False, True]).reset_index(drop=True)


def combine(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([a, b], ignore_index=True)
    if both.empty:
        return both
    return _sorted(both.groupby("term", as_index=False)[["count", "doc_freq"]].sum())


# ----------------------------------------------------------------------------
# Folder processing
# ----------------------------------------------------------------------------
def process_folder(folder: str, path: Path, nlp, stop: set) -> dict:
    pdfs = sorted(p for p in path.rglob("*") if p.suffix.lower() in DOC_SUFFIXES)
    doc_sents, skipped, ocr_used = [], [], []
    for pdf in tqdm(pdfs, desc=f"{folder}", unit="file"):
        text, status = extract_text(pdf, folder)
        if status == "skip":
            skipped.append(pdf.name)
            continue
        if status == "ocr":
            ocr_used.append(pdf.name)
        doc_sents.append(clean_text(text, nlp))

    doc_tokens = [unigram_tokens(s, stop) for s in doc_sents]
    keep, removed, raw_tf, raw_df = build_vocab_filter(folder, doc_tokens, MIN_DF[folder], stop)
    return {
        "found": len(pdfs), "processed": len(doc_sents), "skipped": skipped, "ocr": ocr_used,
        "removed": removed, "raw_tf": raw_tf, "raw_df": raw_df,
        "unigram": count_unigrams(doc_tokens, keep),
        "bigram": count_bigrams(doc_sents, keep, MIN_DF[folder], stop),
    }


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def make_wordcloud(freqs: Dict[str, int], out: Path, colormap: str):
    if not freqs:
        fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=20, color="gray")
        fig.savefig(out, dpi=100, facecolor="white")
        plt.close(fig)
        return
    WordCloud(width=1600, height=900, background_color="white", max_words=150,
              colormap=colormap, collocations=False,
              random_state=RANDOM_STATE).generate_from_frequencies(freqs).to_file(str(out))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stop = get_stopwords()
    nlp = get_nlp()
    paths = {"notice": resolve_folder("notice"), "specfion": resolve_folder("specfion")}

    res = {f: process_folder(f, p, nlp, stop) for f, p in paths.items()}
    cmaps = {"notice": "Blues", "specfion": "YlOrRd"}
    for folder in ["notice", "specfion"]:
        for kind in ["unigram", "bigram"]:
            df = res[folder][kind]
            df.to_csv(OUTPUT_DIR / f"{kind}_{folder}.csv", index=False)
            freqs = {r.term: int(r.count) for r in df.itertuples()}
            make_wordcloud(freqs, OUTPUT_DIR / f"{kind}_{folder}_cloud.png", cmaps[folder])

    for folder, r in res.items():
        print(f"{folder}: {r['processed']} of {r['found']} documents processed")
        for name in r["skipped"]:
            print(f"  skipped (no readable text): {name}")


if __name__ == "__main__":
    main()