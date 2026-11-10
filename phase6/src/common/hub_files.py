"""Finds Phase-1 files: an attached Kaggle dataset first, then a local cache, then Hugging Face."""
import os
import re

from huggingface_hub import hf_hub_download

KAGGLE_INPUT = "/kaggle/input"


def _squash(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_in_kaggle_input(filename, hint):
    """Only accepts a file whose path mentions the architecture (e.g. 'arch3', 'arch-3', 'Arch_3'),
    because arch 2, 3 and 4 all call their weights model_final.pt."""
    if not os.path.isdir(KAGGLE_INPUT):
        return None
    hits = []
    for root, _, files in os.walk(KAGGLE_INPUT):
        if filename in files:
            path = os.path.join(root, filename)
            if _squash(hint) in _squash(path):
                hits.append(path)
    return sorted(hits)[0] if hits else None


def fetch_from_hub(repo, filename, local_dir):
    try:
        return hf_hub_download(repo, filename, local_dir=local_dir, token=os.environ.get("HF_TOKEN"))
    except Exception:
        return None


def locate(filenames, repos, hint, local_dir):
    """Names are tried in order of preference. Returns (path, where it came from)."""
    for name in filenames:
        path = find_in_kaggle_input(os.path.basename(name), hint)
        if path:
            return path, "kaggle input"
    for name in filenames:
        for repo in repos:
            cached = os.path.join(local_dir, _squash(repo), name)
            if os.path.exists(cached):
                return cached, f"cache of hf://{repo}"
            path = fetch_from_hub(repo, name, os.path.join(local_dir, _squash(repo)))
            if path:
                return path, f"hf://{repo}"
    raise FileNotFoundError(f"None of {filenames} found under /kaggle/input (path must mention '{hint}') or in "
                            f"{repos}. Attach the Phase-1 weights as a Kaggle dataset, or check HF_TOKEN and the repo names.")
