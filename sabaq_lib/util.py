# stdlib

import time
import os
import sys

# 3rd party

import git

import spacy
from spacy.language import Language as SpacyLanguage
import spacy.util as spacy_util

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


def gen_run_id(module: str, source: str, language: str) -> str:
    return f"{module}-{source}-{language}-{sha}-{int(time.time())}"


def find_path_to_site_packages() -> str | None:
    for p in sys.path:
        if "site-packages" in p:
            return p
    return None


def load_spacy_model(model: str) -> SpacyLanguage:
    version = spacy_util.get_package_version(model)
    cfg_dir = f"{model}-{version}"

    spacy_model_path = os.path.join(find_path_to_site_packages(), model, cfg_dir)

    return spacy.load(spacy_model_path)
