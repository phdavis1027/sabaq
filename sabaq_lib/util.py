import time
import sys

import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


def gen_run_id(module: str, source: str, language: str) -> str:
    return f"{module}-{source}-{language}-{sha}-{int(time.time())}"


def find_path_to_site_packages() -> str:
    for p in sys.path:
        if "site-packages" in p:
            return p
    return None
