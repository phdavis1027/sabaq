import time

import git


def gen_run_id(module: str, source: str, language: str) -> str:
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    return f"{module}-{source}-{language}-{sha}-{int(time.time())}"
