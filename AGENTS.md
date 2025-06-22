# `AGENTS.md` - Sabaq

## The Golden Rule
When unsure about implementation details, ALWAYS ask the developer.

## Project Context
Sabaq accelerates language learning by automatically extracting flashcards for unfamiliar vocabulary out of native texts.

## Critical Architecture Decisions

### Components
- **scrape_tools** utilities for scraping and managing training data
- **sabaq_llm** models used for training idiom-detection LLM models and converting data into a format they understand
- **sabaq_server** the server which serves sabaq's public API

### Why MongoDB for training data?
Open document structures that allows quick iteration when dealing with non-uniform datasets. This might be changed in the future.

### Why Django for the server?
- Out-of-the-box auth
- Keep all the code in Python
- Managed migrations
- Transparent ORM keeps database relatively portable

### Why Scrapy for scraping?
- Able to re-use pipeline components between different data sources

## Code Style and Patterns

### Anchor comments

Add specially formatted comments throughout the codebase, where appropriate, for yourself as inline knowledge that can be easily `grep`ped for.

### Type annotations

If you know the type of a variable, always annotate it. For example:

```python
from django.db import models
def my_function(arg1: str, arg2: int, arg3: models.Model) -> bool:
	pass
```

### Guidelines:

- Use `AIDEV-NOTE:`, `AIDEV-TODO:`, or `AIDEV-QUESTION:` (all-caps prefix) for comments aimed at AI and developers.
- **Important:** Before scanning files, always first try to **grep for existing anchors** `AIDEV-*` in relevant subdirectories.
- **Update relevant anchors** when modifying associated code.
- **Do not remove `AIDEV-NOTE`s** without explicit human instruction.
- Make sure to add relevant anchor comments, whenever a file or piece of code is:
  * too complex, or
  * very important, or
  * confusing, or
  * could have a bug

## Domain Glossary (Agent, learn these!)

- **BERT** - Bidirectional encoder representations from transformers (BERT). The base model used for idiom detection.
- **SpaCy** - Production-grade NLP toolkit

## What AI Must NEVER Do

1. **Never modify test files** - Tests encode human intent
2. **Never change API contracts** - Breaks real applications
3. **Never alter migration files** - Data loss risk
4. **Never commit secrets** - Use environment variables
5. **Never assume business logic** - Always ask
6. **Never remove AIDEV- comments** - They're there for a reason

Remember: We optimize for maintainability over cleverness.
When in doubt, choose the boring solution.
