"""Load data/faq.json into the pgvector knowledge base.

Run with: python -m app.rag.seed
"""

import json
from pathlib import Path

from app.rag.store import FaqEntry, init_db, upsert_entries

DATA_FILE = Path(__file__).resolve().parents[2] / "data" / "faq.json"


def main() -> None:
    entries = json.loads(DATA_FILE.read_text())
    init_db()
    count = upsert_entries(entries)
    print(f"Loaded {count} entries into {FaqEntry.__tablename__}")


if __name__ == "__main__":
    main()
