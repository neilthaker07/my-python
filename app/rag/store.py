"""pgvector-backed knowledge base of task-management FAQ entries."""

from dataclasses import dataclass

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index, String, Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import settings
from app.rag.embeddings import EMBEDDING_DIM, embed_documents, embed_query

# Separate from app.database: tasks and the knowledge base can live in different databases.
engine = create_engine(settings.knowledge_db_url)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class KnowledgeBase(DeclarativeBase):
    pass


class FaqEntry(KnowledgeBase):
    __tablename__ = "faq_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    category: Mapped[str] = mapped_column(String(50), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    __table_args__ = (
        # Approximate nearest-neighbour index for cosine distance. Ten rows don't
        # need it, but this is how you'd keep search fast with thousands.
        Index(
            "ix_faq_entries_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


@dataclass
class SearchHit:
    id: int
    category: str
    question: str
    answer: str
    score: float  # cosine similarity, 1.0 = identical meaning


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    KnowledgeBase.metadata.create_all(engine)


def upsert_entries(entries: list[dict]) -> int:
    """Embed and insert entries; existing ids are overwritten, so re-seeding is safe."""
    # Embed question and answer together so a search matches on either one.
    vectors = embed_documents([f"{e['question']}\n{e['answer']}" for e in entries])
    with SessionLocal() as db:
        for entry, vector in zip(entries, vectors):
            db.merge(FaqEntry(**entry, embedding=vector))
        db.commit()
    return len(entries)


def search(query: str, top_k: int = 3) -> list[SearchHit]:
    distance = FaqEntry.embedding.cosine_distance(embed_query(query))
    stmt = select(FaqEntry, distance.label("distance")).order_by(distance).limit(top_k)
    with SessionLocal() as db:
        rows = db.execute(stmt).all()
    return [
        SearchHit(
            id=entry.id,
            category=entry.category,
            question=entry.question,
            answer=entry.answer,
            score=round(1 - dist, 3),
        )
        for entry, dist in rows
    ]
