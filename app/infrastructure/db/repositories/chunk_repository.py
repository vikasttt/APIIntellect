"""SQLAlchemy async implementation of IChunkRepository with pgvector."""
from __future__ import annotations
import uuid
from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, func, select, text, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.entities.chunk import Chunk
from app.domain.repositories.chunk_repository import IChunkRepository
from app.infrastructure.db.models import ChunkModel
class SqlChunkRepository(IChunkRepository):
    """pgvector-backed chunk repository with similarity and hybrid search."""
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
    # ─── Mapping helpers ────────────────────────────────────────────────────
    @staticmethod
    def _to_entity(model: ChunkModel) -> Chunk:
        return Chunk(
            id=model.id,
            document_id=model.document_id,
            user_id=str(model.user_id),
            content=model.content,
            chunk_index=model.chunk_index,
            token_count=model.token_count,
            embedding=model.embedding,
            metadata=model.chunk_metadata or {},
            page_number=model.page_number,
            section_title=model.section_title,
            chunk_type=model.chunk_type,
            created_at=model.created_at,
        )
    @staticmethod
    def _to_model(entity: Chunk) -> ChunkModel:
        return ChunkModel(
            id=entity.id,
            document_id=entity.document_id,
            user_id=entity.user_id,
            content=entity.content,
            chunk_index=entity.chunk_index,
            token_count=entity.token_count,
            embedding=entity.embedding,
            chunk_metadata=entity.metadata,
            page_number=entity.page_number,
            section_title=entity.section_title,
            chunk_type=entity.chunk_type,
            created_at=entity.created_at,
        )
    # ─── IChunkRepository ───────────────────────────────────────────────────
    async def bulk_create(self, chunks: list[Chunk]) -> list[Chunk]:
        models = [self._to_model(c) for c in chunks]
        self._session.add_all(models)
        await self._session.flush()
        return [self._to_entity(m) for m in models]
    async def get_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        result = await self._session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.chunk_index)
        )
        return [self._to_entity(m) for m in result.scalars().all()]
    async def delete_by_document(self, document_id: uuid.UUID) -> int:
        result = await self._session.execute(
            delete(ChunkModel).where(ChunkModel.document_id == document_id)
        )
        await self._session.flush()
        return result.rowcount  # type: ignore[return-value]
    async def similarity_search(
        self,
        embedding: list[float],
        *,
        user_id: str,
        top_k: int = 10,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Cosine similarity search using pgvector <=> operator."""
        emb_param = cast(embedding, Vector(3072))
        similarity = (1 - ChunkModel.embedding.cosine_distance(embedding)).label("score")
        query = (
            select(ChunkModel, similarity)
            .where(ChunkModel.user_id == user_id)
            .where(ChunkModel.embedding.is_not(None))
        )
        if document_ids:
            query = query.where(ChunkModel.document_id.in_(document_ids))
        query = query.order_by(similarity.desc()).limit(top_k)
        result = await self._session.execute(query)
        return [(self._to_entity(row[0]), float(row[1])) for row in result.all()]
    async def hybrid_search(
        self,
        embedding: list[float],
        query_text: str,
        *,
        user_id: str,
        top_k: int = 10,
        alpha: float = 0.7,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Hybrid search: Reciprocal Rank Fusion of vector search + full-text search.
        alpha=1.0 → pure vector, alpha=0.0 → pure keyword
        """
        emb_param = cast(embedding, Vector(3072))
        # Vector candidates
        vector_query = (
            select(
                ChunkModel,
                func.row_number()
                .over(
                    order_by=(1 - ChunkModel.embedding.op("<=>")(emb_param)).desc()
                )
                .label("vector_rank"),
            )
            .where(ChunkModel.user_id == user_id)
            .where(ChunkModel.embedding.is_not(None))
        )
        # Full-text candidates (PostgreSQL tsvector)
        ts_query = func.plainto_tsquery("english", query_text)
        fts_query = (
            select(
                ChunkModel,
                func.row_number()
                .over(
                    order_by=func.ts_rank(
                        func.to_tsvector("english", ChunkModel.content), ts_query
                    ).desc()
                )
                .label("fts_rank"),
            )
            .where(ChunkModel.user_id == user_id)
            .where(
                func.to_tsvector("english", ChunkModel.content).op("@@")(ts_query)
            )
        )
        if document_ids:
            vector_query = vector_query.where(ChunkModel.document_id.in_(document_ids))
            fts_query = fts_query.where(ChunkModel.document_id.in_(document_ids))
        vector_query = vector_query.limit(top_k)
        fts_query = fts_query.limit(top_k)
        # Execute both in parallel via union approach (RRF scoring)
        v_result = await self._session.execute(vector_query)
        f_result = await self._session.execute(fts_query)
        rrf_k = 60
        scores: dict[uuid.UUID, tuple[Chunk, float]] = {}
        for model, rank in v_result.all():
            chunk = self._to_entity(model)
            rrf_score = alpha * (1.0 / (rrf_k + rank))
            scores[chunk.id] = (chunk, scores.get(chunk.id, (chunk, 0.0))[1] + rrf_score)
        for model, rank in f_result.all():
            chunk = self._to_entity(model)
            rrf_score = (1 - alpha) * (1.0 / (rrf_k + rank))
            if chunk.id in scores:
                old_chunk, old_score = scores[chunk.id]
                scores[chunk.id] = (old_chunk, old_score + rrf_score)
            else:
                scores[chunk.id] = (chunk, rrf_score)
        return sorted(scores.values(), key=lambda x: x[1], reverse=True)[:top_k]
