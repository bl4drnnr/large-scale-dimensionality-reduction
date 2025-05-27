from typing import List, Optional
from sentence_transformers import SentenceTransformer
import streamlit as st
from large_scale_dimensionality_reduction.vector_db.db import VectorDB


class Embeddings:
    def __init__(self, vector_db: VectorDB, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vector_db = vector_db

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.model.encode(text).tolist()

    def batch_process_texts(
        self,
        texts: List[str],
        collection_name: str,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 128,
    ):
        """
        Batch process texts to generate embeddings and store them in ChromaDB.

        :param texts: List of texts to embed.
        :param collection_name: Name of the collection in ChromaDB.
        :param metadatas: Optional list of metadata dicts for each text.
        :param ids: Optional list of custom IDs for each text.
        :param batch_size: Number of samples per batch.
        """
        texts = [text for text in texts if text and text.strip()]
        if not texts:
            raise ValueError("No valid texts provided for batch processing.")

        self.vector_db.add_collection(name=collection_name)

        num_batches = (len(texts) + batch_size - 1) // batch_size
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Initializing embedding generation...")
            progress_bar.progress(5)

            for i, batch_start in enumerate(range(0, len(texts), batch_size)):
                batch_texts = texts[batch_start : batch_start + batch_size]
                batch_metadatas = metadatas[batch_start : batch_start + batch_size] if metadatas else None
                batch_ids = ids[batch_start : batch_start + batch_size] if ids else None

                status_text.text(f"Processing batch {i + 1}/{num_batches}...")
                embeddings = self.generate_embedding(batch_texts)

                if batch_metadatas is None:
                    batch_metadatas = [{"source": "batch_process"} for _ in batch_texts]

                self.vector_db.add_items_to_collection(
                    name=collection_name,
                    texts=batch_texts,
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadata=batch_metadatas,
                )

                progress = min(95, int((i + 1) / num_batches * 100))
                progress_bar.progress(progress)

            status_text.text("Embedding generation completed!")
            progress_bar.progress(100)

        finally:

            def cleanup():
                import time

                time.sleep(1)
                progress_bar.empty()
                status_text.empty()

            cleanup()

    def query_similar_texts(self, query_text: str, collection_name: str, top_k: int = 5):
        """
        Query similar texts from a collection in ChromaDB.

        :param query_text: Text to search for similar documents.
        :param collection_name: Name of the collection to search in.
        :param top_k: Number of most similar results to return.
        :return: List of results with id, text, metadata and distance.
        """
        query_embedding = self.generate_embedding(query_text)

        results = self.vector_db.query_collection(
            name=collection_name,
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )

        return formatted_results

    def batch_query_similar_texts(
        self,
        query_texts: List[str],
        collection_name: str,
        top_k: int = 5,
    ) -> List[List[dict]]:
        """
        Batch query similar texts for a list of input queries.

        :param query_texts: List of query texts.
        :param collection_name: Collection to search in.
        :param top_k: Number of top results per query.
        :return: List of results for each query.
        """
        query_embeddings = self.generate_embedding(query_texts)

        results = self.vector_db.query_collection(
            name=collection_name,
            query_embeddings=query_embeddings,
            n_results=top_k,
        )

        batch_results = []
        for query_idx in range(len(query_texts)):
            single_query_results = []
            for i in range(len(results["ids"][query_idx])):
                single_query_results.append(
                    {
                        "id": results["ids"][query_idx][i],
                        "text": results["documents"][query_idx][i],
                        "metadata": results["metadatas"][query_idx][i],
                        "distance": results["distances"][query_idx][i],
                    }
                )
            batch_results.append(single_query_results)

        return batch_results
