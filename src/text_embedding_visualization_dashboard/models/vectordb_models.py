from typing import Literal

type QUERY_INCLUDE = list[Literal["embeddings", "metadatas", "documents", "distances"]]
type GET_INCLUDE = list[Literal["embeddings", "metadatas", "documents"]]
