"""Real-Time Incremental RAG on Pathway.

Add / modify / delete on the watched document folder (and the live finance feed)
propagate automatically through Pathway's differential-dataflow ``DocumentStore``
into the vector index — no manual restart, no batch re-indexing.
"""

__version__ = "1.0.0"
