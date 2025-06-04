from mxbai_rerank import MxbaiRerankV2
if __name__=='__main__':
    # Initialize the reranker
    reranker = MxbaiRerankV2("mixedbread-ai/mxbai-rerank-base-v2")  # or large-v2

    # Example query and documents
    query = "Who wrote 'To Kill a Mockingbird'?"
    documents = [
        "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960.",
        "The novel 'Moby-Dick' was written by Herman Melville.",
        "Harper Lee was born in 1926 in Monroeville, Alabama."
    ]

    results = reranker.rank(query=query, documents=documents)

    print(results)