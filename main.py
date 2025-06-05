from mxbai_rerank import MxbaiRerankV2


def print_list(str_list):
    for item in str_list:
        print(item)


if __name__ == '__main__':
    # Initialize the reranker
    reranker = MxbaiRerankV2(r"E:\aiModel\maxkbModel\rerank\mxbai-rerank-base-v2")  # or large-v2

    # Example query and documents
    # query = "Who wrote 'To Kill a Mockingbird'?"
    # documents = [
    #     "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960.",
    #     "The novel 'Moby-Dick' was written by Herman Melville.",
    #     "Harper Lee was born in 1926 in Monroeville, Alabama."
    # ]
    query = "乱世佳人绝对是一部杰作"
    documents = [
        "早上好，我想飞上天是一首杰作",
        "乱世佳人是一部叙事拙劣的作品",
        "乱世佳人是一部经典作品",
        "红粉佳人是一部普通的文学作品",
        "吃饭两年半睡觉时长的练习生绝对是一个非凡的人",
        "现在的NBA球风偏软现象绝对是肖华的杰作"
    ]
    results = reranker.rank(query=query, documents=documents)

    print_list(results)
