from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


def split_chunks(
        document: list, 
        chunk_size: int = 500, 
        chunk_overlap: int = 0, 
        tokenizer: AutoTokenizer = None
    ) -> list:
    def length_function(s: str):
        return len(tokenizer.tokenize(s))
    
    if len(document) == 0:
        return []

    chunk_overlap = min(chunk_overlap, chunk_size // 2)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";"],
        is_separator_regex=True,
    )

    doc_pages = text_splitter.create_documents([document])
    return doc_pages