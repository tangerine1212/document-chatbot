import numpy as np
import matplotlib.pyplot as plt
from langchain_community.document_loaders import Docx2txtLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from scipy.spatial.distance import cosine
import jieba
import re

def read_docx(file_path):
    """Read text from a DOCX file using Docx2txtLoader."""
    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    return documents[0].page_content

def get_sentences(text):
    """Split Chinese text into sentences using regex, retaining punctuation."""
    sentence_enders = r'([。？！\n]+)'
    segments = re.split(sentence_enders, text)
    sentences = []
    for i in range(0, len(segments)-1, 2):
        sentence = segments[i] + (segments[i+1] if i+1 < len(segments) else '')
        if sentence.strip():
            sentences.append(sentence)
    if len(segments) % 2 == 1 and segments[-1].strip():
        sentences.append(segments[-1])
    return [s for s in sentences if s.strip()]

def get_sentence_embeddings(sentences, model):
    """Generate embeddings for a list of sentences using a sentence transformer model."""
    return model.embed_documents(sentences)

def plot_chunks(distances, breakpoint_distance_threshold, pairs):
    """Plot the cosine distances with chunk breakpoints and shaded regions based on pairs."""
    plt.plot(distances)
    y_upper_bound = 0.5
    plt.ylim(0, y_upper_bound)
    plt.xlim(0, len(distances))
    plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')
    num_chunks = len(pairs) + 1  # Number of chunks is number of pairs + 1
    plt.text(x=(len(distances)*.01), y=y_upper_bound/50, s=f"{num_chunks} Chunks")
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, (start_index, end_index) in enumerate(pairs):
        plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
        plt.text(x=np.average([start_index, end_index]),
                 y=breakpoint_distance_threshold + (y_upper_bound)/20,
                 s=f"Chunk #{i+1}", horizontalalignment='center',
                 rotation='vertical')
    
    if pairs:
        last_end = pairs[-1][1]
        if last_end < len(distances):
            plt.axvspan(last_end, len(distances), facecolor=colors[len(pairs) % len(colors)], alpha=0.25)
            plt.text(x=np.average([last_end, len(distances)]),
                     y=breakpoint_distance_threshold + (y_upper_bound)/20,
                     s=f"Chunk #{len(pairs)+1}",
                     rotation='vertical')
    
    plt.title("Text Chunks Based On Embedding Breakpoints")
    plt.xlabel("Index of sentences in text (Sentence Position)")
    plt.ylabel("Cosine distance")
    plt.show()

def compute_sliding_window_distances(sentences, model, window_size=3):
    """Compute cosine distances between consecutive sliding windows of concatenated sentences."""
    distances = []
    for i in range(window_size, len(sentences)):
        prev_window_sentences = sentences[i-window_size-1:i]
        curr_window_sentences = sentences[i-window_size:i+1]
        prev_window_text = "".join(prev_window_sentences)
        curr_window_text = "".join(curr_window_sentences)
        window_embeddings = model.embed_documents([prev_window_text, curr_window_text])
        prev_embedding = window_embeddings[0]
        curr_embedding = window_embeddings[1]
        distance = cosine(prev_embedding, curr_embedding)
        distances.append(distance)
    return distances

def create_pairs(numbers, length):
    pairs = []
    indices = []
    pairs.append(0)
    if numbers[0] == 0:
        i = 1
    else:
        i = 0
    while i < len(numbers) - 1:  
        consecutive_group = [numbers[i]]
        j = i+1
        while j < len(numbers) and numbers[j] - numbers[j-1] == 1:
            consecutive_group.append(numbers[j])
            j += 1
        
        if len(consecutive_group) > 1:
            pairs.append(consecutive_group[-1])
            pairs.append(consecutive_group[0])
        else:
            pairs.append(numbers[i])
            pairs.append(numbers[i])
        i = j
    for i in range(0, len(pairs), 2):
        if i+1 < len(pairs):
            indices.append((pairs[i], pairs[i+1]))
    indices.append((pairs[-1], length))
    return indices

def get_chunks(sentences, pairs, window_size=3, chunk_overlap=0):
    """Split sentences into chunks based on breakpoint indices, handling consecutive indices with overlap."""
    chunks = [] 
    for pair in pairs:
        chunks.append("".join(sentences[pair[0]:pair[1]]))
    return chunks

def semantic_chunking(document_text, embeddings, window_size=3, percentile_threshold=92, chunk_overlap=2):
    """Perform semantic chunking on a DOCX file."""
    sentences = get_sentences(document_text)
    if not sentences:
        raise ValueError("No sentences found in the document.")
    distances = compute_sliding_window_distances(sentences, embeddings, window_size)
    
    breakpoint_distance_threshold = np.percentile(distances, percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]
    pairs = create_pairs(indices_above_thresh, len(sentences))
    # plot_chunks(distances, breakpoint_distance_threshold, pairs)
    
    chunks = get_chunks(sentences, pairs, window_size, chunk_overlap=chunk_overlap)
    
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    return documents

def recursive_chunking(documents, chunk_size=500, chunk_overlap=0):
    """Perform recursive character-based chunking."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";"],
    )
    return text_splitter.create_documents([documents])

if __name__ == "__main__":
    file_path = "../大语言模型.docx"
    model_name = "../models/bge-m3-hit-ft"
    embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"},
        )
    document = read_docx(file_path)
    chunks = semantic_chunking(document, embeddings=embeddings, window_size=5, percentile_threshold=92)
    # chunks = recursive_chunking(document)
    result = ""
    for i, chunk in enumerate(chunks):
        # print(f"\nChunk #{i}:\n{chunk}\n{'-'*50}")
        result += f"\nChunk #{i}:\n{chunk.page_content}\n{'-'*50}"
    with open("result_recursive.txt", 'w', encoding="utf-8") as file:
        file.write(result)