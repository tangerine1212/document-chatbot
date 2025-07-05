import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from .semantic_chunking import semantic_chunking
from .parser import load_file, load_url, split_chunks, PDFProcesser
import numpy as np
import matplotlib.pyplot as plt
import torch

class VectorDatabaseBuilder:
    """A class to load documents, chunk them, and create a FAISS vector database."""
    
    def __init__(self, config):
        """Initialize with embedding model and cache directory."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["model_name"],
            model_kwargs={"model_kwargs" :{"torch_dtype": torch.float16}}
        )
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.chunking_method = config["method"]
        self.pdf_processer = PDFProcesser(config["ocr_model_path"])
        self.config = config

    def get_sentence_embeddings(self, sentences, model):
        """Generate embeddings for a list of sentences."""
        return model.embed_documents(sentences)

    def plot_chunks(self, distances, breakpoint_distance_threshold, indices_above_thresh):
        """Plot cosine distances with chunk breakpoints and shaded regions."""
        plt.plot(distances)
        y_upper_bound = 0.15
        plt.ylim(0, y_upper_bound)
        plt.xlim(0, len(distances))
        plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')
        num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold])
        plt.text(x=(len(distances)*.01), y=y_upper_bound/50, s=f"{num_distances_above_theshold + 1} Chunks")
        
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, breakpoint_index in enumerate(indices_above_thresh):
            start_index = 0 if i == 0 else indices_above_thresh[i - 1]
            end_index = breakpoint_index if i <= len(indices_above_thresh) - 1 else len(distances)
            plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
            plt.text(x=np.average([start_index, end_index]),
                     y=breakpoint_distance_threshold + (y_upper_bound)/20,
                     s=f"Chunk #{i}", horizontalalignment='center',
                     rotation='vertical')
        if indices_above_thresh:
            last_breakpoint = indices_above_thresh[-1]
            if last_breakpoint < len(distances):
                plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
                plt.text(x=np.average([last_breakpoint, len(distances)]),
                         y=breakpoint_distance_threshold + (y_upper_bound)/20,
                         s=f"Chunk #{i+1}",
                         rotation='vertical')
        plt.title("Text Chunks Based On Embedding Breakpoints")
        plt.xlabel("Index of sentences in text (Sentence Position)")
        plt.ylabel("Cosine distance between sequential sentences")
        plt.show()

    def recursive_chunking(self, documents, chunk_size=500, chunk_overlap=0):
        """Perform recursive character-based chunking."""
        def length_function(s: str):
            return len(self.embedding_tokenizer.tokenize(s))
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";"],
            s_separator_regex=True,
        )
        return text_splitter.create_documents([documents])

    def build_vector_database(self, file_paths, chunking_method):
        """Build and save a FAISS vector database from multiple documents using specified chunking method."""
        if not isinstance(file_paths, (list, tuple)):
            file_paths = [file_paths] 
        
        all_docs = []
        for file_path in file_paths:
            document_text = load_file(file_path, self.pdf_processer)
            file_name = self.extract_file_name(file_path, self.embedding_tokenizer, max_len=None)
            if chunking_method == "recursive":
                docs = self.recursive_chunking(document_text, **self.config["recursive"])
            elif chunking_method == "semantic":
                docs = semantic_chunking(document_text, self.embeddings, **self.config["semantic"]) 
            else:
                raise ValueError(f"Unsupported chunking method: {self.chunking_method}. Choose 'recursive' or 'semantic'.")
            
            for i, doc_page in enumerate(docs):
                doc_page.metadata["url"] = file_path
                doc_page.metadata["page_id"] = i + 1
                doc_page.metadata["title"] = file_name
                doc_page.page_content = "{}\t{}".format(file_name, doc_page.page_content)
            
            all_docs.extend(docs)
        
        if not all_docs:
            raise ValueError("No chunks generated from any of the provided documents.")
        
        vectordb = FAISS.from_documents(documents=all_docs, embedding=self.embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)

        save_path = os.path.join("./vector_database", "faiss_index")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vectordb.save_local(save_path)
        
        return vectordb
    
    def extract_file_name(self, file_path, tokenizer, max_len):
        file_name = os.path.basename(file_path)

        file_basic_name, file_extension = os.path.splitext(file_name)
        file_basic_name_ids = tokenizer(file_basic_name, add_special_tokens=False)

        if max_len is not None and len(file_basic_name_ids) > max_len:
            ellipsis_ids = tokenizer("...", add_special_tokens=False)
            file_name = tokenizer.decode(file_basic_name_ids[:max_len//2] + ellipsis_ids + file_basic_name_ids[-(max_len//2):]) + file_extension
        
        return file_name