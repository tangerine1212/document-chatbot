import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from RAG.vector_database import VectorDatabaseBuilder
from src.utils  import generate_qa_prompt, process_func, process_qa_prompt, genereate_retrieval_prompt, sort_documents_by_doc_page

class DocumentQASystem:
    """A class to perform question answering on documents using a pre-built FAISS vector database."""
    
    def __init__(self, config):
        """Initialize with model, tokenizer, vector database path, and embedding model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=False, trust_remote_code=True)
        self.max_new_tokens = config["max_new_tokens"]
        self.max_context_len = config["max_context_len"]
        self.vb_builder = VectorDatabaseBuilder(config["vector_base"])
        self.vb = None
    
    def build_vector_database(self, file_paths, chunking_method):
        self.faiss_db = self.vb_builder.build_vector_database(file_paths, chunking_method)
        

    def retrieve_documents(self, question, history=None, retrieve_num=4):
        """Retrieve top_k relevant document chunks for the given question."""
        retrieval_instruction = genereate_retrieval_prompt(question, history, self.vb_builder.embedding_tokenizer, history_max_len=500)
    
        retrieved_documents_with_score = self.faiss_db.similarity_search_with_score(retrieval_instruction, k=200)[:retrieve_num]
        retrieved_documents = [item[0] for item in retrieved_documents_with_score]
        retrieved_documents = sort_documents_by_doc_page(retrieved_documents)
        return retrieved_documents

    def answer_question(self, question, history=None, retrieve_num=4):
        """Answer a user's question using retrieved documents and save results."""
        retrieved_documents = self.retrieve_documents(question, retrieve_num=retrieve_num)
        
        qa_instruction = generate_qa_prompt(
            question=question,
            documents=retrieved_documents,
            tokenizer=self.tokenizer,
            max_context_len=self.max_context_len
        )
        
        messages = [
            {"role": "system", "content": "请根据用户的问题和提供的文档片段提供准确的回答。"},
            {"role": "user", "content": qa_instruction}
        ]
        
        response = self.predict(messages)
        
        result_entry = {
            "question": question,
            "history": None,  
            "documents": retrieved_documents,
            "prediction": response
        }
        
        return result_entry

    def predict(self, messages, device="cuda"):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id 
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
