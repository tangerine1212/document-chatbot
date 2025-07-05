import gradio as gr
import json
import os
from typing import List, Dict, Any, Tuple
import PyPDF2
import markdown
import docx2txt
from src.chatbot import DocumentQASystem
import yaml
import argparse

def init_model(config_path):
    global qa_system
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    qa_system = DocumentQASystem(config)

def handle_citation_click(citation: str, retrieved_docs: List[Dict[str, Any]])-> str:
    citation = int(citation)
    return f"<pre style='white-space: pre-wrap;'>{retrieved_docs[int(citation)-1].page_content}</pre>"

def upload_file(prev_doc_texts, prev_files, doc_files: List) -> Tuple[Any, Any, str, Dict, List]:
    doc_texts = prev_doc_texts
    for file in doc_files:
        if file in prev_files:
            continue
        if file.name.endswith(".txt"):
            with open(file, "r", encoding="utf-8") as f:
                doc_texts[os.path.basename(file.name)] = f.read()
        elif file.name.endswith(".pdf"):
            with open(file, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                doc_texts[os.path.basename(file.name)] = text
        elif file.name.endswith(".docx"):
            text = docx2txt.process(file)
            doc_texts[os.path.basename(file.name)] = text
        else:
            doc_texts[os.path.basename(file.name)] = "Unsupported file type."

    doc_choices = list(doc_texts.keys())
    return (
        gr.update(choices=doc_choices, value=doc_choices[-1]),
        gr.update(placeholder="Files uploaded. Click 'Finish' to process documents."),
        doc_texts,
        doc_files,
        gr.update(interactive=True)
    )
    

def finish_click(file_paths: List[str], chunking_method: str) -> Tuple[str, Dict]:
    qa_system.build_vector_database(file_paths, chunking_method)
    return gr.update(placeholder="Ask a question about the uploaded documents."), gr.update(interactive=True)

def doc_select(selected_doc: str, doc_text_state: Dict) -> str:
    if not selected_doc or not doc_text_state:
        return "<p>No documents uploaded.</p>"
    
    content = doc_text_state.get(selected_doc, "")
    html_content = f"<pre style='white-space: pre-wrap;'>{content}</pre>"
    return html_content

def html2str(html_response: str) -> str:
    footnote_start = html_response.find("<hr>")
    if footnote_start != -1:
        html_response = html_response[:footnote_start]
    
    plain_text = re.sub(r'<[^>]+>', '', html_response)
    
    return plain_text.strip()

def chat(message: str, history: List[Dict[str, Any]], retrieve_num: int) -> Tuple[str, List[Dict[str, str]]]:
    if not message.strip():
        return "Please enter a question.", []
    
    new_history = []
    for m in history:
        new_history.append({"role": m["role"], "content": html2str(m["content"])})

    response = qa_system.answer_question(message, new_history, retrieve_num)

    html_response = response["prediction"]
    retrieved_docs = response["documents"]
    citation_map = {str(i + 1): i for i in range(len(retrieved_docs))}
    
    cited_indices = set()
    for citation in citation_map:
        if f"[{citation}]" in html_response:
            cited_indices.add(citation)

    footnotes = "<hr><h3>References</h3><ol>" if cited_indices else ""
    for citation in sorted(cited_indices, key=int):  
        doc_idx = citation_map[citation]
        if doc_idx < len(retrieved_docs):
            html_response = html_response.replace(
                f"[{citation}]",
                f'<a href="#" class="citation-link" style="color: blue; text-decoration: underline;">[{citation}]</a>'
            )
            footnotes += f'<li id="cite-{citation}">{retrieved_docs[doc_idx].metadata["title"]}</li>'
    footnotes += "</ol>" if cited_indices else ""

    if cited_indices:
        html_response += footnotes

    return html_response, retrieved_docs

def clear_docs() -> Tuple[Any, List, str]:
    return gr.update(choices=[]), [], "", "<p>No documents uploaded.</p>"

def clear_attr() -> str:
    return "<p>click citation to check</p>"

def main():

    with open("html/head.html", encoding="utf-8") as html_file:
        head = html_file.read().strip()

    with open("html/styles.css", encoding="utf-8") as css_file:
        css = css_file.read().strip()

    with gr.Blocks(title="Document QA System", head=head, css=css, fill_height=True, analytics_enabled=False) as demo:
        doc_text_state = gr.State(value={})
        file_paths = gr.State(value=[])
        retrieved_docs = gr.State(value=[])

        citation_input = gr.Textbox(visible=False, elem_id="citation_input", interactive=True)
        citation_button = gr.Button(visible=False, elem_id="citation_button", interactive=True)

        with gr.Row(equal_height=False):
            with gr.Column(min_width=700):
                doc_files_box = gr.File(
                    label="Upload Documents",
                    file_types=[".txt", ".docx", ".pdf"],
                    file_count="multiple",
                    height=60
                )
                finish_button = gr.Button("Finish", variant="secondary", interactive=False)
                with gr.Accordion(label="Preprocessing Parameters", open=False):
                    chunking_method = gr.Radio(["semantic", "recursive"], value="semantic", label="Chunking Method", render=True)

                with gr.Tabs(visible=True) as tabs:
                    with gr.Tab("Document Content", id="doc-content"):
                        doc_selection_dropdown = gr.Dropdown(label=None, show_label=False, container=False, interactive=True, scale=1)
                        doc_display_html = gr.HTML("<p>No documents uploaded.</p>", label="Document Content", elem_classes="html-text")
                    with gr.Tab("Attribution Chunks", elem_classes="tab", id="tab-attr-chunk") as attr_chunks_tab:
                        attribution_chunks_html = gr.HTML("<p>click citation to check</p>", label="Attribution Chunks", elem_classes="html-text")

            with gr.Column(min_width=750):

                chat_interface = gr.ChatInterface(
                    fn=chat,
                    title = "🤖 Document Question Answering System",
                    chatbot=gr.Chatbot(label="Chatbot", show_copy_button=False, type="messages", elem_id="chatbot"),
                    textbox=gr.Textbox(
                        placeholder="Please upload files first.",
                        interactive=False,
                        max_lines=4
                    ),
                    additional_inputs=[
                        gr.components.Slider(minimum=1, maximum=10, step=1, value=4, label="retrieve num", render=False)
                    ],
                    additional_outputs=[retrieved_docs]
                )

        citation_button.click(
            handle_citation_click,
            inputs=[citation_input, retrieved_docs],
            outputs=[attribution_chunks_html]
        )

        doc_files_box.upload(
            upload_file,
            [doc_text_state, file_paths, doc_files_box],
            [doc_selection_dropdown, chat_interface.textbox, doc_text_state, file_paths, finish_button]
        )

        doc_files_box.clear(
            clear_docs, 
            outputs=[doc_selection_dropdown, file_paths, doc_text_state, doc_display_html]
        )
        doc_files_box.clear(clear_docs, outputs=[doc_selection_dropdown, file_paths, doc_text_state, doc_display_html, finish_button])
        chat_interface.chatbot.undo(clear_attr, outputs=[attribution_chunks_html])
        chat_interface.chatbot.retry(clear_attr, outputs=[attribution_chunks_html])

        finish_button.click(
            finish_click,
            [file_paths, chunking_method],
            [chat_interface.textbox, chat_interface.textbox]
        )

        doc_selection_dropdown.change(
            doc_select,
            [doc_selection_dropdown, doc_text_state],
            doc_display_html
        )


    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="yamls/chatbot.yaml",
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    init_model(args.config)
    main()