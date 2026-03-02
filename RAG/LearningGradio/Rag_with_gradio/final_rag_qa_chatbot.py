from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from typing import List
import gradio as gr

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')


## Safe Embedding Wrapper - embeds one chunk at a time to avoid index mismatch
class SafeWatsonxEmbeddings(Embeddings):
    def __init__(self):
        self.model = WatsonxEmbeddings(
            model_id='ibm/slate-125m-english-rtrvr',
            url="https://us-south.ml.cloud.ibm.com",
            project_id="skills-network",
            params={EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            text = text.strip()
            if not text:
                continue
            try:
                result = self.model.embed_documents([text])
                if result and len(result) > 0:
                    embeddings.append(result[0])
            except Exception as e:
                print(f"Skipping chunk due to embedding error: {e}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.model.embed_documents([text.strip()])
        return result[0]


## LLM
def get_llm():
    parameters = {
        GenParams.MAX_NEW_TOKENS: 256,
        GenParams.TEMPERATURE: 0.5,
    }
    return WatsonxLLM(
        model_id='ibm/granite-3-2-8b-instruct',
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=parameters,
    )


## Document loader
def document_loader(file):
    loader = PyPDFLoader(file)
    return loader.load()


## Text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(data)


## Vector database
def vector_database(chunks):
    # Filter empty chunks
    valid_chunks = [c for c in chunks if c.page_content.strip()]
    if not valid_chunks:
        raise ValueError("No valid text chunks found in the document.")

    embedding_model = SafeWatsonxEmbeddings()
    return Chroma.from_documents(valid_chunks, embedding_model)


## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


## QA Chain
def retriever_qa(file, query):
    if file is None:
        return "Please upload a PDF file before asking a question."
    if not query or query.strip() == "":
        return "Please enter a question."

    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )
    response = qa.invoke(query)
    return response['result']


# Gradio Interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="Karanpreet Chatbot",
    description="Upload a PDF document first, then type your question and click Submit.",
    submit_btn="Submit",
    clear_btn="Clear",
)

rag_application.launch(server_name="127.0.0.1", server_port=7860)



# from ibm_watsonx_ai.foundation_models import ModelInference
# from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
# from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
# from ibm_watsonx_ai import Credentials
# from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.chains import RetrievalQA
# from langchain.embeddings.base import Embeddings
# from typing import List
# import gradio as gr

# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn
# warnings.filterwarnings('ignore')


# ## Safe Embedding Wrapper - embeds one chunk at a time to avoid index mismatch
# class SafeWatsonxEmbeddings(Embeddings):
#     def __init__(self):
#         self.model = WatsonxEmbeddings(
#             model_id='ibm/slate-125m-english-rtrvr',
#             url="https://us-south.ml.cloud.ibm.com",
#             project_id="skills-network",
#             params={EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 512},
#         )

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         embeddings = []
#         for text in texts:
#             text = text.strip()
#             if not text:
#                 continue
#             try:
#                 result = self.model.embed_documents([text])
#                 if result and len(result) > 0:
#                     embeddings.append(result[0])
#             except Exception as e:
#                 print(f"Skipping chunk due to embedding error: {e}")
#         return embeddings

#     def embed_query(self, text: str) -> List[float]:
#         result = self.model.embed_documents([text.strip()])
#         return result[0]


# ## LLM
# def get_llm():
#     parameters = {
#         GenParams.MAX_NEW_TOKENS: 256,
#         GenParams.TEMPERATURE: 0.5,
#     }
#     return WatsonxLLM(
#         model_id='ibm/granite-3-2-8b-instruct',
#         url="https://us-south.ml.cloud.ibm.com",
#         project_id="skills-network",
#         params=parameters,
#     )


# ## Document loader
# def document_loader(file):
#     loader = PyPDFLoader(file)
#     return loader.load()


# ## Text splitter
# def text_splitter(data):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200,
#         length_function=len,
#     )
#     return splitter.split_documents(data)


# ## Vector database
# def vector_database(chunks):
#     # Filter empty chunks
#     valid_chunks = [c for c in chunks if c.page_content.strip()]
#     if not valid_chunks:
#         raise ValueError("No valid text chunks found in the document.")

#     embedding_model = SafeWatsonxEmbeddings()
#     return Chroma.from_documents(valid_chunks, embedding_model)


# ## Retriever
# def retriever(file):
#     splits = document_loader(file)
#     chunks = text_splitter(splits)
#     vectordb = vector_database(chunks)
#     return vectordb.as_retriever()


# ## QA Chain
# def retriever_qa(file, query):
#     if file is None:
#         return "Please upload a PDF file before asking a question."
#     if not query or query.strip() == "":
#         return "Please enter a question."

#     llm = get_llm()
#     retriever_obj = retriever(file)
#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever_obj,
#         return_source_documents=False,
#     )
#     response = qa.invoke(query)
#     return response['result']


# # Gradio Interface
# rag_application = gr.Interface(
#     fn=retriever_qa,
#     allow_flagging="never",
#     inputs=[
#         gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
#         gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
#     ],
#     outputs=gr.Textbox(label="Output"),
#     title="Karanpreet Chatbot",
#     description="Upload a PDF document first, then type your question and click Submit.",
#     submit_btn="Submit",
#     clear_btn="Clear",
# )

# rag_application.launch(server_name="127.0.0.1", server_port=7860)