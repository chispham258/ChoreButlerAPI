# Library
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import chromadb
import os
import numpy

# Path
path = 'Data'
vectorDB = './chromadb'

def CreateChromaDB():
    # Load data
    loader = DirectoryLoader(path, glob = '*.pdf', loader_cls = PyPDFLoader)
    pdf = loader.load()

    # Chunk data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ".", " ", ""],
        chunk_size = 512,
        chunk_overlap = 64,
        length_function = len
    )
    chunks = text_splitter.split_documents(pdf)

    # Extract text from chunk
    chunks_text = [chunk.page_content for chunk in chunks]

    # Embedding using bkai-foundation-models/vietnamese-bi-encoder for vietnamese
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    embeddings = model.encode(chunks_text)

    # Tạo ChromaDB
    chroma_client = chromadb.PersistentClient(path = vectorDB)
    collection = chroma_client.get_or_create_collection(name = 'chroma_vietnamese')

    # Save data into database
    for i, doc in enumerate(chunks_text):
        collection.add(
            ids = [str(uuid.uuid4())],
            documents = [doc],
            embeddings = [embeddings[i].tolist()]
        )

    return collection

import langchain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import speech_recognition as sr
from gtts import gTTS
from pygame import mixer
import os
import time


REJECTION_PATTERNS = [
    "tớ xin lỗi",
    "tớ chưa có thông tin",
    "tớ không chắc",
    "hãy hỏi câu khác",
    "không tìm thấy thông tin",
]

def is_rejection(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(pattern in answer_lower for pattern in REJECTION_PATTERNS)

api_key = "AIzaSyBXcwfz4KAHwHImehIltmcryrKrhlS5gfE"

# Link database

from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
import fastapi

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading resources...")
    # collection = CreateChromaDB()

    embeddings = HuggingFaceEmbeddings(model_name='bkai-foundation-models/vietnamese-bi-encoder')
    vector_store = Chroma(
        collection_name = 'chroma_vietnamese',
        persist_directory = './chromadb',
        embedding_function = embeddings
    )

    # Create Retrieval for LLM
    retrieval = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k" : 2
        }
    )

    # Create ChatPromptTemplate
    system_message = """Xin chào! Bạn là một trợ lý ảo thân thiện và linh hoạt, chuyên hướng dẫn trẻ em thực hiện các công việc nhà.
    Nhiệm vụ của bạn:
    - Không trả lời bất cứ câu hỏi nào mà bạn không chắc chắn.
    - Trả lời các câu hỏi của trẻ dựa trên thông tin từ cơ sở dữ liệu có sẵn bằng tiếng việt.
    - Sử dụng ngôn ngữ vui vẻ, gần gũi, ngắn gọn và dễ hiểu để phù hợp với trẻ nhỏ.
    - Khuyến khích và khen ngợi trẻ khi chúng đặt câu hỏi hoặc học điều mới.
    - Nếu không tìm thấy câu trả lời trong dữ liệu, hãy xin lỗi một cách nhẹ nhàng và đề nghị trẻ đặt câu hỏi khác. Ví dụ: 
    "Ôi, tớ xin lỗi nhé! Tớ chưa có thông tin về điều này. Cậu có thể hỏi một câu khác được không?
    Nhớ rằng, bạn không được sử dụng từ ngữ khiếm nhã hoặc thông tin ngoài cơ sở dữ liệu. Hãy luôn là một người bạn tốt và đáng tin cậy của trẻ em nhé!"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("user", "Ngữ cảnh: {context}\n\nCâu hỏi: {question}")
        ]
    )

    # Create model llm by API gemini
    llm = ChatGoogleGenerativeAI(
        model = "models/gemini-2.0-flash",
        temperature = 0.3,
        max_tokens = 1024,
        max_retries = 2,
        top_p = 0.9,
        google_api_key = api_key
    )

    # Create memory to maintain conversation
    memory = ConversationBufferWindowMemory(
        output_key = "answer",           # Khóa lưu câu trả lời
        memory_key = "chat_history",  # Khóa lưu lịch sử trò chuyện
        return_messages = True,       # Trả về dạng danh sách tin nhắn
        k = 3                         # Chỉ nhớ 3 lượt truy cập gần nhất
    )

    # Create chain for conversation
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = retrieval,
        memory = memory,
        combine_docs_chain_kwargs = {
            "prompt": qa_prompt
        }
    )

    print("Resources loaded!")

    app.state.qa_chain = qa_chain
    app.state.embeddings = embeddings
    app.state.vector_store = vector_store
    app.state.self_update = qa_prompt | llm

    yield 
    
    print("Closing resources...")

app = FastAPI(lifespan = lifespan)

class Question(BaseModel):
    query: str


@app.post("/ask")
def generate(request : Request, prompt: Question):
    # Truy vấn LLM
    question = prompt.query

    qa_chain = request.app.state.qa_chain
    response = qa_chain.invoke({"question": question})
    answer = response["answer"]

    if not is_rejection(str(answer)):
        return {
            "answer": answer,
        }

    return {
        "answer" : answer
    }

    self_update = request.app.state.self_update
    self_response = self_update.invoke({
        "question": question, 
        "context" : """
        Đưa ra từng bước nhỏ chi tiết ngắn gọn để trả lời câu hỏi.
        Trong đó phải có các bước chuẩn bị, thực hiện và những lưu ý cho các bạn nhỏ. 
        """
    })

    answer = self_response.content
    if not is_rejection(str(answer)):
        new_doc = f" Câu hỏi : {question} Trả lời : {answer}"

        vector_store = request.app.state.vector_store
        embeddings = request.app.state.embeddings

        embedding = embeddings.embed_documents([new_doc])[0]
        vector_store._collection.add(
            documents = [new_doc],
            ids = [str(uuid.uuid4())],
            embeddings = [embedding]
        )
    
    response["answer"] = answer

    return {
        "answer" : answer
    }


