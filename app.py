from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

text_splitter = CharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separator="\n"
    )


#Loader
loader = PyPDFLoader("docs/carry.pdf")
new_documents = loader.load_and_split()

texts = text_splitter.split_documents(new_documents)

# 임베딩 모델 로드
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask", encode_kwargs={'normalize_embeddings': True}
)

# Chroma DB에 기사 벡터화하여 저장하기
from langchain.vectorstores import Chroma
docsearch = Chroma.from_documents(texts, embeddings, persist_directory="./news_chroma_db")

#Chroma VectorDB 에 질의하기
# r = docsearch.similarity_search("캐리프로토콜이란")
# print(r)




# llm = LlamaCpp(
# 	# model_path: 로컬머신에 다운로드 받은 모델의 위치
#     model_path="models/Llama-2-ko-7B-chat-gguf-q4_0.bin",
#     temperature=0.75,
#     top_p=0.95,
#     max_tokens=8192,
#     verbose=True,
#     # n_ctx: 모델이 한 번에 처리할 수 있는 최대 컨텍스트 길이
#     n_ctx=8192,
#     # n_gpu_layers: 실리콘 맥에서는 1이면 충분하다고 한다
#     n_gpu_layers=1,
#     n_batch=512,
#     f16_kv=True,
#     n_threads=16,
# )

llm = LlamaCpp(
    model_path="models/Llama-2-ko-7B-chat-gguf-q4_0.bin",
    n_ctx=1024,
    n_batch=1024,
    n_gpu_layers=256 , #gpu 가속을 원하는 경우 주석을 해제하고 Metal(Apple M1) 은 1, Cuda(Nvidia) 는 Video RAM Size 를 고려하여 적정한 수치를 입력합니다.
    verbose=True,
)



# 유사도에 맞추어 대상이 되는 텍스트를 임베딩함
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.5
)
# 압축 검색기 생성
compression_retriever = ContextualCompressionRetriever(
    # embeddings_filter 설정
    base_compressor=embeddings_filter,
    # retriever 를 호출하여 검색쿼리와 유사한 텍스트를 찾음
    base_retriever=docsearch.as_retriever()
)
# context 를 참조하여 순서에 맞게 명료하게 응답하게 구성

prompt_template = """
Give an answer by referring to the context, and include the address within the context in the answer, and clearly number the answer.

{context}

Question: {question}
Answer in Korea:"""

# RetrievalQA 클래스의 from_chain_type이라는 클래스 메서드를 호출하여 질의응답 객체를 생성
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
    verbose=True,

)


res = qa({"query":"캐리프로토콜에 대해 알려줘?"})
print(res)