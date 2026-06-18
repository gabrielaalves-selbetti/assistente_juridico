import os 
import tempfile
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys
sys.stdout.reconfigure(encoding='utf-8')

st.set_page_config(page_title='Assistente Jurídico', page_icon=':100:', layout='wide')

with st.sidebar:
    st.header('Configurações')
    api_key = st.text_input('Insira a chave da sua API da Groq', type='password')

st.title('Assistente Jurídico')

if not api_key:
    st.warning('Informe a API Key da Groq para continuar.')
    st.stop()

os.environ['GROQ_API_KEY'] = api_key
os.environ['TOKENNIZERS_PARALLELISM'] = 'false'

llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.2, max_tokens=1024)
pdf_file = st.file_uploader('Envie um PDF da área jurídica', type=['pdf'])

if 'chroma_dir' not in st.session_state:
    st.session_state.chroma_dir = tempfile.mkdtemp(prefix='chroma_rag_')

def cria_banco_vetorial(pdf_bytes) -> Chroma:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    
    docs = PyPDFLoader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/msmarco-bert-base-dot-v5')
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory = st.session_state.chroma_dir)

    return vectordb

if pdf_file and 'vectordb_ready' not in st.session_state:
    with st.spinner('Indexando o PDF no ChromaDB...'):
        st.session_state.vectordb = cria_banco_vetorial(pdf_file.read())
        st.session_state.vectordb_ready = True
        st.success('Indexação concluída.')

retriever = None

if st.session_state.get('vectordb_ready'):
    retriever = st.session_state.vectordb.as_retriever(search_kwargs = {'k': 3})

system_block = """Você é um assistente jurídico que responde usando estritamente o conteúdo do PDF fornecido quando possível. Se a resposta não estiver no PDF, diga que não encontrou no documento e ofereça passos de verificação. Formate a resposta com: Resumo, Fundamentação (com citações de trechos entre aspas) e Próximos passos. Se houver conflito entre o PDF e conhecimento externo, priorize o PDF e sinalize a divergência."""

qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_block),
    ('human', 'Contexto do PDF:\n{context}\n\nPergunta: {pergunta}\nResponda de forma sucinta, técnica e didática')
])

def formata_docs(docs):
    out = []

    for d in docs:
        meta = d.metadata or {}
        where = f'p.{meta.get('page', '?')}'
        out.append(f'[{where}] "{d.page_content.strip()[:800]}{'...' if len(d.page_content) > 800 else ""}"')

    return "\n\n".join(out)

pergunta = st.text_area('Escreva sua pergunta jurídica', height=120, placeholder='Ex.: Quais cláusula tratam de rescisão e multas?')

col1, col2 = st.columns([1,1])
with col1:
    btn = st.button('Perguntar')

if btn: 
    if not retriever:
        st.error('Envie um pdf para habilitar.')
        st.stop()
    
    rag_pipeline = RunnableParallel(
        context = retriever | formata_docs,
        pergunta = RunnablePassthrough()
    ) | qa_prompt | llm | StrOutputParser

    with st.spinner('Gerando a resposta...'):
        answer = rag_pipeline.invoke(pergunta)

    st.markdown('### Resposta')
    st.write(answer)