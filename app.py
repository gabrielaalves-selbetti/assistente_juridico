import os 
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

st.set_page_config(page_title='Assistente Jurídico', page_icon=':100:', layout='wide')

with st.sidebar:
    st.header('Configurações')
    api_key = st.text_input('Insira a chave da sua API da Groq', type='password')

st.title('Assistente Jurídico')

if not api_key:
    st.warning('Informe a API Key da Groq para continuar.')
    st.stop()

os.environ['GROQ_API_KEY'] = api_key

llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.2, max_tokens=1024)

system_block = """Você é um assistente jurídico que escreve de forma objetiva e clara, sem dar aconselhamento legal definitivo.
Forneça análise, contexto e referências genéricas (leis, súmulas, doutrina) sem afirmar certeza absoluta.
Se algo for incerto, explique como um profissional verificaria a informação (consultar legislação atualizada, jurisprudência local, etc.).
Estruture a resposta com: Contexto breve, Pontos principais, Riscos/limitações, Próximos passos práticos."""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_block), 
    MessagesPlaceholder(variable_name='history'),
    ('human', 'Pergunta: {pergunta}\nResponda de forma sucinta, técnica e didática')]
)

if 'history' not in st.session_state:
    st.session_state.history = []

with st.form('form'):
    pergunta = st.text_area('Pergunta', height=120, placeholder='Digite a sua dúvida sobre jurídica')
    enviado = st.form_submit_button('Enviar')

if enviado: 
    msgs = prompt.invoke({'history': st.session_state.history, 'pergunta': pergunta})
    resp = llm.invoke(msgs.messages)

    st.session_state.history.extend(
        [
            HumanMessage(content=f'Pergunta: {pergunta}'), 
            AIMessage(content=resp.content if hasattr(resp, "content") else str(resp))
        ]
    )

    st.markdown('### Resposta')
    st.write(resp.content)