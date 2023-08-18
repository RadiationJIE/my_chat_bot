my_api_key = "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"

# Document Loading
# from langchain.document_loaders import Docx2txtLoader

# loader = Docx2txtLoader("docs/期货交易所管理办法.docx")
# data = loader.load()

# Splitting
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
# all_splits = text_splitter.split_documents(data)

# Storage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(openai_api_key=my_api_key), persist_directory="./chroma_db")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(openai_api_key=my_api_key))

# Retrieval
# question = "什么是需求统筹方？"
# docs = vectorstore.similarity_search(question)

# Generation
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate

# template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer. 
# Use three sentences maximum and keep the answer as concise as possible. 
# Remember always say Chinese. 
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key = my_api_key)
# qa_chain = RetrievalQA.from_chain_type(
#     llm,
#     retriever=vectorstore.as_retriever(),
#     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
# )
# result = qa_chain({"query": question})
# print(result["result"])

# Conversation
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key = my_api_key)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)
# result = chat({"question": "本办法的适用范围是？"})
# print(result["answer"])
# result = chat({"question": "能详细说说吗？"})
# print(result["answer"])

# ChatBot
import streamlit as st
from streamlit_chat import message

with st.sidebar:
    radio = st.radio(
        "菜单",
        ("聊天机器人", "查看原文档")
    )

if radio != "聊天机器人":
    with open('docs/期货交易所管理办法.txt', 'r') as f: 
        body = f.read()
    st.markdown(body, unsafe_allow_html=True)
else:
    if 'questions' not in st.session_state:
        st.session_state['questions'] = []

    if 'answers' not in st.session_state:
        st.session_state['answers'] = ['您可以基于《期货交易所管理办法》文档向我提问哦！']

    question = st.chat_input("输入您的问题")
    if question:
        answer = chat({"question": question})["answer"]
        st.session_state['questions'].append(question)
        st.session_state['answers'].append(answer)

    if st.session_state['answers']:
        if len(st.session_state['answers']) == 1:
            st.chat_message("assistant").write(st.session_state['answers'][0])
        else:
            for i in range(0, len(st.session_state['answers'])):
                st.chat_message("assistant").write(st.session_state['answers'][i])
                if i != len(st.session_state['answers'])-1:
                    st.chat_message("user").write(st.session_state['questions'][i])

