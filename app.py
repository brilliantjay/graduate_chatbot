import streamlit as st
from persist import persist, load_widget_state
from dotenv import load_dotenv
from llm import *
import pandas as pd
import time

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()   
st.set_page_config(page_title="대학원 입학 정보 챗봇")


st.title("대학원 입학 정보 챗봇")
st.caption("대학원 입학에 관련된 서류상의 내용을 검색해 답해드립니다!")   
#질문시 무엇인가요?로 끝나야 한다. 

chain = get_wikipedia()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="대학원 입학에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        ai_response2 = chain.invoke(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)            
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
            
        with st.chat_message("ai2"):                
            st.write("[위키피디아] "+ai_response2)               
            st.session_state.message_list.append({"role": "ai2", "content": ai_response2})



