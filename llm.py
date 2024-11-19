from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
#from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_upstage import UpstageEmbeddings

from config import answer_examples
from langchain_upstage import ChatUpstage
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from langchain_openai import ChatOpenAI
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    #embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    embedding = UpstageEmbeddings(model="solar-embedding-1-large")
    index_name = 'data2'
    #index_name = "graduate"
    
    '''
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    )

    loader = Docx2txtLoader("D:/llm/data.docx")
    document_list = loader.load_and_split(text_splitter=text_splitter)    
   '''
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    #database = PineconeVectorStore.from_documents(document_list,embedding=embedding,index_name=index_name)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever

def get_llm(model='gpt-4o'):
    #llm = ChatOpenAI(model=model)    
    llm  = ChatUpstage(temperature=0.3)
    return llm

def get_history_retriever(llm):    
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_dictionary_chain(llm):
    dictionary = ["사람을 나타내는 표현 -> 수험생", "알려주세요 -> 무엇인가요?"]
    
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain


def get_rag_chain(llm):    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "[AI] {answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    system_prompt = (
        "당신은 입학 전문가입니다. 사용자의 입학에 관한 질문에 답변해주세요"
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"       
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
             few_shot_prompt,            
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = get_history_retriever(llm)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    
    return conversational_rag_chain


def get_ai_response(user_message):
    llm = get_llm()
    dictionary_chain = get_dictionary_chain(llm)
    rag_chain = get_rag_chain(llm)
    tax_chain = {"input": dictionary_chain} | rag_chain 
    ai_response = tax_chain.stream(
        {
            "question": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_wikipedia():
    retriever = WikipediaRetriever()
    llm = ChatOpenAI(model="gpt-4o-mini")  

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the context provided.
        Context: {context}
        Question: {question}
        """
    )

    chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )  

    return chain


