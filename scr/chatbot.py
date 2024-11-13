import streamlit as st
from streamlit_chat import message
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["OPENAI_API_KEY"] = "YOUR API KEY"

# 법률 정보를 위한 벡터 DB 로드 함수
def load_vector_db():
    df = pd.read_csv("law_df.csv")  # 법률 정보 CSV 파일
    documents = [Document(page_content=f"질문: {row['Text']}\n답변: {row['Completion']}", metadata={"source": f"row_{i}"}) for i, row in df.iterrows()]
    
    model_name = "sentence-transformers/LaBSE"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    vectors = FAISS.from_documents(documents, embeddings)
    return vectors

# 벡터 DB 로드
vector_db = load_vector_db()

# 심리 상담 모델 로드
tokenizer = AutoTokenizer.from_pretrained("Soeon/bitaminnlp1")
model = AutoModelForCausalLM.from_pretrained("Soeon/bitaminnlp1")

# 문장 임베딩 모델 로드
sentence_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

st.title('법률 정보 제공 및 심리 상담 챗봇😎')

# 사용자 정보 입력
user_name = st.text_input("이름을 입력하세요:")
user_age = st.number_input("나이를 입력하세요:", min_value=0, step=1)
user_gender = st.selectbox("성별을 선택하세요:", ("남자", "여자", "선택 안 함"))

if user_name and user_age and user_gender:
    st.write(f"안녕하세요, {user_name}님! ({user_age}세, {user_gender})")

    prompt_template = """
    당신은 법률 전문가입니다. 법률에 관한 질문에 해당 **{context}**를 철저히 검토하고,
    정확하고 명확한 법률적 조언을 제공해주세요. 특히, 법적 용어와 개념을 이해기 쉽도록 자세히 설명하고,
    질문에 직접적으로 답변하는 것을 우선시해주세요. 
    상대방의 연령과 상황에 맞는 말투와 설명 방식을 사용해 자연스럽고 인간적인 대화를 이어가세요.

    {system_prompt}

    question: {question}

    Assistant: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["system_prompt", "question", "context"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
        retriever=vector_db.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    def generate_prompt(age):
        if age <= 19:
            return "청소년을 대상으로 하는 친근하고 쉬운 말투로 대화해주세요."
        else:
            return "성인을 대상으로 하는 정중하고 명확한 말투로 대화해주세요."

    def legal_chat(query):
        system_prompt = generate_prompt(user_age)
        result = chain(
            {
                "question": query, 
                "chat_history": st.session_state['history'],
                "system_prompt": system_prompt
            }
        )
        return result['answer']

    def psychological_chat(query):
        input_ids = tokenizer.encode(query, return_tensors="pt")
        output = model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    def get_embedding(text):
        return sentence_model.encode(text)

    def ensemble_chat(query):
        rag_response = legal_chat(query)
        ft_response = psychological_chat(query)
        
        query_embedding = get_embedding(query)
        rag_embedding = get_embedding(rag_response)
        ft_embedding = get_embedding(ft_response)
        
        rag_similarity = cosine_similarity([query_embedding], [rag_embedding])[0][0]
        ft_similarity = cosine_similarity([query_embedding], [ft_embedding])[0][0]
        
        if rag_similarity > ft_similarity:
            return rag_response
        else:
            return ft_response

    def conversational_chat(query):
        return ensemble_chat(query)

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["안녕하세요! 법률 및 심리 상담에 대해 질문해 주세요."]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["안녕하세요!"]
        
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='Conv_Question', clear_on_submit=True):           
            user_input = st.text_input("Query:", placeholder="법률이나 심리 상담에 대해 물어보세요.", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji", seed="Nala")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Fluffy")

else:
    st.write("사용자 정보를 모두 입력해 주세요.")
