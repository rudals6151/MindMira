import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import pandas as pd
import os

# 환경 변수 설정
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Streamlit 제목
st.title('Transformer 기반 RAG 챗봇')

# 사용자 정보 입력
user_name = st.text_input("이름을 입력하세요:")
user_age = st.number_input("나이를 입력하세요:", min_value=0, step=1)
user_gender = st.selectbox("성별을 선택하세요:", ("남자", "여자", "선택 안 함"))

if user_name and user_age and user_gender:
    st.write(f"안녕하세요, {user_name}님! ({user_age}세, {user_gender})")

    # 법률 정보 벡터 DB 로드 함수
    def load_vector_db():
        df = pd.read_csv("law_data.csv")  # 법률 정보 데이터 불러오기
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

    # 트랜스포머 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/rag-token-nq")

    # 기본 프롬프트 템플릿
    prompt_template = """
    당신은 법률 전문가이자 심리 상담 전문가입니다. 법률에 관한 질문에 해당 **{context}**를 철저히 검토하고,
    정확하고 명확한 법률적 조언을 제공해주세요. 심리 상담에 대해서는 공감적이고 이해하기 쉬운 방식으로 대화해주세요.
    특히, 법적 용어와 개념을 이해하기 쉽도록 자세히 설명하고, 질문에 직접적으로 답변하는게 중요합니다.
    상대방의 연령과 상황에 맞는 말투와 설명 방식을 사용해 자연스럽고 인간적인 대화를 이어가세요.

    question: {question}

    Assistant: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question"]
    )

    # 대화형 체인 설정
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vector_db.as_retriever(),
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

    # 질문 처리 함수 정의
    def chat(query):
        inputs = tokenizer(query, return_tensors="pt")
        outputs = model.generate(**inputs)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result

    # 대화 기록 초기화
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # 대화 컨테이너
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='Chat_Question', clear_on_submit=True):
            user_input = st.text_input("질문을 입력하세요:", placeholder="법률이나 심리 상담에 대해 물어보세요.", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = chat(user_input)

            # 대화 기록 업데이트
            st.session_state['history'].append({"user": user_input, "assistant": output})

    if st.session_state['history']:
        with response_container:
            for entry in st.session_state['history']:
                st.write(f"사용자: {entry['user']}")
                st.write(f"챗봇: {entry['assistant']}")
else:
    st.write("사용자 정보를 모두 입력해 주세요.")
