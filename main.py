import os
import pandas as pd
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_openai import OpenAIEmbeddings
from kiwipiepy.utils import Stopwords
from kiwipiepy import Kiwi
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages.chat import ChatMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from Utils.utils import retrieve_text, kiwi_tokenize
from Prompt.prompt import user_History_prompt_new, prompt_new
from operator import itemgetter
import streamlit as st
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# env 파일에서 OPENAI API KEY 들여옴
load_dotenv()

# LangChain 추적 시작
logging.langsmith("1112_Test_BuyerAgent")

# kiwi 지정
kiwi = Kiwi(typos="basic", model_type="sbg")
stopwords = Stopwords()
stopwords.remove(("사람", "NNG"))

# Embedding
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", api_key=os.environ["OPENAI_API_KEY"]
)

# LLM 설정
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

# 판매자 데이터
df = pd.read_excel("data/경희대학교_음식데이터_1108.xlsx")

st.set_page_config(page_title="Buyer Agent", page_icon="🍽️", layout="wide")
st.title("Buyer Agent")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []
    # 도움말 메시지 추가
    help_message = """안녕하세요! Buyer Agent입니다. 경희대학교 주변 맛집 정보, 제가 다 모아뒀어요. 당신의 취향과 상황에 딱 맞는 추천을 드릴게요. 먹고 싶은 음식이나 원하는 조건을 편하게 말해 주세요.\n
예를 들어 이런 질문이 가능해요.\n1. 요즘 다이어트 중인데, 칼로리가 낮은 메뉴 뭐가 있을까요?\n2. 헬스를 하고 있어서 고단백 음식을 먹어야합니다.\n3. 밥을 먹어야하는데 돈이 부족합니다. 양 많고 저렴한 음식을 찾고 있습니다."""
    st.session_state["messages"].append(
        ChatMessage(role="assistant", content=help_message)
    )

# Chain 저장용
if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 대화 내용을 기억하기 위한 저장소 생성
if "store" not in st.session_state:
    st.session_state["store"] = {}

# 사이드바 생성
with st.sidebar:
    st.header("옵션💡")
    # 초기화 버튼 생성
    clear_btn = st.button("대화 다시 시작")
    st.header("")

    # 사용자 음식 구매 관련 특이사항 데이터
    st.header("음식 구매 관련 의사결정💡")
    food_category = st.text_input(label="선호하는 음식 카테고리", placeholder="")
    taste_performance = st.text_input(label="선호하는 맛", placeholder="")
    price = st.text_input(label="구매 금액대", placeholder="")
    dietary_restrictions = st.text_input(label="특이사항", placeholder="")

    user_btn = st.button("입력")

    # 전역 변수로 text_data 선언
    text_data = ""

    # 사용자 음식 구매 관련 버튼을 누를시...
    if user_btn:
        text_data += f"""
    "선호하는 음식 카테고리": {food_category},
    "선호하는 맛": {taste_performance},
    "구매 금액대": {price},
    "식이제한 및 알레르기": {dietary_restrictions}"""
        st.success("입력 완료")


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        role = "👤 사용자" if chat_message.role == "user" else "🧑‍💼"
        bg_color = "#F0F0F0" if chat_message.role == "user" else "#E0FFE0"

        with st.container():
            st.markdown(
                f"<div style='display: flex; align-items: center; background-color: {bg_color}; padding: 10px; border-radius: 10px; margin-bottom: 10px;'>"
                f"<div><strong>{role}:</strong> {chat_message.content}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환


# Chain 생성
def create_chain():
    # Prompt 생성
    prompt = prompt_new

    # LLM 생성
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

    # retriever 생성
    retriever = retrieve_text()

    # Chain 생성
    chain = (
        {
            "question": itemgetter("question"),
            "user_history": lambda _: text_data,
            "context": lambda _: itemgetter("question") | retriever,
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지 키
    )

    return rag_with_history


# 초기화 버튼일 눌리면..
if clear_btn:
    st.session_state.clear()
    # 도움말 메시지 추가
    help_message = """안녕하세요! Buyer Agent입니다. 경희대학교 주변 맛집 정보, 제가 다 모아뒀어요. 당신의 취향과 상황에 딱 맞는 추천을 드릴게요. 먹고 싶은 음식이나 원하는 조건을 편하게 말해 주세요.😄\n
예를 들어 이런 질문이 가능해요.\n1. 요즘 다이어트 중인데, 칼로리가 낮은 메뉴 뭐가 있을까요?\n2. 헬스를 하고 있어서 고단백 음식을 먹어야합니다.\n3. 밥을 먹어야하는데 돈이 부족합니다. 양 많고 저렴한 음식을 찾고 있습니다."""
    st.session_state["messages"].append(
        ChatMessage(role="assistant", content=help_message)
    )

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("먹고 싶은 음식이나 원하는 조건을 편하게 말해 주세요 😄")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

if st.session_state["chain"] is None:
    st.session_state["chain"] = create_chain()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        add_message("user", user_input)
        st.chat_message("user").markdown(f"{user_input}")

        # 로딩 스피너 추가
        with st.spinner("원하시는 음식 정보를 준비하고 있어요! 잠시만 기다려주세요 😄"):
            # 스트리밍 호출
            config = {"configurable": {"session_id": "abc123"}}
            response = chain.stream({"question": user_input}, config=config)
            ai_answer = ""
            with st.chat_message("assistant"):
                container = st.empty()
                for token in response:
                    ai_answer += token
                    formatted_answer = (
                        ai_answer.replace(
                            "음식 및 판매처 정보", "\n**🍽️ 음식 및 판매처 정보**"
                        )
                        .replace("[구매방식]", "\n**[구매방식]**")
                        .replace("오프라인:", "\n**오프라인:**")
                        .replace("온라인:", "\n**온라인:**")
                        .replace("판매처:", "\n**판매처:**")
                        .replace("메뉴:", "\n**메뉴:**")
                        .replace("위치:", "\n**위치:**")
                        .replace("연락처:", "\n**연락처:**")
                        .replace("구매링크:", "\n**구매링크:**")
                        .replace("추천 이유:", "\n**추천 이유:**")
                    )
                    container.markdown(formatted_answer)

        # 대화기록을 저장한다.
        add_message("assistant", ai_answer)
