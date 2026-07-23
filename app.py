import os
import streamlit as st
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
)

# ----------------------------
# Step 1 - Building the RAG retrievers (cached so it runs only once)
# ----------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def build_retriver(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        document = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(document)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    acedemic_retriever = build_retriver("academics_handbook.pdf")
    fee_retriever = build_retriver("fee_structure.pdf")

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", 
                   api_key=os.environ.get("GROQ_API_KEY"), 
                   temperature=0.4)

    return acedemic_retriever, fee_retriever, llm


acedemic_retriever, fee_retriever, llm = load_resources()


# ----------------------------
# Step 2 - State
# ----------------------------
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ----------------------------
# Step 3 - Nodes generation
# ----------------------------
def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""

    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state["messages"][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


# ----------------------------
# Step 4 - router function
# ----------------------------
def route_query(state: State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


# ----------------------------
# Step 5 - Building the graph (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def build_graph():
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")

    graph.add_conditional_edges("classifier", route_query)

    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


app = build_graph()

# ----------------------------
# Step 6 - Streamlit UI
# ----------------------------

# --- Custom CSS ---
# Visual identity: a campus registrar's ledger — deep academic green,
# brass/gold accents, a serif crest header, and routing badges styled
# like office stamps for the three paths (academic / fee / general).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Petrona:ital,wght@0,500;0,600;0,700;1,500&family=Source+Serif+4:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }

html, body, [class*="css"] {
    font-family: 'Source Serif 4', serif;
}

.stApp {
    background:
        radial-gradient(900px 480px at 15% -10%, #16302a 0%, transparent 60%),
        radial-gradient(800px 460px at 100% 0%, #122922 0%, transparent 55%),
        #0d1f19;
}

.block-container {
    padding-top: 2.6rem;
    max-width: 760px;
}

/* ---------- Header ---------- */
.main-header {
    text-align: center;
    padding: 0.4rem 0 1.6rem 0;
    border-bottom: 1px solid rgba(201,162,39,0.22);
    margin-bottom: 1.6rem;
}
.crest {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 52px;
    height: 52px;
    border-radius: 50%;
    border: 1.5px solid rgba(201,162,39,0.55);
    font-size: 1.5rem;
    margin-bottom: 0.7rem;
    background: rgba(201,162,39,0.08);
}
.main-header h1 {
    font-family: 'Petrona', serif;
    font-weight: 600;
    font-size: 2.15rem;
    color: #f2ead2;
    letter-spacing: -0.01em;
    margin: 0 0 0.35rem 0;
}
.main-header p {
    font-family: 'IBM Plex Mono', monospace;
    color: #9fb8ab;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    margin: 0;
}

/* ---------- Sidebar: styled like a registration card ---------- */
section[data-testid="stSidebar"] {
    background: #0f231d;
    border-right: 1px solid rgba(201,162,39,0.18);
}
section[data-testid="stSidebar"] * {
    color: #dce7e0;
}
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Petrona', serif;
    color: #f2ead2;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(201,162,39,0.2);
}
div[data-testid="stSelectbox"] > div {
    border-radius: 8px;
}

/* ---------- Chat bubbles ---------- */
div[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 0.3rem 0.4rem;
    margin-bottom: 0.4rem;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
    background: rgba(201,162,39,0.07);
    border: 1px solid rgba(201,162,39,0.18);
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
    background: #f2ead6;
    border: 1px solid rgba(0,0,0,0.05);
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) p,
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) li {
    color: #241f14 !important;
}

div[data-testid="stChatInput"] {
    border-radius: 12px;
    border: 1px solid rgba(201,162,39,0.3);
}

/* ---------- Routing badges (office stamps) ---------- */
.query-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 12px;
    border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}
.query-badge::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}
.badge-academic { background: rgba(93,141,224,0.16); color: #7fa4ef; border: 1px solid rgba(127,164,239,0.35); }
.badge-fee { background: rgba(201,162,39,0.16); color: #d9b64a; border: 1px solid rgba(217,182,74,0.4); }
.badge-general { background: rgba(94,166,110,0.16); color: #7fc491; border: 1px solid rgba(127,196,145,0.4); }

/* ---------- Buttons ---------- */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    border-radius: 8px;
    border: 1px solid rgba(201,162,39,0.35);
    background: rgba(201,162,39,0.08);
    color: #f2ead2;
}
div.stButton > button:hover {
    border-color: rgba(201,162,39,0.6);
    background: rgba(201,162,39,0.14);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <div class="crest">🎓</div>
    <h1>College Assistant</h1>
    <p>ACADEMICS &nbsp;·&nbsp; FEES &nbsp;·&nbsp; GENERAL ENQUIRIES</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar: programme selection ---
with st.sidebar:
    st.markdown("### Student card")

    programme_map = {
        "BCA": "BCA",
        "BBA": "BBA",
        "B.Com (H)": "B.Com (H)",
    }

    student_programme = st.selectbox(
        "Programme",
        options=list(programme_map.keys()),
        index=0,
    )

    st.caption(f"Registered as **{student_programme}**")

    st.markdown("---")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.lc_messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**How questions are routed**")
    st.markdown(
        '<span class="query-badge badge-academic">ACADEMIC</span> handbook lookup',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="query-badge badge-fee">FEE</span> fee structure lookup',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="query-badge badge-general">GENERAL</span> answered directly',
        unsafe_allow_html=True,
    )

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # for display: list of {"role", "content", "query_type"}

if "lc_messages" not in st.session_state:
    st.session_state.lc_messages = []  # actual langgraph message history (human/ai tuples)

# --- Render chat history ---
for msg in st.session_state.messages:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant" and msg.get("query_type"):
            badge_class = f"badge-{msg['query_type']}"
            st.markdown(
                f'<span class="query-badge {badge_class}">{msg["query_type"].upper()}</span>',
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])

# --- Chat input ---
user_query = st.chat_input("Type your question here...")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    # Append to LangGraph message history
    st.session_state.lc_messages.append(("human", user_query))

    # Invoke the graph
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            result = app.invoke({
                "programme": student_programme,
                "messages": st.session_state.lc_messages
            })

            ai_response = result["messages"][-1].content
            query_type = result.get("query_type", "general")

            badge_class = f"badge-{query_type}"
            st.markdown(
                f'<span class="query-badge {badge_class}">{query_type.upper()}</span>',
                unsafe_allow_html=True
            )
            st.markdown(ai_response)

    # Update histories
    st.session_state.lc_messages = result["messages"]
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response,
        "query_type": query_type
    })