import streamlit as st
from querying import global_search, vector_search, local_search
import asyncio

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In Streamlit oder Jupyter
        return asyncio.ensure_future(coro)
    else:
        # Normaler Python-Prozess
        return asyncio.run(coro)
    
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

button_style = """
    <style>
    div.stButton > button {
        width: 100%;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>TrapGPT</h1>", unsafe_allow_html=True)

query = st.text_input("Stell deine Frage..")
answer = None
tokens = None
cost = None
error = None

col_left, col1, col2, col3, col_right = st.columns([2, 2, 2, 2, 2])

with col1:
    if st.button("Vector Search"):
        if query.strip():
            answer, tokens, cost = run_async(vector_search(query))
            st.session_state.answer = answer
            st.session_state.total_tokens += tokens
            st.session_state.total_cost += cost
        else:
            error = "Bitte gib zuerst eine Frage ein."

with col2:
    if st.button("Local Search"):
        if query.strip():
            answer, tokens, cost = run_async(local_search(query))
            st.session_state.answer = answer
            st.session_state.total_tokens += tokens
            st.session_state.total_cost += cost
        else:
            error = "Bitte gib zuerst eine Frage ein."

with col3:
    if st.button("Global Search"):
        if query.strip():
            answer, tokens, cost = run_async(global_search(query))
            st.session_state.answer = answer
            st.session_state.total_tokens += tokens
            st.session_state.total_cost += cost
        else:
            error = "Bitte gib zuerst eine Frage ein."

if error:
    st.error(error)
elif answer:
    st.markdown("#### Antwort")
    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 16px 20px; border-radius: 10px;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); font-size: 1rem;
                    border: 1px solid #e6e6e6;">
            {answer}
        </div>
        """,
        unsafe_allow_html=True
    )

    if tokens is not None and cost is not None:
        st.markdown(
    f"""
    <div style="margin-top: 12px; font-size: 0.95rem;">
        📈 <b>Tokenverbrauch:</b> {tokens:,} &nbsp;&nbsp;&nbsp;
        💸 <b>Geschätzte Kosten:</b> {cost:.4f} €
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div style='position: relative; height: 60px;'>
        <div style='position: fixed; bottom: 50px; right: 80px;
                    background-color: #f5f5f5; padding: 8px 16px;
                    border-radius: 8px; font-size: 0.9rem;
                    box-shadow: 0 0 5px rgba(0,0,0,0.1);'>
            📈 <b>Tokens:</b> {st.session_state.total_tokens:,} &nbsp;&nbsp;
            💸 <b>Kosten:</b> {st.session_state.total_cost:.2f} €
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
