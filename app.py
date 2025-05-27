import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import LoginError
import sqlite3

st.set_page_config(layout="wide", page_title="LangGraph Chat Agent")

########################### Database Setup for User Settings ################################
def init_user_settings_db():
    """Initialize the user settings database"""
    conn = sqlite3.connect("chatbot.sqlite3", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            username TEXT PRIMARY KEY,
            api_key TEXT,
            base_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

def save_user_settings(username, api_key, base_url):
    """Save user API settings to database"""
    conn = sqlite3.connect("chatbot.sqlite3", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO user_settings (username, api_key, base_url, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
    ''', (username, api_key, base_url))
    conn.commit()
    conn.close()

def load_user_settings(username):
    """Load user API settings from database"""
    conn = sqlite3.connect("chatbot.sqlite3", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('SELECT api_key, base_url FROM user_settings WHERE username = ?', (username,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {"api_key": result[0], "base_url": result[1]}
    return {"api_key": "", "base_url": ""}

# Initialize user settings database
if "settings_db_initialized" not in st.session_state:
    init_user_settings_db()
    st.session_state.settings_db_initialized = True

########################### Authentication ################################
cred = {
    "credentials": {
    "usernames": {
    "AminGhz": {
        "email": "amin.ghareyazi@gmail.com",
        "failed_login_attempts": 0,
        "first_name": "Amin",
        "last_name": "Ghareyazi",
        "logged_in": False,
        "password": "amin123",
        "roles": ["admin", "editor", "viewer"],
    },
    "mammad": {
        "email": "ghareyazi.a@gmail.com",
        "failed_login_attempts": 0,
        "first_name": "Rebecca",
        "last_name": "Briggs",
        "logged_in": False,
        "password": "def",
        "roles": ["viewer"],
    },
    }
    },
    "cookie": {
    "expiry_days": 0,
    "key": '42',
    "name": "ahmad"
    }
}

# Creating the authenticator object
authenticator = stauth.Authenticate(
    cred['credentials'],
    cred['cookie']['name'],
    cred['cookie']['key'],
    cred['cookie']['expiry_days']
)

if st.session_state["authentication_status"] is False:
 st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    try:
        authenticator.login(location = "sidebar",clear_on_submit = True)
    except LoginError as e:
        st.error(e)
    st.warning('Please enter your username and password')
    if st.session_state["authentication_status"]:
        st.rerun()
########################### Authentication ################################

elif st.session_state["authentication_status"]:
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*',)

    ########################### User API Settings ################################    
    # Load existing settings for the user
    if "user_api_settings" not in st.session_state:
        st.session_state.user_api_settings = load_user_settings(st.session_state["username"])
    
    # Check if settings are already configured
    settings_configured = bool(st.session_state.user_api_settings.get("api_key") and st.session_state.user_api_settings.get("base_url"))
    
    # Create expander - collapsed if settings are configured, expanded if not
    with st.sidebar.expander("🔧 API Settings", expanded=not settings_configured):
        # API Key input
        api_key_input = st.text_input(
            "API Key:",
            value=st.session_state.user_api_settings.get("api_key", ""),
            type="password",
            help="Enter your API key",
            key="api_key_input"
        )
        
        # Base URL input
        base_url_input = st.text_input(
            "Base URL:",
            value=st.session_state.user_api_settings.get("base_url", ""),
            help="Enter the base URL for the API",
            placeholder="https://api.example.com/v1",
            key="base_url_input"
        )
        
        # Save settings button
        if st.button("💾 Save API Settings"):
            if api_key_input.strip() and base_url_input.strip():
                save_user_settings(st.session_state["username"], api_key_input.strip(), base_url_input.strip())
                st.session_state.user_api_settings = {"api_key": api_key_input.strip(), "base_url": base_url_input.strip()}
                st.success("API settings saved successfully!")
                st.rerun()
            else:
                st.error("Please fill in both API Key and Base URL")
    
    # Display current settings status outside the expander
    if not settings_configured:
        st.sidebar.warning("⚠️ Please configure your API settings")
    ########################### End User API Settings ################################

    authenticator.logout(location='sidebar')

    st.session_state.user_id = st.session_state["username"]

    # Check if API settings are configured before proceeding
    if not st.session_state.user_api_settings.get("api_key") or not st.session_state.user_api_settings.get("base_url"):
        st.warning("⚠️ Please configure your API settings in the sidebar before using the chat.")
        st.stop()

    import streamlit as st
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
    from langgraph.graph import MessagesState, StateGraph, START, END
    from langgraph.checkpoint.sqlite import sqlite3, SqliteSaver
    from langchain_openai import ChatOpenAI
    from typing import Literal, List, Dict, Any
    from langchain_openai import OpenAIEmbeddings
    from sqlite_vec_store import SqliteVecStore
    from langgraph.prebuilt import create_react_agent
    from langmem import create_manage_memory_tool, create_search_memory_tool
    from datetime import datetime

    # Use user's API settings instead of hardcoded values
    AVALAPIS_API_KEY = st.session_state.user_api_settings["api_key"]
    AVALAPIS_BASE_URL = st.session_state.user_api_settings["base_url"]

    embedding_model = OpenAIEmbeddings(
    # temperature=0,
    model='text-embedding-3-large', # Ensure this model is available at your endpoint
    api_key=AVALAPIS_API_KEY,
    base_url=AVALAPIS_BASE_URL
    )

    # --- LangGraph Setup ---

    # We will add a `summary` attribute (in addition to `messages` key, which MessagesState already has)
    class State(MessagesState):
        summary: str

    # We will use this model for both the conversation and the summarization
    # Define it once to be used by functions
    chat_model = ChatOpenAI(
    temperature=0,
    model='gpt-4.1',
    api_key=AVALAPIS_API_KEY,
    base_url=AVALAPIS_BASE_URL
    )

    st.session_state.store = SqliteVecStore(
    db_file="chatbot.sqlite3",
    index={
    "dims": 3072,
    "embed": embedding_model,
    }
    )

    manage_memory_tool = create_manage_memory_tool(store=st.session_state.store, namespace=("memory","{user_id}"))
    search_memory_tool = create_search_memory_tool(store=st.session_state.store, namespace=("memory","{user_id}"))

    react_agent_executor = create_react_agent(
    model=chat_model,
    tools=[manage_memory_tool, search_memory_tool],
    prompt=SystemMessage("You are a helpful assistant. Respond to the user's last message based on the provided context and conversation history and memories. Store any interests and topics the user talks about."),
    store=st.session_state.store
    )

    ############################### Call ReAct Agent and pass last 5 messages and summary if available ##############################
    def call_model(state: State):
        summary = state.get("summary", "")
        current_messages = state["messages"]
        last_messages_to_send = current_messages[-5:] # Send last 5 actual messages

        messages_for_react_agent_input: List[BaseMessage] = []
        if summary:
        # The summary is prepended as a system message for context
            system_message_content = f"Here is a summary of the conversation so far: {summary}. Use this to inform your response."
            messages_for_react_agent_input.append(SystemMessage(content=system_message_content))

        messages_for_react_agent_input.extend(last_messages_to_send)

        # The ReAct agent expects input in the format {"messages": [list_of_messages]}
        # The last message in this list should be the one it needs to respond to (typically HumanMessage).
        agent_input = {"messages": messages_for_react_agent_input}

        # Invoke the ReAct agent.
        # Since there are no tools, it will essentially be an LLM call structured by the ReAct framework.
        # The react_agent_executor is already compiled.
        response_dict = react_agent_executor.invoke(agent_input)

        # The ReAct agent's response (AIMessage) will be in the 'messages' key of the output dictionary.
        # It's a list, and the agent's response is typically the last message added.
        ai_response_message = response_dict["messages"][-1]

        if not isinstance(ai_response_message, AIMessage):
            # Fallback or error handling if the last message isn't an AIMessage
            # This shouldn't happen in a typical ReAct flow without tool errors.
            st.error("ReAct agent did not return an AIMessage as expected.")
            return {"messages": [AIMessage(content="Sorry, I encountered an issue.")]}

        # The graph will automatically append the response to state["messages"]
        # We just need to return the new message to be added
        return {"messages": [ai_response_message]}
    ############################################################################################################################

    ############################### Call model to summarize messages. Update summary if available ############################## 
    def summarize_conversation(state: State):
        summary = state.get("summary", "")
        current_messages = state["messages"]
        # Let's use more messages for a better summary, e.g., last 6 (3 turns) that led to summarization
        # The last two messages are the AI response that triggered the summary, and the user message before that.
        # We want to summarize the conversation *before* the current turn that might be too long.
        messages_to_summarize = current_messages[:-2] # Exclude the last AI response and user query that triggered summary
        if len(messages_to_summarize) > 10 : # Cap the number of messages to summarize to avoid large prompts
            messages_to_summarize = messages_to_summarize[-10:]


        if not messages_to_summarize: # Nothing to summarize yet (e.g., if called too early)
            return {"summary": summary, "messages": []}


        summary_prompt_parts = []
        if summary:
            summary_prompt_parts.append(f"This is the current summary of the conversation: {summary}\n")

        summary_prompt_parts.append("Based on the following recent messages:\n")
        for msg in messages_to_summarize:
            if isinstance(msg, HumanMessage):
                summary_prompt_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                summary_prompt_parts.append(f"Assistant: {msg.content}")

        summary_prompt_parts.append("\nPlease update or create a concise summary of the entire conversation.")

        final_summary_prompt = "\n".join(summary_prompt_parts)

        # Construct messages for summarization
        messages_for_summary_llm = [HumanMessage(content=final_summary_prompt)]

        response = chat_model.invoke(messages_for_summary_llm)
        return {"summary": response.content, "messages": []} 
    ############################################################################################################################

    ####################################### Logic for calling summarize_conversation node ######################################
    def should_continue(state: State) -> Literal["summarize_conversation", END]:
        messages = state["messages"]
        # Trigger summary if there are more than 6 messages (e.g., 3 user, 3 AI + 1 new user = 7 messages)
        # The summarization will happen *after* the AI responds to the current user message.
        if len(messages) > 6: 
            return "summarize_conversation"
        return END
    ############################################################################################################################

    # --- End of LangGraph Setup ---

    # --- Streamlit App ---

    # st.set_page_config(layout="wide", page_title="LangGraph Chat Agent")

    # Helper function to get thread IDs
    def get_thread_ids(conn, user_id):
        if conn:
            try:
                cursor = conn.cursor()
                query = """
                SELECT DISTINCT thread_id
                FROM checkpoints
                WHERE SUBSTR(thread_id, 1, INSTR(thread_id, '@') - 1) = ?
                ORDER BY thread_id DESC;
                """
                cursor.execute(query, (user_id,))
                return [item[0] for item in cursor.fetchall()]
            except sqlite3.OperationalError: # Table might not exist yet
                return []
        return []

    # Helper function to load messages for a thread
    def load_messages_for_thread(thread_id: str, checkpointer: SqliteSaver) -> List[Dict[str, Any]]:
        if not thread_id or not checkpointer:
            return []

        agent_config = {"configurable": {"thread_id": st.session_state.thread_id ,"user_id": st.session_state.user_id}}
        state_data = checkpointer.get(config=agent_config)

        if state_data and "channel_values" in state_data and "messages" in state_data["channel_values"]:
            raw_messages = state_data["channel_values"]["messages"]
            return raw_messages
        return []

    # Initialize session state variables
    if "conn" not in st.session_state:
        st.session_state.conn = sqlite3.connect("chatbot.sqlite3", check_same_thread=False)
        st.session_state.checkpointer = SqliteSaver(conn=st.session_state.conn)

        # Define and compile the graph (only once)
        workflow = StateGraph(State)
        # Nodes:
        workflow.add_node("conversation", call_model)
        workflow.add_node("summarize_conversation", summarize_conversation) 
        # Edges
        workflow.add_edge(START, "conversation")
        workflow.add_conditional_edges("conversation", should_continue)
        workflow.add_edge("summarize_conversation", END) 

        st.session_state.app = workflow.compile(checkpointer=st.session_state.checkpointer, store=st.session_state.store)
        st.info("LangGraph app compiled and checkpointer initialized.")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = None

    if "display_messages" not in st.session_state: # For displaying in Streamlit chat
        st.session_state.display_messages = []

    if "current_summary" not in st.session_state:
        st.session_state.current_summary = ""


    # --- Sidebar for Thread Management ---
    st.sidebar.title("Chat Threads")

    available_threads = get_thread_ids(st.session_state.conn, st.session_state.user_id)

    # Dropdown for selecting existing threads
    if available_threads:
        options = available_threads
        if st.session_state.thread_id and st.session_state.thread_id not in options:
            options = [st.session_state.thread_id] + options

        try:
            current_selection_index = options.index(st.session_state.thread_id) if st.session_state.thread_id in options else 0
        except ValueError:
            current_selection_index = 0 

        selected_thread = st.sidebar.selectbox(
            "Select a Thread:",
            options,
            index=current_selection_index,
            key="thread_selector" # Added key for stability
        )

        if selected_thread and selected_thread != st.session_state.thread_id:
            st.session_state.thread_id = selected_thread
            raw_lc_messages = load_messages_for_thread(st.session_state.thread_id, st.session_state.checkpointer)
            st.session_state.display_messages = raw_lc_messages

            agent_config = {"configurable": {"thread_id": st.session_state.thread_id ,"user_id": st.session_state.user_id}}
            state_data = st.session_state.checkpointer.get(config=agent_config)
            if state_data and "channel_values" in state_data and "summary" in state_data["channel_values"]:
                st.session_state.current_summary = state_data["channel_values"]["summary"]
            else:
                st.session_state.current_summary = ""
            st.rerun()
    elif not st.session_state.thread_id and not available_threads: # Show if no threads and no current selection
        st.sidebar.info("No threads yet. Click 'New Thread' to start.")


    # "Start New Thread" button
    if st.sidebar.button("➕ New Thread"):
        # A more robust way to generate new thread IDs if multiple users or sessions
        # For simplicity, we'll use a counter based on existing threads.
        # In a real app, consider UUIDs or a sequence from the DB.
        new_thread_id_num = f"{st.session_state.user_id}@{datetime.now().strftime('%Y%m%d_%H%M%SS')}"

        st.session_state.thread_id = str(new_thread_id_num)
        st.session_state.display_messages = []
        st.session_state.current_summary = ""
        st.success(f"Started new thread: {st.session_state.thread_id}")
        # Add new thread to available_threads for immediate selection if needed, though rerun handles it
        if st.session_state.thread_id not in available_threads:
            available_threads.insert(0, st.session_state.thread_id) # Prepend for visibility
        st.rerun() 


    # --- Main Chat Interface ---
    st.title("🤖 LangGraph Powered Chat")

    if st.session_state.thread_id:
        st.markdown(f"**Current Thread ID:** `{st.session_state.thread_id}`")
        if st.session_state.current_summary:
            with st.expander("Conversation Summary", expanded=False): # Start collapsed
                st.markdown(st.session_state.current_summary)

    # Display chat messages from history
        for msg in st.session_state.display_messages:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(msg.content)

    # Chat input for the user
        if prompt := st.chat_input("What would you like to discuss?"):
            st.session_state.display_messages.append(HumanMessage(content=prompt))
            with st.chat_message("user"):
                st.markdown(prompt)

            graph_input = {"messages": [HumanMessage(content=prompt)]}
            agent_config = {"configurable": {"thread_id": st.session_state.thread_id ,"user_id": st.session_state.user_id}}

            with st.spinner("AI is thinking..."):
                try:
                    # Stream events from the graph
                    # We don't need to iterate through events if we're just reloading state after
                    for _ in st.session_state.app.stream(graph_input, config=agent_config):
                        pass # Consume the stream

                    # After invocation, reload the state to get AI's response and any summary
                    updated_lc_messages = load_messages_for_thread(st.session_state.thread_id, st.session_state.checkpointer)
                    st.session_state.display_messages = updated_lc_messages

                    state_data = st.session_state.checkpointer.get(config=agent_config)
                    if state_data and "channel_values" in state_data and "summary" in state_data["channel_values"]:
                        st.session_state.current_summary = state_data["channel_values"]["summary"]
                    # else: # If summary node didn't run, current_summary remains unchanged
                    # st.session_state.current_summary = "" # This might clear a valid older summary

                except Exception as e:
                    st.error(f"Error interacting with the agent: {e}")
                    import traceback
                    st.error(traceback.format_exc()) # Print full traceback for debugging

            st.rerun() 

    else:
        st.info("Please select a thread or start a new one from the sidebar to begin chatting.")
