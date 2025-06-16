import streamlit as st
import streamlit_authenticator as stauth
from streamlit_authenticator.utilities import LoginError
import sqlite3
from langchain_openai import OpenAIEmbeddings
from sqlite_vec_store import SqliteVecStore  
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime
from app_functions import (
    init_user_settings_db,
    save_user_settings,
    load_user_settings,
    get_thread_ids,
    load_messages_for_thread
)
import yaml
from yaml.loader import SafeLoader

st.set_page_config(layout="wide", page_title="LangGraph Chat Agent")

# Initialize user settings database
if "settings_db_initialized" not in st.session_state:
    init_user_settings_db(db="chatbot.sqlite3")
    st.session_state.settings_db_initialized = True

########################### Authentication ################################
with open('config.yaml', 'r', encoding='utf-8') as file:
    cred = yaml.load(file, Loader=SafeLoader)

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
        (email_of_registered_user,
        username_of_registered_user,
        name_of_registered_user) = authenticator.register_user(location='main',roles=['viewer'],)
        if email_of_registered_user:
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(cred, file, default_flow_style=False)
            st.success('User registered successfully! Please sign in using sidebar sign-in widget.')
    except LoginError as e:
        st.error(e)
    st.warning('Please enter your username and password')
    if st.session_state["authentication_status"]:
        st.rerun()
elif st.session_state["authentication_status"]:
    st.sidebar.write(f'Welcome *{st.session_state["name"]}*',)
########################### Authentication ################################

    ########################### User API Settings ################################    
    # Load existing settings for the user
    if "user_api_settings" not in st.session_state:
        st.session_state.user_api_settings = load_user_settings(db="chatbot.sqlite3", username=st.session_state["username"])
    
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
                save_user_settings(db="chatbot.sqlite3", username=st.session_state["username"], api_key=api_key_input.strip(), base_url=base_url_input.strip())
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

        # Import the new function
    from graph import create_graph
    
    # Initialize session state variables
    if "conn" not in st.session_state:
        st.session_state.conn = sqlite3.connect("chatbot.sqlite3", check_same_thread=False)
    
    if "store" not in st.session_state:
        embedding_model = OpenAIEmbeddings(
            model='text-embedding-3-large',
            api_key=st.session_state.user_api_settings["api_key"],
            base_url=st.session_state.user_api_settings["base_url"]
        )
        st.session_state.store = SqliteVecStore(
            db_file="chatbot.sqlite3",
            index={
                "dims": 3072,
                "embed": embedding_model,
            }
        )
    
    # Create graph if not exists or if API settings changed
    if "app" not in st.session_state or "last_api_settings" not in st.session_state or st.session_state.last_api_settings != st.session_state.user_api_settings:
        st.session_state.app, st.session_state.checkpointer = create_graph(
            api_key=st.session_state.user_api_settings["api_key"],
            base_url=st.session_state.user_api_settings["base_url"],
            conn=st.session_state.conn,
            store=st.session_state.store,
            user_id=st.session_state.user_id
        )
        st.session_state.last_api_settings = st.session_state.user_api_settings.copy()
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

                except Exception as e:
                    st.error(f"Error interacting with the agent: {e}")
                    import traceback
                    st.error(traceback.format_exc()) # Print full traceback for debugging

            st.rerun() 

    else:
        st.info("Please select a thread or start a new one from the sidebar to begin chatting.")
