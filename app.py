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
from openai import OpenAI

st.set_page_config(layout="wide", page_title="LangGraph Chat Agent")
st.markdown("""
<style>
h1, h2, h3, h4, h5, h6 {
    font-family: 'Segoe UI Emoji', 'Segoe UI Symbol', 'Segoe UI', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)
# Initialize user settings database
if "settings_db_initialized" not in st.session_state:
    try: 
        init_user_settings_db(db="chatbot.sqlite3")
        st.session_state.settings_db_initialized = True
        st.session_state.db_initialization_time = datetime.now()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        st.session_state.db_init_error = str(e)

########################### Start of Authentication ################################
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
    authenticator.logout(location='sidebar')
########################### End of Authentication ################################

    ########################### Start of User API Settings ################################    
    if "user_api_settings" not in st.session_state:
        st.session_state.user_api_settings = load_user_settings(db="chatbot.sqlite3", username=st.session_state["username"])
    settings_configured = bool(
        st.session_state.user_api_settings.get("api_key") and
        st.session_state.user_api_settings.get("base_url") and 
        st.session_state.user_api_settings.get("model")
    )

    @st.cache_resource
    def get_openai_client(api_key, base_url):
        """Caches the OpenAI client to avoid re-initializing on every rerun."""
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            return client
        except Exception as e:
            # We show error when called, but get_openai_client should return None so downstream knows.
            st.error(f"Failed to initialize OpenAI client: {e}")
            return None

    # Cache the model-list fetch so that repeated calls with same args don't refetch.
    @st.cache_data(show_spinner=False)
    def fetch_model_list(api_key, base_url):
        """Fetch model IDs once per unique (api_key, base_url)."""
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        # Depending on your client library: ensure models.data exists
        if getattr(models, "data", None):
            return [m.id for m in models.data]
        else:
            return []

    with st.sidebar.expander("🔧 API Settings", expanded=not settings_configured):
        # Text inputs for API key and Base URL
        api_key_input = st.text_input(
            "API Key:",
            value=st.session_state.user_api_settings.get("api_key", ""),
            type="password",
            help="Enter your API key",
            key="api_key_input"
        )
        base_url_input = st.text_input(
            "Base URL:",
            value=st.session_state.user_api_settings.get("base_url", ""),
            placeholder="https://api.example.com/v1",
            help="Enter the base URL for the API",
            key="base_url_input"
        )

        # Determine which values to use (prefer fresh inputs if non-empty)
        effective_api_key = api_key_input.strip() or st.session_state.user_api_settings.get("api_key", "").strip()
        effective_base_url = base_url_input.strip() or st.session_state.user_api_settings.get("base_url", "").strip()

        # Initialize client if possible
        if effective_api_key and effective_base_url:
            client = get_openai_client(effective_api_key, effective_base_url)
        else:
            client = None
            st.warning("Enter both API Key and Base URL to initialize client")

        # --- Manage session_state for model list caching per credential ---
        # If credentials changed since last fetch, clear stored model list & selection:
        prev_key = st.session_state.get("__api_key_for_models")
        prev_url = st.session_state.get("__base_url_for_models")
        if (effective_api_key and effective_base_url) and (effective_api_key != prev_key or effective_base_url != prev_url):
            # Credentials changed: clear previous model list & selection
            st.session_state["model_ids"] = []
            if "selected_model" in st.session_state:
                del st.session_state["selected_model"]
            st.session_state["__api_key_for_models"] = effective_api_key
            st.session_state["__base_url_for_models"] = effective_base_url

        # Initialize model_ids in session_state if not existing
        if "model_ids" not in st.session_state:
            st.session_state["model_ids"] = []

        # Only fetch if client is available AND we have not fetched yet
        if client:
            if not st.session_state["model_ids"]:
                # show spinner while fetching
                with st.spinner("Fetching available models..."):
                    try:
                        ids = fetch_model_list(effective_api_key, effective_base_url)
                        if ids:
                            st.session_state["model_ids"] = ids
                        else:
                            # Could be empty list: warn user
                            st.warning("No models returned by the API endpoint.")
                    except Exception as e:
                        st.error(f"Error fetching models: {e}")
                        # Leave model_ids empty so user can try again later
            # After attempted fetch, if we have model_ids, show dropdown
            if st.session_state["model_ids"]:
                model_ids = st.session_state["model_ids"]
                # Determine default index if already selected
                default_idx = 0
                if st.session_state.user_api_settings["model"]:
                    default_idx = model_ids.index(st.session_state.user_api_settings["model"])
                if ("selected_model" in st.session_state and st.session_state.selected_model in model_ids):
                    default_idx = model_ids.index(st.session_state.selected_model)
                st.selectbox(
                    "Choose a model:",
                    options=model_ids,
                    index=default_idx,
                    key="selected_model",
                    help="Select an available model from your endpoint"
                )
            else:
                # model_ids is empty after fetch attempt
                st.info("No available models to choose. Check your API settings or the endpoint.")
        else:
            # client is None
            st.warning("OpenAI client could not be initialized. Please check API key & Base URL.")

        # Save settings button
        if st.button("💾 Save API Settings"):
            if api_key_input.strip() and base_url_input.strip():
                save_user_settings(
                    db="chatbot.sqlite3",
                    username=st.session_state["username"],
                    api_key=api_key_input.strip(),
                    base_url=base_url_input.strip(),
                    model=st.session_state.selected_model
                )
                st.session_state.user_api_settings = {
                    "api_key": api_key_input.strip(),
                    "base_url": base_url_input.strip(),
                    "model": st.session_state.selected_model
                }
                st.success("API settings saved successfully!")
                st.rerun()
            else:
                st.error("Please fill in both API Key and Base URL")
    # Outside the expander, show a warning if not yet configured
    if not settings_configured:
        st.sidebar.warning("⚠️ Please configure your API settings")
    ########################### End of User API Settings ################################

    ########################### Start of Web Search Settings ################################
    with st.sidebar.expander("🔍 Web Search Settings", expanded=False):
        # Web search toggle
        web_search_enabled = st.checkbox(
            "Enable Web Search",
            value=st.session_state.get("web_search_enabled", False),
            help="Allow the assistant to search the web for current information"
        )
        
        # Show additional options only if web search is enabled
        if web_search_enabled:
            # Search method radio buttons
            search_method = st.radio(
                "Search Method:",
                options=["RAG", "Direct"],
                index=0 if st.session_state.get("search_method_rag", True) else 1,
                help="RAG: Use vector search on results\nDirect: Pass full results to LLM"
            )
            
            # Number of results slider
            num_results = st.slider(
                "Number of Results:",
                min_value=1,
                max_value=10,
                value=st.session_state.get("num_results", 5),
                help="Number of web search results to retrieve"
            )
            
            # Update session state
            st.session_state.web_search_enabled = web_search_enabled
            st.session_state.search_method_rag = (search_method == "RAG")
            st.session_state.num_results = num_results
        else:
            # Disable web search
            st.session_state.web_search_enabled = False
            st.session_state.search_method_rag = True
            st.session_state.num_results = 5

    # Check if web search settings changed
    web_search_settings_changed = False
    if "last_web_search_settings" in st.session_state:
        current_settings = {
            "enabled": st.session_state.web_search_enabled,
            "rag": st.session_state.search_method_rag,
            "num_results": st.session_state.num_results
        }
        if st.session_state.last_web_search_settings != current_settings:
            web_search_settings_changed = True
            st.session_state.last_web_search_settings = current_settings
    else:
        st.session_state.last_web_search_settings = {
            "enabled": st.session_state.web_search_enabled,
            "rag": st.session_state.search_method_rag,
            "num_results": st.session_state.num_results
        }
    ########################### End of Web Search Settings ################################


    st.session_state.user_id = st.session_state["username"]

    # Check if API settings are configured before proceeding
    if not st.session_state.user_api_settings.get("api_key") or not st.session_state.user_api_settings.get("base_url"):
        st.warning("⚠️ Please configure your API settings in the sidebar before using the chat.")
        st.stop()

    
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

    from graph import create_graph
    # Create graph if not exists or if API settings changed
    if  "app" not in st.session_state or \
        "last_api_settings" not in st.session_state or \
        st.session_state.last_api_settings != st.session_state.user_api_settings or \
        web_search_settings_changed:

        st.session_state.app, st.session_state.checkpointer = create_graph(
            model=st.session_state.user_api_settings["model"],
            api_key=st.session_state.user_api_settings["api_key"],
            base_url=st.session_state.user_api_settings["base_url"],
            conn=st.session_state.conn,
            store=st.session_state.store,
            user_id=st.session_state.user_id,
            web_search_enabled=st.session_state.get("web_search_enabled", False),
            search_method_rag=st.session_state.get("search_method_rag", True),
            num_results=st.session_state.get("num_results", 5)
        )
        st.session_state.last_api_settings = st.session_state.user_api_settings.copy()
        st.info("LangGraph app compiled with updated settings.")
    

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
            agent_config = {"configurable": {"thread_id": st.session_state.thread_id, "user_id": st.session_state.user_id}}

            with st.status("Processing your request...", expanded=True) as status:
                try:
                    step_logs = []
                    
                    for event in st.session_state.app.stream(graph_input, config=agent_config, stream_mode="updates"):
                        if event:
                            for node_name, node_data in event.items():
                                if node_name == "conversation":
                                    if "messages" in node_data:
                                        last_message = node_data["messages"][-1] if node_data["messages"] else None
                                        if last_message:
                                            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                                                for tool_call in last_message.tool_calls:
                                                    tool_name = tool_call.get('name', 'Unknown Tool')
                                                    tool_args = tool_call.get('args', {})
                                                    
                                                    if 'search' in tool_name.lower():
                                                        query = tool_args.get('query', 'information')
                                                        log_msg = f"🔍 Searching web for: {query[:50]}..."
                                                    elif 'memory' in tool_name.lower():
                                                        if 'manage' in tool_name.lower():
                                                            log_msg = "🧠 Storing information in memory"
                                                        else:
                                                            log_msg = "🧠 Retrieving from memory"
                                                    else:
                                                        log_msg = f"🔧 Using tool: {tool_name}"
                                                    
                                                    st.write(log_msg)
                                                    step_logs.append(log_msg)
                                            
                                            elif hasattr(last_message, 'content') and last_message.content:
                                                log_msg = "💭 Generating response..."
                                                st.write(log_msg)
                                                step_logs.append(log_msg)
                                
                                elif node_name == "summarize_conversation":
                                    log_msg = "📝 Summarizing conversation"
                                    st.write(log_msg)
                                    step_logs.append(log_msg)

                    # Update status to complete
                    status.update(label="✅ Complete!", state="complete")
                    
                    # After invocation, reload the state
                    updated_lc_messages = load_messages_for_thread(st.session_state.thread_id, st.session_state.checkpointer)
                    st.session_state.display_messages = updated_lc_messages

                    state_data = st.session_state.checkpointer.get(config=agent_config)
                    if state_data and "channel_values" in state_data and "summary" in state_data["channel_values"]:
                        st.session_state.current_summary = state_data["channel_values"]["summary"]

                except Exception as e:
                    status.update(label="❌ Error occurred", state="error")
                    st.error(f"Error interacting with the agent: {e}")
                    import traceback
                    st.error(traceback.format_exc())

            st.rerun() 

    else:
        st.info("Please select a thread or start a new one from the sidebar to begin chatting.")