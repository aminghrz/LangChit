# Customizable and portable LangGraph Streamlit UI with Persistent Memory

A full-featured Streamlit chat application powered by LangGraph with persistent memory capabilities, user authentication, and web search integration.

![Chat Interface](/img/1.png)
![Chat Interface](/img/2.png)
![Chat Interface](/img/3.png)

## Overview

This is a general-purpose Streamlit chat interface for serving LangGraph agents with advanced memory management and web search capabilities. The graph architecture can be easily modified or replaced with <ins>**YOUR CUSTOM IMPLEMENTATION**</ins>, making it perfect for testing or providing chat services.

## Features

### 🧠 **Dual-Layer Memory System**
1. **Thread-level Memory**: Persistent conversation memory using SQLite as checkpointer
2. **User-level Memory**: Cross-thread persistent memory using [SqliteVecStore](https://github.com/aminghrz/langmem-sqlite-vec) for vector storage

### 💾 **Zero-Setup Persistence**
- No external database setup required
- Creates a single SQLite database for both checkpointer and vector store
- Automatic database initialization

### 🔄 **Smart Context Management**
- Generates conversation summaries after 3 rounds (based on 10 previous messages)
- Reduces token consumption in long threads while maintaining context
- Summary stored in checkpointer for persistence

### 🔐 **User Management**
- Built-in authentication system
- User-specific thread access
- Secure session management
- Sign-up for new users

### 💬 **Thread Management**
- Multiple conversation threads per user
- Thread selection and creation interface
- Complete message history preservation

### ⚡ **Real-time Updates**
- Live memory updates (both checkpointer and vector store)
- Streaming responses
- Dynamic UI updates

### 🔑 **User API Key & Endpoint Management**
- Each user can securely enter their own API key and Base URL for any OpenAI-compatible provider
- Credentials are stored only once per user in the same SQLite database as the store and checkpointer
- Enables flexible use of different LLM providers per user
- API provider model selection

### 🔍 **Web Search Integration** (NEW)
- **Toggle Web Search**: Enable/disable web search functionality on demand
- **DuckDuckGo Integration**: Privacy-focused web search using the `ddgs` library
- **Dual Search Modes**:
  - **RAG Mode**: Stores search results as embeddings and uses vector search to find the most relevant information
  - **Direct Mode**: Passes all search results directly to the LLM for processing
- **Configurable Results**: Adjustable number of search results (1-10) via slider
- **Persistent Web Search Memory**: Search results are stored in a separate namespace (`web_search`) for future reference
- **Dynamic Graph Rebuilding**: The agent graph automatically rebuilds when web search settings change

## Architecture

The application uses a LangGraph workflow with the following components:

- **ReAct Agent**: Handles conversation logic with memory and web search tools
- **Memory Tools**: Powered by langmem for storing and retrieving user memories
- **Web Search Tool**: DuckDuckGo integration for current information retrieval
- **Summarization Node**: Automatically creates conversation summaries
- **SQLite Persistence**: Triple storage for checkpoints, vector embeddings, and web search results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/aminghrz/langgraph_streamlit_UI.git
cd langgraph_streamlit_UI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add [SqliteVecStore](https://github.com/aminghrz/langmem-sqlite-vec) to your directory.

4. Run the application:
```bash
streamlit run app.py
```
5. Sign-up and/or sign-in using the widgets.

6. Set up your OpenAI API key: 
Only one time, insert your OpenAI compatible API endpoint token and URL after sign-up/sign-in in the sidebar widget. Change anytime you want.

7. Configure web search (optional):
   - Toggle "Enable Web Search" in the sidebar
   - Choose between RAG or Direct search methods
   - Adjust the number of results with the slider

You're all set!

## Authentication Setup

Create a `config.yaml` file containing your users information in the format below:
```yaml
credentials:
  usernames:
    your_username:
      email: your_email@example.com
      first_name: Your
      last_name: Name
      password: pass
      roles: [admin, editor, viewer]
cookie:
  expiry_days: 30
  key: your_secret_key
  name: your_cookie_name
```
Remember that passwords will be hashed everytime the user information is saved. You can change the password in .yaml file anytime you want.

### Graph Customization
The LangGraph workflow can be easily modified in the graph definition section. Key components:

- `call_model()`: Main conversation handler
- `summarize_conversation()`: Summary generation logic
- `should_continue()`: Conditional logic for triggering summarization
- `search_web()`: Web search tool implementation

You can add your nodes, tools, conditional edges and graph (workflow) in graph.py.
Just keep the logic, and you will be good to go.

## Usage

1. **Login**: Use the sidebar authentication
2. **Sign-up**: Sign-up using the widget in first page
3. **Configure API**: Set your OpenAI-compatible API key and base URL
4. **Enable Web Search** (optional): 
   - Toggle web search in the sidebar
   - Select RAG for semantic search or Direct for full results
   - Set the number of results to retrieve
5. **Create Thread**: Click "➕ New Thread" to start a new conversation
6. **Switch Threads**: Select from existing threads in the dropdown
7. **Chat**: Type messages in the chat input
8. **View Summary**: Expand the conversation summary when available

### Web Search Usage Examples
- "Search the web for the latest news about AI"
- "Who is currently the president of the USA?"
- "Find information about climate change solutions"

## Dependencies

- `streamlit`: Web interface
- `langgraph`: Graph-based agent framework
- `langchain-openai`: OpenAI integration
- `langmem`: Memory management tools
- `streamlit-authenticator`: User authentication
- `sqlite-vec`: Vector storage (For SqliteVecStore)
- `ddgs`: DuckDuckGo search integration
- [`SqliteVecStore`](https://github.com/aminghrz/langmem-sqlite-vec): For persistent long-term memory

## ⚠️ Important Warnings

### Scalability
- **This is not production-ready for heavy loads**
- For production deployments, consider:
  - PostgreSQL instead of SQLite for the checkpointer
  - pgvector for vector storage
  - Proper connection pooling
  - Load balancing
  - Error handling and logging

### Performance
- Monitor token usage with long conversations (Use [LangFuse](https://langfuse.com/))
- Consider adjusting summary triggers based on your use case
- Implement rate limiting for production use
- Web search results add to token consumption - use RAG mode for efficiency

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
