# 🦾🤖 LangChit: A LangGraph Streamlit Chat Experience

_A full-featured, persistent, web-integrated Streamlit chat app built on LangGraph._

LangChit is a robust, user-friendly chat application—taking Streamlit and LangGraph to the next level. Built for power users, enthusiasts, and anyone who wants more from their chat interface. Includes long-term memory, and power-user control, LangChit offers features far beyond basic agent UIs—web page retrieval, local Deep Research inspired by Google Gemini, and more.

<sup>**Based on the [langgraph-streamlit-chat-interface](https://github.com/aminghrz/langgraph-streamlit-chat-interface), now with advanced multi-user and research features.**</sup>

---

## Screenshot Gallery

**Login and user management**
![Login Screen](/img/Login.jpg)

**Deep memory & threading**
![Persistent Memory](/img/Memory.jpg)

**Webpage retrieval, including JS-powered content**
![Webpage Retrieval (JS enabled)](/img/JS_page_retrieval.jpg)

**Integrated web search**
![Web Search](/img/Search.jpg)

**Gemini-style Deep Research Mode**
![Deep Research](/img/Deep_Reseaech.jpg)

---

## Why LangChit?

LangChit is for users and researchers who need:
- **Persistent, context-rich chat** – Memory at both thread and user level, always ready to continue your work.
- **Powerful online research** – Live web search and contextual retrieval, plus automatic in-depth research tools inspired by cutting-edge LLM frameworks.
- **Web content extraction & retrieval** – From static sites to dynamic JavaScript-driven pages.
- **Multi-user control and security** – Built-in authentication, per-user data, and secure API vault.

---

## Key Features

### 🧠 Intelligent, Persistent Memory
- _Thread-level:_ Each chat thread’s memory/history, checkpointed with SQLite.
- _User-level:_ Long-term, cross-thread memory. Uses [SqliteVecStore](https://github.com/aminghrz/langmem-sqlite-vec) for scalable vector storage.

### 🔄 Live Summaries & Efficient Context
- Automatic conversation summarization after every few messages—optimizes LLM context window usage.

### 🔑 API & Model Flexibility
- Each user can securely set and store their own LLM API key and endpoint (one-time), works with any OpenAI-compatible provider.

### 🧑‍💼 Multi-User, Multi-Thread
- Users can sign up, log in, and manage multiple chat threads—with all memory saved, per user.

### 🔎 Web Search & Retrieval (Advanced)
- **Integrated Private Web Search:** Privacy-focused DuckDuckGo search (via `ddgs`)—toggle on/off per user.
- **Retrieval Modes:** 
  - **RAG (Retrieval-Augmented Generation):** Embeds search results for LLM-optimized context.
  - **Direct Mode:** Directly feeds results to the LLM.
- **Results Control:** Slider for number of search results.
- **Persistent Web Search Memory:** All findings saved in a dedicated search memory namespace.
- **On-the-fly Reconfiguration:** Toggling search options triggers an automatic update to the agent graph.

### 🌐 Webpage Retrieval (Including Dynamic JavaScript Pages)
- **Built-in URL-based retrieval:** Enter a webpage and LangChit will attempt detailed extraction, including rendering JavaScript with headless browser tech.
- **Research assistant integration:** Perfect for news, blog, or academic content.

### 🔬 Gemini-Style Deep Research Tool
- **Not just a “web search” button:** LangChit includes a Deep Reasearch tool inspired by [Google’s Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart), orchestrating a workflow for in-depth context gathering and synthesis.
- You can have Deep Research in your local machine with any model you want from any procider. 

### 🔐 Secure, Ready-to-Use Authentication
- Credential management, hashed passwords, session handling, and per-user data isolation.

---

## Quickstart

1. **Clone this repo**
   ```bash
   git clone https://github.com/aminghrz/LangChit.git
   cd LangChit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add [SqliteVecStore](https://github.com/aminghrz/langmem-sqlite-vec) to your directory.**

4. **Prepare `config.yaml` for authentication (see below).**

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## Authentication Setup

Create a `config.yaml` file:

```yaml
credentials:
  usernames:
    your_username:
      email: your_email@example.com
      first_name: Your
      last_name: Name
      password: pass
      roles: [admin]
cookie:
  expiry_days: 30
  key: your_secret_key
  name: your_cookie_name
```
_Passwords will be hashed on save._  
_Edit YAML and restart to reset._

---

## Usage

1. **Sign up or sign in**
2. **(One time) Enter your LLM API key and endpoint in the sidebar**
3. **Enable web search, deep research, or webpage retrieval via sidebar toggles**
4. **Create/select threads, chat, and see summaries build automatically**
5. **Retrieve and synthesize fresh web content as you go**

---

## Dependencies

- `streamlit`
- `langgraph`
- `langchain-openai`
- `langmem` + [`sqlite-vec`](https://github.com/aminghrz/langmem-sqlite-vec) (for persistent memory layers)
- `streamlit-authenticator`
- `ddgs` (DuckDuckGo search)
- Headless browser (for JS retrieval; see requirements.txt)

---

## ⚠️ Note: For Research and Prototyping Use

Not production-scale. For production, upgrade to a robust DB (Postgres/pgvector), use proper connection pooling, logging, rate limiting, and host securely.

---

## Credits & Inspiration

- Built on [langgraph-streamlit-chat-interface](https://github.com/aminghrz/langgraph-streamlit-chat-interface)
- Deep research: inspired by [Google Gemini Fullstack LangGraph Quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart)
- Memory in SQLite: [langmem-sqlite-vec](https://github.com/aminghrz/langmem-sqlite-vec) 

---

## License

MIT

---

## Contributing

Pull requests and forks welcome. Issues? [File them here](https://github.com/aminghrz/LangChit/issues).
