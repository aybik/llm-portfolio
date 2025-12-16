# 🤖 Large Language Models Portfolio

A comprehensive collection of LLM projects, implementations, and experiments showcasing practical applications of modern language models, RAG systems, and AI agents.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-412991.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Technical Stack](#technical-stack)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Usage Examples](#usage-examples)
- [Key Learnings](#key-learnings)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [Contact](#contact)

## 🎯 Overview

This repository contains hands-on implementations and experiments with Large Language Models, covering everything from fundamental concepts to production-ready applications. Each project demonstrates practical problem-solving using state-of-the-art LLM technologies.

**Key Focus Areas:**
- 🔍 Retrieval-Augmented Generation (RAG)
- 🔗 LangChain & LangGraph workflows
- 💬 Conversational AI & Chatbots
- 📊 Semantic Search & Vector Databases
- 🎯 Prompt Engineering & Optimization
- 🤖 AI Agents & Tool Integration

## 🚀 Projects

### 1. RAG Systems

#### **Document Q&A with Semantic Search**
Advanced RAG implementation using vector databases for intelligent document retrieval and question answering.

**Technologies:** LangChain, ChromaDB/Qdrant, OpenAI Embeddings, FAISS
**Features:**
- Multi-document ingestion and chunking strategies
- Hybrid search (semantic + keyword)
- Context-aware answer generation
- Source attribution and citation

📂 Location: `projects/rag-systems/`

#### **Conversational RAG with Memory**
Enhanced RAG system with conversation history and context management.

**Technologies:** LangChain, Vector Stores, Memory Management
**Features:**
- Persistent conversation memory
- Follow-up question handling
- Context window optimization
- Multi-turn dialogue support

📂 Location: `projects/rag-systems/conversational-rag/`

---

### 2. LangChain Applications

#### **Multi-Agent System**
Coordinated AI agents working together to solve complex tasks.

**Technologies:** LangChain, LangGraph, Agent Tools
**Features:**
- Task decomposition and delegation
- Tool-augmented agents (web search, calculator, etc.)
- Agent orchestration and communication
- Error handling and retry logic

📂 Location: `projects/langchain-apps/multi-agent/`

#### **Custom Chain Architectures**
Various LangChain implementations for different use cases.

**Technologies:** LangChain, Custom Chains, Prompt Templates
**Examples:**
- Sequential chains for multi-step reasoning
- Router chains for dynamic task routing
- MapReduce chains for document summarization
- Transform chains for data processing

📂 Location: `projects/langchain-apps/custom-chains/`

---

### 3. Prompt Engineering

#### **Prompt Optimization Framework**
Systematic approach to crafting and testing prompts for optimal results.

**Features:**
- Prompt templates library
- A/B testing framework
- Performance metrics tracking
- Best practices documentation

📂 Location: `projects/prompt-engineering/`

#### **Few-Shot Learning Examples**
Collection of effective few-shot prompting strategies across domains.

**Domains:**
- Code generation
- Data extraction
- Text classification
- Creative writing

📂 Location: `projects/prompt-engineering/few-shot/`

---

### 4. Vector Databases & Embeddings

#### **Embedding Comparison Study**
Performance analysis of different embedding models and vector databases.

**Technologies:** OpenAI, Cohere, Sentence-Transformers, ChromaDB, Qdrant, Pinecone
**Metrics:**
- Retrieval accuracy
- Query latency
- Storage efficiency
- Cost analysis

📂 Location: `projects/vector-databases/`

---

### 5. Production Applications

#### **Intelligent Chatbot**
Production-ready chatbot with enterprise features.

**Technologies:** FastAPI, Streamlit, LangChain, Docker
**Features:**
- REST API endpoints
- Web UI interface
- User authentication
- Conversation persistence
- Rate limiting
- Logging and monitoring

📂 Location: `projects/production-apps/chatbot/`

## 🛠 Technical Stack

### **Core Technologies**
- **Languages:** Python 3.9+
- **LLM Providers:** OpenAI, Anthropic, HuggingFace
- **Frameworks:** LangChain, LangGraph, Haystack
- **Vector Databases:** ChromaDB, Qdrant, Pinecone, FAISS
- **Web Frameworks:** FastAPI, Streamlit, Gradio

### **Key Libraries**
```
langchain>=0.1.0
openai>=1.0.0
chromadb>=0.4.0
qdrant-client>=1.6.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4
tiktoken>=0.5.0
streamlit>=1.28.0
fastapi>=0.104.0
```

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip or poetry package manager
- API keys (OpenAI, Anthropic, etc.)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aybik/llm-portfolio.git
cd llm-portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run example project
cd projects/rag-systems
python document_qa.py
```

### Using Poetry (Recommended)

```bash
# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell

# Run any project
poetry run python projects/rag-systems/document_qa.py
```

## 📁 Repository Structure

```
llm-portfolio/
├── projects/
│   ├── rag-systems/
│   │   ├── document_qa/
│   │   ├── conversational_rag/
│   │   └── hybrid_search/
│   ├── langchain-apps/
│   │   ├── multi-agent/
│   │   ├── custom-chains/
│   │   └── tools-integration/
│   ├── prompt-engineering/
│   │   ├── templates/
│   │   ├── optimization/
│   │   └── few-shot/
│   ├── vector-databases/
│   │   ├── embeddings-comparison/
│   │   └── performance-benchmarks/
│   └── production-apps/
│       ├── chatbot/
│       └── api-service/
├── notebooks/
│   ├── 01_llm_fundamentals.ipynb
│   ├── 02_langchain_basics.ipynb
│   ├── 03_rag_implementation.ipynb
│   ├── 04_prompt_engineering.ipynb
│   └── 05_advanced_techniques.ipynb
├── utils/
│   ├── embeddings.py
│   ├── vector_stores.py
│   ├── prompts.py
│   └── metrics.py
├── tests/
├── docs/
│   ├── architecture.md
│   ├── best-practices.md
│   └── tutorials/
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## 💡 Usage Examples

### RAG System Example

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Initialize components
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# Query the system
response = qa_chain.run("What are the key findings in the document?")
print(response)
```

### Multi-Agent System Example

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Search the web for current information"
    ),
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Perform mathematical calculations"
    )
]

# Initialize agent
agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run agent
result = agent.run("What is the population of Berlin multiplied by 2?")
```

## 🎓 Key Learnings

### RAG Systems
- **Chunking Strategy:** Optimal chunk size depends on domain (500-1000 tokens for technical docs)
- **Retrieval Metrics:** Precision vs. recall tradeoff in semantic search
- **Context Window Management:** Techniques for handling long documents with limited context

### Prompt Engineering
- **Temperature Settings:** Lower (0.0-0.3) for factual, higher (0.7-1.0) for creative
- **Few-Shot Learning:** 3-5 examples provide optimal guidance
- **Chain-of-Thought:** Significantly improves reasoning on complex tasks

### Vector Databases
- **Embedding Models:** text-embedding-3-large offers best quality/cost ratio
- **Index Types:** HNSW for speed, IVF for memory efficiency
- **Hybrid Search:** Combining semantic + keyword search improves accuracy by 15-20%

### Production Considerations
- **Cost Optimization:** Caching, prompt compression reduce API costs by 40-60%
- **Latency:** Streaming responses improve perceived performance
- **Error Handling:** Retry logic with exponential backoff for API resilience

## 🗺 Future Roadmap

### Q1 2025
- [ ] Fine-tuning experiments with custom datasets
- [ ] Multi-modal RAG (text + images)
- [ ] Advanced agent architectures (ReAct, Plan-and-Execute)

### Q2 2025
- [ ] LLM evaluation framework
- [ ] Knowledge graph integration
- [ ] Edge deployment (GGUF, quantization)

### Q3 2025
- [ ] Agentic workflows for business processes
- [ ] LLM observability and monitoring
- [ ] Cost optimization techniques

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Run tests
pytest tests/

# Run linting
black .
flake8 .
mypy .
```

## 📧 Contact

**Aybik** - [GitHub](https://github.com/aybik)

Project Link: [https://github.com/aybik/llm-portfolio](https://github.com/aybik/llm-portfolio)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT models and APIs
- LangChain community for excellent frameworks
- ChromaDB and Qdrant teams for vector database solutions
- All contributors and open-source projects that made this possible

---

**⭐ If you find this repository useful, please consider giving it a star!**
