# 🚀 NOVA – GenAI Assistant

**Nova** is a locally powered Generative AI personal assistant built using Streamlit and Ollama. It provides an interactive AI experience where users can ask questions, analyze files, generate Python and SQL code, and work with data — without depending on paid cloud-based AI APIs.

The goal of Nova is to create a flexible, intelligent, and user-friendly AI assistant that combines conversational AI with practical data analysis capabilities.

## ✨ Key Features

### 🤖 1. AI-Powered Chat Assistant
- Interactive conversational interface.
- Powered by locally running LLMs through Ollama.
- Supports natural language queries.
- Context-aware conversations.

### 🧠 2. Multiple LLM Support
Nova is designed to support different open-source language models:

- Llama 3.2
- DeepSeek-R1
- Qwen 2.5

Users can switch between models based on their requirements.

### 📂 3. File Upload & Analysis
Upload files and interact with their contents.

Supported file formats include:
- PDF
- Excel (.xlsx, .xls)
- CSV
- Other supported document and binary formats.

### 💻 4. Code Generation
Nova can assist with:
- Python code generation.
- SQL query generation.
- Code explanations.
- Data analysis logic.

### 📊 5. Data Visualization
- Generate charts from uploaded datasets.
- Explore data through natural language prompts.
- Support data-driven analysis.

### 💾 6. Conversation Memory
- Maintain conversational context.
- Support persistent conversation history.
- Designed with multi-user support in mind.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| Ollama | Local LLM execution |
| Llama 3.2 | Language model |
| DeepSeek-R1 | Reasoning model |
| Qwen 2.5 | Language model |
| Pandas | Data manipulation |
| SQL | Database query generation |

## 🏗️ Project Architecture

User Input / File Upload  
↓  
Streamlit Interface  
↓  
Input Processing  
↓  
Ollama Local LLM  
↓  
Selected Language Model  
↓  
Response Generation / Code Generation / Data Analysis  
↓  
Display Results in Streamlit

## ⚙️ Installation & Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Nova
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Ollama

Download Ollama from:

https://ollama.com/

### Step 5: Download the Model

```bash
ollama pull llama3.2
```

Other supported models can also be downloaded:

```bash
ollama pull deepseek-r1
ollama pull qwen2.5
```

### Step 6: Run Nova

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access Nova.

## 🎯 Project Objectives

- Build a personal AI assistant using open-source LLMs.
- Run AI models locally without paid API dependencies.
- Integrate document understanding and data analysis.
- Enable Python and SQL code generation.
- Explore practical applications of Generative AI.

## 🔮 Future Enhancements

- Advanced Retrieval-Augmented Generation (RAG).
- Better document understanding.
- Support for scanned PDFs and image-based documents.
- Advanced conversational memory.
- Improved multi-user session management.
- Agentic AI capabilities.
- More advanced data analytics and visualization.

## 👩‍💻 Author

**Tanya Tiwari**

Data Scientist | Data Analytics | Generative AI | Python | SQL | Power BI

---

⭐ If you find this project interesting, consider giving it a star!
