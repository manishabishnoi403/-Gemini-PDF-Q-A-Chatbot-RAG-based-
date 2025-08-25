# 📄 Gemini PDF Q&A Chatbot (RAG-based)

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDFs and ask natural-language questions.  
Built with _Google Gemini, **LangChain, **ChromaDB, and \*\*Streamlit_.

## 🚀 Features

- Upload multiple PDFs 📂
- Ask questions in natural language 💬
- Get answers with sources 📖
- Adjustable chunk size, overlap, and retriever Top-K ⚙

## ⚙ Tech Stack

- Streamlit (UI)
- LangChain (framework)
- Gemini (LLM + embeddings)
- ChromaDB (vector store)
- PyPDFLoader (PDF text extraction)
- 
## 📷 Demo
Here’s how the chatbot looks in action:
<img width="1906" height="901" alt="image" src="https://github.com/user-attachments/assets/fa72b74d-f8ab-49fe-b406-6eecb68297bc" />

## 🏃 How to Run
```bash
git clone https://github.com/your-username/gemini-pdf-qa-chatbot.git
cd gemini-pdf-qa-chatbot
pip install -r requirements.txt
streamlit run new_app.py
```

