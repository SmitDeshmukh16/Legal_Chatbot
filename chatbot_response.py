import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

chat_history = []

# Load embeddings and FAISS DB
embeddings = HuggingFaceEmbeddings()
vector_db = FAISS.load_local("dataset", embeddings, allow_dangerous_deserialization=True)

# Initialize Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

# Create custom prompt template
custom_prompt = PromptTemplate.from_template("""
You are a legal assistant.

Conversation history:
{chat_history}

User's query:
{question}

Context from documents:
{context}

Instructions:
- If the question is unrelated to legal matters, reply exactly: "I can only assist with legal matters."
- If the provided context has relevant information, answer using only that context.
- If the context does not have the answer, give a correct, general legal explanation using your own knowledge.
- Keep responses **factual** and **well-structured**.
- Cite **relevant Indian laws/regulations** when possible.
- Format output in **Markdown** using headings, bullet points, numbered lists, and emphasis.
""")

# Create chain with custom prompt
retriever = vector_db.as_retriever()
conversational_qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

def get_response(query):
    response = conversational_qa.invoke({"question": query, "chat_history": chat_history})
    chat_history.append((query, response['answer']))
    print(query)
    print(response["answer"])
    return response["answer"]

# Test
if __name__ == "__main__":
    get_response("How do I file a complaint?")
