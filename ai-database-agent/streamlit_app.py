import streamlit as st
import uuid
from datetime import datetime
import os
from google.cloud import bigquery
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextEmbeddingModel
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAI
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain_google_vertexai import VertexAI
import json

# Page config
st.set_page_config(
    page_title="AI Database Agent",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_clients():
    """Initialize Google Cloud clients using service account"""
    try:
        # Get credentials from Streamlit secrets
        credentials_dict = st.secrets["google_cloud"]
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform",
                "https://www.googleapis.com/auth/bigquery",
                "https://www.googleapis.com/auth/cloud-platform.read-only"
            ]
        )

        project_id = credentials_dict["project_id"]

        # Initialize clients
        client = bigquery.Client(credentials=credentials, project=project_id)
        vertexai.init(project=project_id, credentials=credentials)

        return client, project_id, credentials
    except Exception as e:
        st.error(f"Authentication failed: {e}")
        return None, None, None


@st.cache_resource
def initialize_models(_credentials, project_id):
    """Initialize AI models"""
    try:
        # Embedding model
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

        # Chat model
        chat_model = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            api_key=st.secrets["anthropic_api_key"]
        )

        # SQL agent (using Vertex AI for SQL)
        vertex_llm = VertexAI(
            model_name="gemini-pro",
            project=project_id,
            credentials=_credentials
        )

        return embedding_model, chat_model, vertex_llm
    except Exception as e:
        st.error(f"Model initialization failed: {e}")
        return None, None, None


def create_vector_table(client, project_id):
    """Create vector table if it doesn't exist"""
    dataset_id = "guyb_sandbox"
    table_id = "conversations_vector"

    query = f"""
    CREATE TABLE IF NOT EXISTS `{project_id}.{dataset_id}.{table_id}` (
        session_id STRING,
        timestamp TIMESTAMP,
        user_message STRING,
        ai_response STRING,
        user_embedding ARRAY<FLOAT64>,
        ai_embedding ARRAY<FLOAT64>
    )
    """

    try:
        client.query(query)
        return True
    except Exception as e:
        st.error(f"Table creation failed: {e}")
        return False


def get_embedding(text, embedding_model):
    """Convert text to vector embedding"""
    try:
        embeddings = embedding_model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")
        return None


def save_conversation_with_vectors(client, project_id, session_id, user_msg, ai_response, embedding_model):
    """Save conversation with embeddings to BigQuery"""
    try:
        # Generate embeddings
        user_embedding = get_embedding(user_msg, embedding_model)
        ai_embedding = get_embedding(ai_response, embedding_model)

        if not user_embedding or not ai_embedding:
            return False

        # Insert to BigQuery
        rows = [{
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": user_msg,
            "ai_response": ai_response,
            "user_embedding": user_embedding,
            "ai_embedding": ai_embedding
        }]

        table_ref = client.dataset("guyb_sandbox").table("conversations_vector")
        errors = client.insert_rows_json(table_ref, rows)

        return len(errors) == 0
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")
        return False


def find_similar_conversations(client, project_id, query_text, embedding_model, limit=3):
    """Find conversations similar to query using vector search"""
    try:
        # Get embedding for query
        query_embedding = get_embedding(query_text, embedding_model)
        if not query_embedding:
            return []

        # Vector similarity search in BigQuery
        query = f"""
        SELECT 
            session_id,
            user_message,
            ai_response,
            timestamp,
            ML.DISTANCE(user_embedding, {query_embedding}) as similarity
        FROM `{project_id}.guyb_sandbox.conversations_vector`
        ORDER BY similarity ASC
        LIMIT {limit}
        """

        results = client.query(query)
        return list(results)
    except Exception as e:
        st.error(f"Semantic search failed: {e}")
        return []


def chat_with_semantic_memory(client, project_id, session_id, user_message, chat_model, embedding_model):
    """Chat with semantic context"""
    try:
        # Find relevant context
        relevant_context = find_similar_conversations(client, project_id, user_message, embedding_model, limit=3)

        context = ""
        if relevant_context:
            context = "Relevant past conversations:\n"
            for convo in relevant_context:
                context += f"Q: {convo.user_message}\nA: {convo.ai_response}\n\n"

        prompt = f"""{context}Current question: {user_message}
        Please respond based on the context and current question."""

        response = chat_model.invoke(prompt)
        return response.content
    except Exception as e:
        st.error(f"Chat failed: {e}")
        return "I'm sorry, I encountered an error while processing your request."


def create_sql_agent_query(client, project_id, user_message, vertex_llm):
    """Handle SQL database queries"""
    try:
        # Create a simple SQL database connection string for BigQuery
        # Note: This is a simplified approach - you might need to adjust based on your setup

        # For now, let's use a direct BigQuery approach
        db_keywords = ["table", "database", "query", "sql", "count", "show", "select"]

        if any(keyword in user_message.lower() for keyword in db_keywords):
            # This is a placeholder - you'd need to implement proper SQL agent integration
            return f"I understand you're asking about database operations: '{user_message}'. Please implement the SQL agent integration based on your specific database schema."
        else:
            return None
    except Exception as e:
        st.error(f"SQL query failed: {e}")
        return None


def unified_agent_response(client, project_id, session_id, user_message, chat_model, vertex_llm, embedding_model):
    """Unified agent that handles both SQL and chat queries"""

    # Check if it's a database question
    db_keywords = ["table", "database", "query", "sql", "count", "show", "select"]
    is_db_question = any(keyword in user_message.lower() for keyword in db_keywords)

    if is_db_question:
        # Try SQL agent first
        sql_response = create_sql_agent_query(client, project_id, user_message, vertex_llm)
        if sql_response:
            return sql_response

    # Use chat with semantic memory
    return chat_with_semantic_memory(client, project_id, session_id, user_message, chat_model, embedding_model)


def main():
    st.title("🤖 AI Database Agent")
    st.markdown("Chat with your data using natural language and AI memory")

    # Initialize clients
    client, project_id, credentials = initialize_clients()

    if not client:
        st.error("Failed to initialize Google Cloud clients. Please check your credentials.")
        st.stop()

    # Initialize models
    embedding_model, chat_model, vertex_llm = initialize_models(credentials, project_id)

    if not all([embedding_model, chat_model, vertex_llm]):
        st.error("Failed to initialize AI models. Please check your API keys.")
        st.stop()

    # Create vector table
    if not create_vector_table(client, project_id):
        st.warning("Vector table creation failed. Some features may not work.")

    # Sidebar controls
    st.sidebar.header("🛠️ Controls")
    auto_save = st.sidebar.checkbox("Auto-save conversations to memory", value=True)

    # Display current session info
    st.sidebar.info(f"Project: {project_id}")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

    st.sidebar.info(f"Session: {st.session_state.session_id[:8]}...")

    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your data or chat normally..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = unified_agent_response(
                    client, project_id, st.session_state.session_id,
                    prompt, chat_model, vertex_llm, embedding_model
                )

                st.markdown(response)

                # Save to vector database if enabled
                if auto_save:
                    success = save_conversation_with_vectors(
                        client, project_id, st.session_state.session_id,
                        prompt, response, embedding_model
                    )
                    if success:
                        st.success("💾 Saved to memory", icon="✅")

        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()