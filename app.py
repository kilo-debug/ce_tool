import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# RAG Components
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import qdrant_client.http.models as models

# Page config
st.set_page_config(
    page_title="Marketing RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_qdrant(embed_dim: int):
    """Initialize Qdrant with correct embedding dimension"""
    client = QdrantClient(":memory:")
    
    # Delete existing collections if wrong dimension
    try:
        client.delete_collection("conversion_data")
        client.delete_collection("engagement_data")
    except:
        pass
    
    # Create with correct dimension
    client.create_collection(
        collection_name="conversion_data",
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )
    client.create_collection(
        collection_name="engagement_data",
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
    )
    
    return client

@st.cache_data
def generate_sample_data():
    """Generate sample data - FIXED"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    campaigns = [f'Campaign_{i:03d}' for i in range(1, 31)]
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    
    # Conversion data
    conversion_data = []
    for i in range(150):
        campaign = np.random.choice(campaigns)
        product = np.random.choice(products)
        date = np.random.choice(dates)
        target_conversions = np.random.poisson(50)
        control_conversions = np.random.poisson(40)
        test_conversions = target_conversions + np.random.poisson(10)
        incremental = test_conversions - control_conversions
        sales = np.random.poisson(1000) * (1 + incremental/100.0)
        
        conversion_data.append({
            'campaign_id': campaign,
            'product': product,
            'date': date.strftime('%Y-%m-%d'),
            'target_conversions': target_conversions,
            'control_conversions': control_conversions,
            'test_conversions': test_conversions,
            'incremental_lift': incremental,
            'sales_revenue': float(sales),
            'conversion_rate': round(test_conversions / max(target_conversions, 1) * 100, 2),
            'roi': round(sales / max(target_conversions * 10, 1), 2)
        })
    
    # Engagement data
    channels = ['Email', 'Social_Facebook', 'Social_Instagram', 'Social_LinkedIn', 'Display', 'Search', 'Video', 'SMS']
    engagement_data = []
    for i in range(120):
        campaign = np.random.choice(campaigns)
        channel = np.random.choice(channels)
        date = np.random.choice(dates)
        impressions = np.random.poisson(50000)
        clicks = np.random.poisson(int(impressions * 0.02))
        engagements = np.random.poisson(int(clicks * 0.3))
        conversions = np.random.poisson(int(engagements * 0.05))
        
        engagement_data.append({
            'campaign_id': campaign,
            'channel': channel,
            'date': date.strftime('%Y-%m-%d'),
            'impressions': impressions,
            'clicks': clicks,
            'ctr': round(clicks / max(impressions, 1) * 100, 2),
            'engagements': engagements,
            'engagement_rate': round(engagements / max(clicks, 1) * 100, 2),
            'conversions': conversions,
            'cvr': round(conversions / max(engagements, 1) * 100, 2),
            'cost_per_click': round(np.random.uniform(0.5, 3.0), 2),
            'cost_per_conversion': round(np.random.uniform(20, 80), 0)
        })
    
    return pd.DataFrame(conversion_data), pd.DataFrame(engagement_data)

def create_vectors(client, df, collection_name, embeddings):
    """Create and index vectors - FIXED"""
    documents = []
    for _, row in df.iterrows():
        text = f"Campaign: {row['campaign_id']} | "
        if 'date' in row:
            text += f"Date: {row['date']} | "
        
        if collection_name == "conversion_data":
            text += (f"Product: {row['product']} | Target: {row['target_conversions']} | "
                    f"Control: {row['control_conversions']} | Test: {row['test_conversions']} | "
                    f"Lift: {row['incremental_lift']} | Sales: ${row['sales_revenue']:,.0f} | "
                    f"Conv Rate: {row['conversion_rate']}% | ROI: {row['roi']}")
        else:
            text += (f"Channel: {row['channel']} | Impressions: {row['impressions']:,} | "
                    f"Clicks: {row['clicks']:,} | CTR: {row['ctr']}% | "
                    f"Engagements: {row['engagements']:,} | Eng Rate: {row['engagement_rate']}% | "
                    f"Conversions: {row['conversions']} | CVR: {row['cvr']}% | "
                    f"CPC: ${row['cost_per_click']} | CPA: ${row['cost_per_conversion']}")
        
        documents.append(text)
    
    # Simple splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.create_documents(documents)
    
    points = []
    for i, doc in enumerate(splits):
        try:
            embedding = embeddings.embed_query(doc.page_content)
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={"text": doc.page_content, "table": collection_name}
            ))
        except Exception as e:
            st.warning(f"Skipping doc {i}: {str(e)[:100]}")
            continue
    
    if points:
        client.upsert(collection_name=collection_name, points=points)
    return len(points)

def get_model_components(model_provider):
    """Get model components with error handling"""
    try:
        if model_provider == "OpenAI":
            api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ Add OPENAI_API_KEY to Streamlit Secrets")
                st.stop()
            return (
                OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1536),
                ChatOpenAI(model="gpt-4o-mini", temperature=0),
                1536,
                "gpt-4o-mini"
            )
        else:  # Gemini
            api_key = st.secrets.get("GEMINI_API_KEY")
            if not api_key:
                st.error("❌ Add GEMINI_API_KEY to Streamlit Secrets")
                st.stop()
            return (
                GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0),
                768,
                "gemini-1.5-flash"
            )
    except Exception as e:
        st.error(f"Model setup error: {str(e)}")
        st.stop()

@st.cache_resource
def setup_rag_system(_model_provider, _embed_dim):
    """Complete RAG setup with proper caching"""
    model_provider = _model_provider
    embed_dim = _embed_dim
    
    embeddings, llm, dim, model_name = get_model_components(model_provider)
    client = init_qdrant(embed_dim)
    
    # Check if indexed
    if client.get_collection("conversion_data").points_count == 0:
        conv_df, eng_df = generate_sample_data()
        st.session_state.conv_df = conv_df
        st.session_state.eng_df = eng_df
        
        conv_count = create_vectors(client, conv_df, "conversion_data", embeddings)
        eng_count = create_vectors(client, eng_df, "engagement_data", embeddings)
        
        st.success(f"✅ Indexed {conv_count} conv + {eng_count} eng docs")
    
    return client, embeddings, llm

def format_docs(docs):
    return "\n\n".join([doc.payload["text"] for doc in docs])

def create_rag_chain(client, embeddings, llm):
    def retrieve_docs(query, k=6):
        results = []
        for collection in ["conversion_data", "engagement_data"]:
            try:
                hits = client.search(
                    collection_name=collection,
                    query_vector=embeddings.embed_query(query),
                    limit=k
                )
                results.extend(hits)
            except:
                continue
        return sorted(results, key=lambda x: x.score, reverse=True)[:k]
    
    def chain(query):
        docs = retrieve_docs(query)
        if not docs:
            return "No relevant data found. Try rephrasing your question."
        
        context = format_docs(docs)
        prompt = ChatPromptTemplate.from_template("""
You are a marketing analytics expert. Answer using ONLY this campaign data:

CONTEXT:
{context}

QUESTION: {question}

Provide specific numbers, comparisons, trends, and recommendations from the data.
""")
        
        full_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return full_chain.invoke({"context": context, "question": query})
    
    return chain

# Main App
def main():
    st.title("🤖 Marketing RAG Agent - FIXED")
    st.markdown("*Toggle OpenAI ↔ Gemini | 270+ marketing records*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Selection")
        model_options = ["OpenAI GPT-4o-mini", "Gemini 1.5 Flash"]
        selected_model = st.radio("Choose model:", model_options, index=1)
        model_provider = "OpenAI" if "OpenAI" in selected_model else "Gemini"
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Data", type="secondary"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
        
        model_info = {
            "OpenAI": {"dim": 1536, "name": "gpt-4o-mini"},
            "Gemini": {"dim": 768, "name": "gemini-1.5-flash"}
        }
        st.info(f"**Model**: {model_info[model_provider]['name']} | **Dim**: {model_info[model_provider]['dim']}")
        
        # Setup
        client_key = f"client_{model_provider}"
        if client_key not in st.session_state:
            with st.spinner(f"🚀 Setting up {model_provider}..."):
                embed_dim = model_info[model_provider]["dim"]
                st.session_state[client_key] = setup_rag_system(model_provider, embed_dim)
                st.session_state.current_client_key = client_key
        
        client, embeddings, llm = st.session_state[st.session_state.current_client_key]
        
        # Data preview
        if 'conv_df' in st.session_state:
            st.subheader("📊 Sample Data")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(st.session_state.conv_df[['campaign_id', 'product', 'incremental_lift', 'roi']].head())
            with col2:
                st.dataframe(st.session_state.eng_df[['campaign_id', 'channel', 'ctr', 'cvr']].head())
    
    # Chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("💬 Ask: 'Best campaign ROI?' 'Facebook vs Instagram CTR?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                rag_chain = create_rag_chain(client, embeddings, llm)
                response = rag_chain(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
