import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import List, Dict
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
import qdrant_client

# Page config
st.set_page_config(
    page_title="Marketing RAG Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_qdrant():
    """Initialize Qdrant client and collections"""
    client = QdrantClient(":memory:")
    
    client.recreate_collection(
        collection_name="conversion_data",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),  # OpenAI dim
    )
    
    client.recreate_collection(
        collection_name="engagement_data",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    
    return client

@st.cache_data
def generate_sample_data():
    """Generate 150+ sample records (unchanged)"""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='D')
    
    campaigns = [f'Campaign_{i:03d}' for i in range(1, 31)]
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    
    conversion_data = []
    for i in range(150):
        campaign = np.random.choice(campaigns)
        product = np.random.choice(products)
        date = np.random.choice(dates)
        
        target_conversions = np.random.poisson(50)
        control_conversions = np.random.poisson(40)
        test_conversions = target_conversions + np.random.poisson(10)
        incremental = test_conversions - control_conversions
        sales = np.random.poisson(1000) * (1 + incremental/100)
        
        conversion_data.append({
            'campaign_id': campaign,
            'product': product,
            'date': date.strftime('%Y-%m-%d'),
            'target_conversions': target_conversions,
            'control_conversions': control_conversions,
            'test_conversions': test_conversions,
            'incremental_lift': incremental,
            'sales_revenue': sales,
            'conversion_rate': test_conversions / (target_conversions + 1) * 100,
            'roi': sales / (target_conversions * 10)
        })
    
    channels = ['Email', 'Social_Facebook', 'Social_Instagram', 'Social_LinkedIn', 
               'Display', 'Search', 'Video', 'SMS']
    
    engagement_data = []
    for i in range(120):
        campaign = np.random.choice(campaigns)
        channel = np.random.choice(channels)
        date = np.random.choice(dates)
        
        impressions = np.random.poisson(50000)
        clicks = np.random.poisson(impressions * 0.02)
        engagements = np.random.poisson(clicks * 0.3)
        conversions = np.random.poisson(engagements * 0.05)
        
        engagement_data.append({
            'campaign_id': campaign,
            'channel': channel,
            'date': date.strftime('%Y-%m-%d'),
            'impressions': impressions,
            'clicks': clicks,
            'ctr': clicks / (impressions + 1) * 100,
            'engagements': engagements,
            'engagement_rate': engagements / (clicks + 1) * 100,
            'conversions': conversions,
            'cvr': conversions / (engagements + 1) * 100,
            'cost_per_click': np.random.uniform(0.5, 3.0),
            'cost_per_conversion': np.random.uniform(20, 80)
        })
    
    return pd.DataFrame(conversion_data), pd.DataFrame(engagement_data)

def create_vectors(client: QdrantClient, df: pd.DataFrame, collection_name: str, embeddings):
    """Convert DataFrame to text documents and store in Qdrant"""
    documents = []
    for _, row in df.iterrows():
        text = f"Campaign: {row['campaign_id']}, "
        text += f"Date: {row.get('date', 'N/A')}, "
        
        if collection_name == "conversion_data":
            text += (f"Product: {row['product']}, Target Conversions: {row['target_conversions']}, "
                    f"Control: {row['control_conversions']}, Test: {row['test_conversions']}, "
                    f"Incremental Lift: {row['incremental_lift']}, Sales: ${row['sales_revenue']:,.0f}, "
                    f"Conversion Rate: {row['conversion_rate']:.1f}%, ROI: {row['roi']:.2f}")
        else:
            text += (f"Channel: {row['channel']}, Impressions: {row['impressions']:,}, "
                    f"Clicks: {row['clicks']:,}, CTR: {row['ctr']:.2f}%, "
                    f"Engagements: {row['engagements']:,}, Engagement Rate: {row['engagement_rate']:.1f}%, "
                    f"Conversions: {row['conversions']}, CVR: {row['cvr']:.1f}%, "
                    f"Cost/Click: ${row['cost_per_click']:.2f}, Cost/Conversion: ${row['cost_per_conversion']:.0f}")
        
        documents.append(text)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.create_documents([doc for doc in documents])
    
    points = []
    for i, doc in enumerate(splits):
        embedding = embeddings.embed_query(doc.page_content)
        points.append(PointStruct(
            id=i,
            vector=embedding,
            payload={"text": doc.page_content, "table": collection_name}
        ))
    
    client.upsert(collection_name=collection_name, points=points)
    return len(points)

def get_model_components(model_provider: str):
    """Get embeddings and LLM based on selected provider"""
    if model_provider == "OpenAI":
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            st.error("❌ OpenAI API key not found in secrets!")
            st.stop()
        
        return (
            OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key),
            ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key),
            1536
        )
    else:  # Gemini
        api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        if not api_key:
            st.error("❌ Gemini API key not found in secrets!")
            st.stop()
        
        return (
            GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key),
            ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key),
            768
        )

def setup_rag_system(model_provider: str):
    """Initialize complete RAG system with selected model"""
    embeddings, llm, embed_dim = get_model_components(model_provider)
    client = init_qdrant()
    
    if client.get_collection("conversion_data").points_count == 0:
        conv_df, eng_df = generate_sample_data()
        st.session_state.conv_df = conv_df
        st.session_state.eng_df = eng_df
        
        create_vectors(client, conv_df, "conversion_data", embeddings)
        create_vectors(client, eng_df, "engagement_data", embeddings)
        st.success(f"✅ {model_provider} RAG System Ready!")
    
    return client, embeddings, llm

def format_docs(docs):
    return "\n\n".join(doc.payload["text"] for doc in docs)

@st.cache_resource
def create_rag_chain(client, embeddings, llm, _model_provider):
    def retrieve_docs(query, k=5):
        search_results = []
        for collection in ["conversion_data", "engagement_data"]:
            hits = client.search(
                collection_name=collection,
                query_vector=embeddings.embed_query(query),
                limit=k,
                query_filter=None
            )
            search_results.extend(hits)
        return search_results[:k]
    
    def rag_chain(query):
        docs = retrieve_docs(query)
        context = format_docs(docs)
        
        prompt = ChatPromptTemplate.from_template(
            """You are a marketing analytics expert. Use only the provided campaign data context to answer.

Context from {table}:
{context}

Question: {question}

Answer with specific numbers, trends, and insights from the data."""
        )
        
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "table": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain.invoke({"context": context, "question": query, "table": "conversion & engagement tables"})
    
    return rag_chain

# Streamlit UI
def main():
    st.title("🤖 Marketing RAG Agent - Multi-Model")
    st.markdown("**Toggle between OpenAI GPT-4o-mini & Gemini 1.5 Flash**")
    
    # Sidebar - Model Selection
    with st.sidebar:
        st.header("🎯 Model Selection")
        
        model_options = ["OpenAI (GPT-4o-mini)", "Gemini 1.5 Flash"]
        selected_model = st.selectbox("Choose LLM:", model_options, index=1)
        model_provider = "OpenAI" if "OpenAI" in selected_model else "Gemini"
        
        st.info(f"**Selected:** {selected_model}")
        
        if st.button("🔄 Rebuild Index", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        # Setup system
        if 'system_setup' not in st.session_state or st.session_state.get('last_model') != model_provider:
            with st.spinner(f"Setting up {model_provider} RAG..."):
                st.session_state.client, st.session_state.embeddings, st.session_state.llm = setup_rag_system(model_provider)
                st.session_state.system_setup = True
                st.session_state.last_model = model_provider
        
        # Data preview
        if 'conv_df' in st.session_state:
            st.subheader("📋 Sample Data")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(st.session_state.conv_df.head(3), use_container_width=True)
            with col2:
                st.dataframe(st.session_state.eng_df.head(3), use_container_width=True)
    
    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about campaigns, ROI, CTR, channel performance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner(f"{'🤖' if model_provider=='OpenAI' else '⭐'} {model_provider} analyzing..."):
                rag_chain = create_rag_chain(
                    st.session_state.client, 
                    st.session_state.embeddings, 
                    st.session_state.llm,
                    model_provider
                )
                response = rag_chain(prompt)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
