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

st.set_page_config(page_title="Marketing RAG Agent", page_icon="🤖", layout="wide")

@st.cache_resource
def init_qdrant(embed_dim: int):
    client = QdrantClient(":memory:")
    try:
        client.delete_collection("conversion_data")
        client.delete_collection("engagement_data")
    except:
        pass
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
    """FIXED: Proper datetime handling"""
    np.random.seed(42)
    
    # Create dates as Python datetime objects
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    campaigns = [f'Campaign_{i:03d}' for i in range(1, 31)]
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    
    conversion_data = []
    for i in range(150):
        campaign = np.random.choice(campaigns)
        product = np.random.choice(products)
        date = np.random.choice(dates)
        
        target_conversions = int(np.random.poisson(50))
        control_conversions = int(np.random.poisson(40))
        test_conversions = target_conversions + int(np.random.poisson(10))
        incremental = test_conversions - control_conversions
        sales = float(np.random.poisson(1000) * (1 + incremental/100.0))
        
        conversion_data.append({
            'campaign_id': campaign,
            'product': product,
            'date': date.strftime('%Y-%m-%d'),  # Now safe!
            'target_conversions': target_conversions,
            'control_conversions': control_conversions,
            'test_conversions': test_conversions,
            'incremental_lift': incremental,
            'sales_revenue': sales,
            'conversion_rate': round(test_conversions / max(target_conversions, 1) * 100, 2),
            'roi': round(sales / max(target_conversions * 10, 1), 2)
        })
    
    channels = ['Email', 'Social_Facebook', 'Social_Instagram', 'Social_LinkedIn', 'Display', 'Search', 'Video', 'SMS']
    engagement_data = []
    for i in range(120):
        campaign = np.random.choice(campaigns)
        channel = np.random.choice(channels)
        date = np.random.choice(dates)
        
        impressions = int(np.random.poisson(50000))
        clicks = int(np.random.poisson(impressions * 0.02))
        engagements = int(np.random.poisson(clicks * 0.3))
        conversions = int(np.random.poisson(engagements * 0.05))
        
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

def get_model_components(model_provider):
    """Safe model initialization"""
    try:
        if model_provider == "OpenAI":
            return (
                OpenAIEmbeddings(model="text-embedding-3-small"),
                ChatOpenAI(model="gpt-4o-mini", temperature=0),
                1536
            )
        else:
            return (
                GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0),
                768
            )
    except Exception as e:
        st.error(f"Model error: {str(e)}")
        st.stop()

@st.cache_resource
def setup_rag_system(_model_provider):
    model_provider = _model_provider
    embeddings, llm, embed_dim = get_model_components(model_provider)
    client = init_qdrant(embed_dim)
    
    # Index data if empty
    if client.get_collection("conversion_data").points_count == 0:
        conv_df, eng_df = generate_sample_data()
        st.session_state.conv_df = conv_df
        st.session_state.eng_df = eng_df
        
        # Index conversion data
        docs_conv = []
        for _, row in conv_df.iterrows():
            text = (f"Campaign {row['campaign_id']} | {row['product']} | "
                   f"{row['date']} | Target:{row['target_conversions']} "
                   f"Control:{row['control_conversions']} Test:{row['test_conversions']} "
                   f"Lift:{row['incremental_lift']} Sales:${row['sales_revenue']:,.0f} "
                   f"Rate:{row['conversion_rate']}% ROI:{row['roi']}")
            docs_conv.append(text)
        
        # Simple embedding and upsert
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.create_documents(docs_conv)
        points = []
        for i, doc in enumerate(splits):
            embedding = embeddings.embed_query(doc.page_content)
            points.append(PointStruct(id=i, vector=embedding, payload={"text": doc.page_content}))
        
        client.upsert("conversion_data", points)
        
        # Index engagement data (simplified)
        docs_eng = []
        for _, row in eng_df.iterrows():
            text = (f"Campaign {row['campaign_id']} | {row['channel']} | "
                   f"{row['date']} | Impressions:{row['impressions']:,} "
                   f"Clicks:{row['clicks']:,} CTR:{row['ctr']}% "
                   f"Engagements:{row['engagements']:,} CVR:{row['cvr']}%")
            docs_eng.append(text)
        
        splits_eng = splitter.create_documents(docs_eng)
        points_eng = []
        for i, doc in enumerate(splits_eng):
            embedding = embeddings.embed_query(doc.page_content)
            points_eng.append(PointStruct(id=i+1000, vector=embedding, payload={"text": doc.page_content}))
        
        client.upsert("engagement_data", points_eng)
        
        st.success("✅ Data indexed!")
    
    return client, embeddings, llm

def main():
    st.title("🤖 Marketing RAG Agent")
    st.markdown("**Fixed! Toggle OpenAI/Gemini | 270+ records**")
    
    # Sidebar
    with st.sidebar:
        model_provider = st.radio("Choose Model:", ["Gemini 1.5 Flash", "OpenAI GPT-4o-mini"], index=0)
        
        if st.button("🔄 Reset & Reindex", type="primary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        st.info(f"**Active:** {model_provider}")
    
    # Setup RAG
    client, embeddings, llm = setup_rag_system(model_provider)
    
    # Data preview
    if 'conv_df' in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Conversion Data")
            st.dataframe(st.session_state.conv_df.head(3))
        with col2:
            st.subheader("Engagement Data")
            st.dataframe(st.session_state.eng_df.head(3))
    
    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about campaigns, ROI, CTR, best channels..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 Analyzing..."):
                # Simple RAG retrieval
                query_emb = embeddings.embed_query(prompt)
                
                conv_hits = client.search("conversion_data", query_vector=query_emb, limit=3)
                eng_hits = client.search("engagement_data", query_vector=query_emb, limit=3)
                
                context = "\n".join([h.payload["text"] for h in conv_hits + eng_hits])
                
                prompt_template = ChatPromptTemplate.from_template("""
Based on this marketing data, answer: {question}

DATA:
{context}

Answer with specific metrics and insights:
                """)
                
                chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = chain.invoke({"context": context, "question": prompt})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
