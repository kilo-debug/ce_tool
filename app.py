import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import os

# FREE Local Models (No API keys!)
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

st.set_page_config(page_title="FREE Marketing RAG Agent", page_icon="🚀", layout="wide")

@st.cache_resource
def load_free_models():
    """Load FREE local models - 100% offline capable"""
    # FREE Embedding model (384 dim, very fast)
    embeddings = SentenceTransformer('all-MiniLM-L6-v2')
    
    # FREE Chat model (works great for RAG)
    generator = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-medium",  # 345M params, fast
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    return embeddings, generator

@st.cache_resource
def init_qdrant():
    """Qdrant with free model dimensions"""
    client = QdrantClient(":memory:")
    client.recreate_collection(
        "conversion_data",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    client.recreate_collection(
        "engagement_data", 
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    return client

@st.cache_data
def generate_sample_data():
    """150+ conversion + 120+ engagement records"""
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    campaigns = [f'Campaign_{i:03d}' for i in range(1, 31)]
    
    # Conversion data
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    conversion_data = []
    for i in range(150):
        campaign = np.random.choice(campaigns)
        product = np.random.choice(products)
        date = np.random.choice(dates)
        
        target = int(np.random.poisson(50))
        control = int(np.random.poisson(40))
        test = target + int(np.random.poisson(10))
        lift = test - control
        sales = float(np.random.poisson(1000) * (1 + lift/100))
        
        conversion_data.append({
            'campaign_id': campaign, 'product': product, 'date': date.strftime('%Y-%m-%d'),
            'target_conversions': target, 'control_conversions': control, 
            'test_conversions': test, 'incremental_lift': lift,
            'sales_revenue': sales, 'conversion_rate': round(test/max(target,1)*100,2),
            'roi': round(sales/max(target*10,1),2)
        })
    
    # Engagement data
    channels = ['Email', 'FB', 'Instagram', 'LinkedIn', 'Display', 'Search', 'Video', 'SMS']
    engagement_data = []
    for i in range(120):
        campaign = np.random.choice(campaigns)
        channel = np.random.choice(channels)
        date = np.random.choice(dates)
        
        impressions = int(np.random.poisson(50000))
        clicks = int(np.random.poisson(impressions*0.02))
        engagements = int(np.random.poisson(clicks*0.3))
        conversions = int(np.random.poisson(engagements*0.05))
        
        engagement_data.append({
            'campaign_id': campaign, 'channel': channel, 'date': date.strftime('%Y-%m-%d'),
            'impressions': impressions, 'clicks': clicks,
            'ctr': round(clicks/max(impressions,1)*100,2),
            'engagements': engagements, 'engagement_rate': round(engagements/max(clicks,1)*100,2),
            'conversions': conversions, 'cvr': round(conversions/max(engagements,1)*100,2)
        })
    
    return pd.DataFrame(conversion_data), pd.DataFrame(engagement_data)

def index_data(client, embeddings, conv_df, eng_df):
    """Index 270+ documents"""
    if client.get_collection("conversion_data").points_count > 0:
        return
    
    # Conversion docs
    conv_docs = []
    for _, row in conv_df.iterrows():
        doc = f"Campaign: {row['campaign_id']} | Product: {row['product']} | Date: {row['date']} | Target: {row['target_conversions']} | Control: {row['control_conversions']} | Test: {row['test_conversions']} | Lift: {row['incremental_lift']} | Sales: ${row['sales_revenue']:,.0f} | ROI: {row['roi']}"
        conv_docs.append(doc)
    
    # Engagement docs  
    eng_docs = []
    for _, row in eng_df.iterrows():
        doc = f"Campaign: {row['campaign_id']} | Channel: {row['channel']} | Date: {row['date']} | Impressions: {row['impressions']:,} | Clicks: {row['clicks']:,} | CTR: {row['ctr']}% | Conversions: {row['conversions']} | CVR: {row['cvr']}%"
        eng_docs.append(doc)
    
    # Embed and store
    all_docs = conv_docs + eng_docs
    embeddings_list = embeddings.encode(all_docs)
    
    # Conversion points
    conv_points = [PointStruct(id=i, vector=embeddings_list[i], payload={"text": conv_docs[i], "type": "conversion"}) 
                   for i in range(len(conv_docs))]
    client.upsert("conversion_data", conv_points)
    
    # Engagement points
    eng_points = [PointStruct(id=i+1000, vector=embeddings_list[len(conv_docs)+i], payload={"text": eng_docs[i], "type": "engagement"}) 
                  for i in range(len(eng_docs))]
    client.upsert("engagement_data", eng_points)

@st.cache_resource
def setup_rag(_dummy):
    """Complete FREE RAG setup"""
    embeddings, generator = load_free_models()
    client = init_qdrant()
    
    conv_df, eng_df = generate_sample_data()
    st.session_state.conv_df = conv_df
    st.session_state.eng_df = eng_df
    
    index_data(client, embeddings, conv_df, eng_df)
    return client, embeddings, generator, conv_df, eng_df

def rag_query(client, embeddings, generator, query, conv_df, eng_df):
    """FREE RAG pipeline"""
    query_emb = embeddings.encode([query])[0]
    
    # Retrieve
    conv_results = client.search("conversion_data", query_vector=query_emb, limit=3)
    eng_results = client.search("engagement_data", query_vector=query_emb, limit=3)
    
    context = "\n\n".join([r.payload["text"] for r in conv_results + eng_results])
    
    # Generate (FREE local model)
    prompt = f"""Marketing Data:
{context}

Question: {query}

Answer using only the data above with specific numbers:"""
    
    response = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.1, pad_token_id=generator.tokenizer.eos_token_id)
    return response[0]['generated_text'].split("Answer using only the data above with specific numbers:")[-1].strip()

def main():
    st.title("🚀 100% FREE Marketing RAG Agent")
    st.markdown("**No API keys! Local models only | 270+ marketing records**")
    
    # Setup
    client, embeddings, generator, conv_df, eng_df = setup_rag("free")
    
    # Data preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Conversion Sample")
        st.dataframe(conv_df[['campaign_id','incremental_lift','roi']].head())
    with col2:
        st.subheader("Engagement Sample") 
        st.dataframe(eng_df[['campaign_id','channel','ctr','cvr']].head())
    
    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask: 'Best ROI campaign?' 'Top CTR channel?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 FREE model analyzing..."):
                response = rag_query(client, embeddings, generator, prompt, conv_df, eng_df)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
