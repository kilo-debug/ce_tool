import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch

# FREE EMBEDDINGS
from sentence_transformers import SentenceTransformer

# PAID CHAT (toggle OpenAI/Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Vector DB + splitting
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Hybrid RAG Agent", page_icon="🔥", layout="wide")

@st.cache_resource
def load_embeddings():
    """FREE: all-MiniLM-L6-v2 (384 dim, super fast)"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def init_qdrant():
    """384 dim for FREE embeddings"""
    client = QdrantClient(":memory:")
    client.recreate_collection(
        "marketing_data",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    return client

@st.cache_data
def generate_sample_data():
    """270+ marketing records"""
    np.random.seed(42)
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    campaigns = [f'Campaign_{i:03d}' for i in range(1, 31)]
    
    # Conversion (150 records)
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E']
    conv_data = []
    for i in range(150):
        campaign = np.random.choice(campaigns)
        product = np.random.choice(products)
        date = np.random.choice(dates)
        target = int(np.random.poisson(50))
        control = int(np.random.poisson(40))
        test = target + int(np.random.poisson(10))
        lift = test - control
        sales = float(np.random.poisson(1000) * (1 + lift/100))
        
        conv_data.append({
            'type': 'conversion', 'campaign': campaign, 'product': product, 
            'date': date.strftime('%Y-%m-%d'), 'target_conversions': target,
            'control_conversions': control, 'test_conversions': test,
            'incremental_lift': lift, 'sales_revenue': sales,
            'conversion_rate': round(test/max(target,1)*100,2),
            'roi': round(sales/max(target*10,1),2)
        })
    
    # Engagement (120 records)
    channels = ['Email', 'Facebook', 'Instagram', 'LinkedIn', 'Display', 'Search', 'Video', 'SMS']
    eng_data = []
    for i in range(120):
        campaign = np.random.choice(campaigns)
        channel = np.random.choice(channels)
        date = np.random.choice(dates)
        impressions = int(np.random.poisson(50000))
        clicks = int(np.random.poisson(impressions*0.02))
        engagements = int(np.random.poisson(clicks*0.3))
        conversions = int(np.random.poisson(engagements*0.05))
        
        eng_data.append({
            'type': 'engagement', 'campaign': campaign, 'channel': channel,
            'date': date.strftime('%Y-%m-%d'), 'impressions': impressions,
            'clicks': clicks, 'ctr': round(clicks/max(impressions,1)*100,2),
            'engagements': engagements, 'conversions': conversions,
            'cvr': round(conversions/max(engagements,1)*100,2)
        })
    
    return pd.concat([pd.DataFrame(conv_data), pd.DataFrame(eng_data)], ignore_index=True)

def index_data(client, embeddings, df):
    """Index ALL data in single collection"""
    if client.get_collection("marketing_data").points_count > 0:
        return
    
    # Create documents
    documents = []
    for _, row in df.iterrows():
        if row['type'] == 'conversion':
            doc = (f"CONVERSION | Campaign: {row['campaign']} | Product: {row['product']} | "
                  f"Date: {row['date']} | Target: {row['target_conversions']} | "
                  f"Control: {row['control_conversions']} | Test: {row['test_conversions']} | "
                  f"Lift: {row['incremental_lift']} | Sales: ${row['sales_revenue']:,.0f} | "
                  f"Conv%: {row['conversion_rate']} | ROI: {row['roi']}")
        else:
            doc = (f"ENGAGEMENT | Campaign: {row['campaign']} | Channel: {row['channel']} | "
                  f"Date: {row['date']} | Impressions: {row['impressions']:,} | "
                  f"Clicks: {row['clicks']:,} | CTR: {row['ctr']}% | "
                  f"Conversions: {row['conversions']} | CVR: {row['cvr']}%")
        documents.append(doc)
    
    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    splits = splitter.create_documents(documents)
    
    points = []
    for i, doc in enumerate(splits):
        embedding = embeddings.encode(doc.page_content).tolist()
        points.append(PointStruct(
            id=i, 
            vector=embedding, 
            payload={"text": doc.page_content, "source": df.iloc[i%len(df)]['type']}
        ))
    
    client.upsert("marketing_data", points)
    st.success(f"✅ Indexed {len(points)} docs!")

def get_chat_model(model_provider):
    """Paid chat models"""
    if model_provider == "OpenAI":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    else:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

@st.cache_resource
def setup_rag_system(model_provider):
    """Hybrid setup: FREE embed + Paid chat"""
    embeddings = load_embeddings()
    client = init_qdrant()
    df = generate_sample_data()
    
    index_data(client, embeddings, df)
    st.session_state.df = df
    
    llm = get_chat_model(model_provider)
    return client, embeddings, llm, df

def main():
    st.title("🔥 Hybrid RAG: FREE Embed + Paid Chat")
    st.markdown("**all-MiniLM-L6-v2 (FREE) + GPT-4o/Gemini (toggle)**")
    
    # Model toggle
    with st.sidebar:
        model_options = ["Gemini 1.5 Flash", "OpenAI GPT-4o-mini"]
        model_provider = st.radio("Chat Model:", model_options, index=0)
        st.info(f"**Embeddings**: all-MiniLM-L6-v2 (FREE)")
        st.info(f"**Chat**: {model_provider}")
        
        if st.button("🔄 Reindex", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
    
    # Setup
    client, embeddings, llm, df = setup_rag_system(model_provider)
    
    # Data preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Conversion")
        st.dataframe(df[df['type']=='conversion'][['campaign','roi','incremental_lift']].head())
    with col2:
        st.subheader("Engagement") 
        st.dataframe(df[df['type']=='engagement'][['campaign','ctr','cvr']].head())
    
    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("💬 'Best ROI?' 'Channel comparison?'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner(f"Analyzing with {model_provider}..."):
                # Retrieve
                query_emb = embeddings.encode(prompt).tolist()
                results = client.search("marketing_data", query_vector=query_emb, limit=5)
                context = "\n\n".join([r.payload["text"] for r in results])
                
                # RAG Chain
                template = """You are a marketing expert. Use ONLY this data:

{context}

Q: {question}

Answer with specific metrics, comparisons, trends:"""
                
                chain = (
                    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                    | ChatPromptTemplate.from_template(template)
                    | llm
                    | StrOutputParser()
                )
                
                response = chain.invoke({"context": context, "question": prompt})
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
