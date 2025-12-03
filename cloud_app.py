import streamlit as st
import os
import sys

# --- 导入核心库 ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
except Exception as e:
    st.error(f"环境加载失败: {e}")
    st.stop()

# --- 页面设置 ---
st.set_page_config(page_title="云端知识库", page_icon="☁️")
st.title("❤ 小茹专属科一知识库问答助手")

# --- 核心配置 ---
DATA_FOLDER = "knowledge"  # 你的 PDF 文件夹名字

# --- 获取 API Key (优先从云端通过 Secrets 获取，如果没有则让用户填) ---
# 这样你可以把 Key 藏在后台，客户不用填，也看不到
if "DEEPSEEK_API_KEY" in st.secrets:
    api_key = st.secrets["DEEPSEEK_API_KEY"]
else:
    api_key = st.sidebar.text_input("请输入 DeepSeek API Key", type="password")

# --- 侧边栏展示 ---
with st.sidebar:
    st.markdown("### 📚 已加载文档")
    if os.path.exists(DATA_FOLDER):
        files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
        for f in files:
            st.text(f"📄 {f}")
    else:
        st.error(f"未找到 {DATA_FOLDER} 文件夹")

# --- 核心函数 ---
@st.cache_resource
def load_knowledge_base():
    # 1. 扫描文件
    if not os.path.exists(DATA_FOLDER):
        return None
    
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    if not files:
        return None
    
    all_documents = []
    print(f"正在加载 {len(files)} 个文件...")
    
    # 2. 加载
    for filename in files:
        file_path = os.path.join(DATA_FOLDER, filename)
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"加载失败: {filename}")

    # 3. 切分
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(all_documents)

    # 4. 向量化 (云端会自动下载模型，速度很快)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 5. 存入 FAISS
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore

# --- 主逻辑 ---
if not api_key:
    st.warning("请在侧边栏输入 Key，或在 Secrets 中配置。")
    st.stop()

# 加载知识库
with st.spinner("正在启动云端引擎..."):
    vectorstore = load_knowledge_base()

if vectorstore:
    # 准备问答链
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    
    st.success("✅ 系统就绪！")
    
    # 聊天界面
    question = st.text_input("请输入你的问题：", placeholder="关于这些文档，你想问什么？")
    
    if question:
        with st.spinner("AI 正在思考..."):
            result = qa_chain.invoke({"query": question})
            st.write("### 🤖 回答：")
            st.info(result['result'])
else:
    st.error("知识库为空，请检查 GitHub 仓库中是否上传了 PDF 文件。")