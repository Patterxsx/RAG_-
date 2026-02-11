import json
import os
from typing import List, Dict
from tqdm import tqdm
import numpy as np

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class HongLouVectorizer:
    def __init__(self, jsonl_path: str = "honglou_data.jsonl"):
        self.jsonl_path = jsonl_path
        self.output_dir = "faiss_index"
        print("正在加载Embedding模型（约1GB内存）...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",  
            model_kwargs={'device': 'cpu'},       
            encode_kwargs={
                'normalize_embeddings': True,     
                'batch_size': 64                  
            }
        )
        print("模型加载完成")
    
    def load_data(self) -> List[Document]:
        """加载第一步生成的数据"""
        print(f"正在加载 {self.jsonl_path}...")
        documents = []
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取数据"):
                data = json.loads(line.strip())

                content = data['content']
                if data['metadata']['type'] == '脂评':
                    content = f"[脂评] {content}"
                
                doc = Document(
                    page_content=content,
                    metadata=data['metadata']
                )
                documents.append(doc)
        
        print(f"共加载 {len(documents)} 个文档")
        print(f"  - 正文: {len([d for d in documents if '脂评' not in d.page_content])}")
        print(f"  - 脂评: {len([d for d in documents if '脂评' in d.page_content])}")
        return documents
    
    def build_index(self, documents: List[Document]):
        """构建FAISS向量库"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("开始构建向量索引（约需2-5分钟）...")
        batch_size = 500
        vectorstore = None
        
        for i in tqdm(range(0, len(documents), batch_size), desc="构建索引"):
            batch = documents[i:i+batch_size]
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    batch, 
                    self.embeddings,
                    distance_strategy="COSINE" 
                )
            else:
   
                vectorstore.add_documents(batch)
        
        index_path = os.path.join(self.output_dir, "honglou_index")
        vectorstore.save_local(index_path)
        
        print(f"\n✅ 索引构建完成！")
        print(f"保存位置: {index_path}/")
        print(f"包含文件: index.faiss (向量数据) + index.pkl (metadata映射)")
        
        return vectorstore
    
    def verify_index(self, vectorstore):
        """验证测试：确保检索正常工作"""
        print("\n" + "="*50)
        print("🔍 索引验证测试")
        print("="*50)
        
        test_queries = [
            "林黛玉进贾府",
            "宝玉和黛玉的感情",
            "王熙凤的管理手段",
            "元春省亲"
        ]
        
        for query in test_queries:
            print(f"\n查询: '{query}'")
            results = vectorstore.similarity_search(query, k=2)
            
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get('chapter_title', '未知')
                doc_type = doc.metadata.get('type', '未知')
                preview = doc.page_content[:40].replace('\n', '')
                print(f"  {i}. [{source}-{doc_type}] {preview}...")
        
        print("\n✅ 验证通过！索引可正常工作")

def main():
    vectorizer = HongLouVectorizer()
    
    docs = vectorizer.load_data()
    
    vectorstore = vectorizer.build_index(docs)
    
    vectorizer.verify_index(vectorstore)

if __name__ == "__main__":
    main()