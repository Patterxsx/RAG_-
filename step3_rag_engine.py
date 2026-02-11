import os
import torch
from typing import List, Dict
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

class HongLouRAG:
    def __init__(self):
        print("🎋 正在初始化红楼RAG系统...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        print("📚 加载知识库...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = FAISS.load_local(
            "faiss_index/honglou_index",
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ 已加载 {self.vectorstore.index.ntotal} 条知识")
    
        print("🧠 加载Qwen2.5-7B（4-bit量化，约4.5GB显存）...")
        self._load_llm()
        
        self._init_prompts()
        
        print("✨ 系统就绪！显存占用:", self._get_gpu_memory(), "MB")
    
    def _load_llm(self):
        model_name = "qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
        
        print(f"  正在下载/加载 {model_name}...")
        model_dir = snapshot_download(model_name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            pad_token='<|im_end|>',
            padding_side='left'
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",         
            trust_remote_code=True,
            torch_dtype=torch.float16,   
        )
        
        self.llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,         
            temperature=0.3,           
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
            return_full_text=False       
        )
        print(" 模型加载完成")
    
    def _init_prompts(self):
        """初始化红学专家Prompt"""
        
        self.rag_template = """你是一位精通《红楼梦》的资深红学研究者，擅长文本细读与脂砚斋评注解读。

【任务】基于以下检索到的原文与评注，回答用户的问题。回答要求：
1. 必须引用具体回目（如"见于第三回"）
2. 结合脂砚斋评注（如有）分析深层含义
3. 指出艺术手法（草蛇灰线、春秋笔法等）

【检索资料】
{context}

【用户问题】
{question}

【红学分析】"""
    
    def _get_gpu_memory(self):

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return 0
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """检索相关段落"""
        print(f"🔍 检索: '{query}'...")
        docs = self.vectorstore.similarity_search(query, k=k)
        
        seen_chapters = set()
        unique_docs = []
        for doc in docs:
            ch = doc.metadata.get('chapter', 0)
            if ch not in seen_chapters:
                seen_chapters.add(ch)
                unique_docs.append(doc)
        
        return unique_docs[:3]  # 取前3个不同回目
    
    def generate(self, query: str) -> Dict:
        """生成回答"""
        docs = self.retrieve(query)

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('type', '正文')
            chapter = doc.metadata.get('chapter_title', 'unknown')
            content = doc.page_content[:300] 
            
            context_parts.append(
                f"[{i}] {chapter}（{source}）：{content}..."
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = self.rag_template.format(
            context=context,
            question=query
        )
        
        print("📝 生成回答中...")
        try:
            response = self.llm(prompt)[0]['generated_text']
        
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return {
                "query": query,
                "answer": response,
                "sources": docs,
                "context": context
            }
        except Exception as e:
            print(f"生成错误: {e}")
            return {"error": str(e)}
    
    def chat(self):
        """交互式对话"""
        print("\n" + "="*60)
        print("红楼RAG")
        print("="*60)
        print("输入问题，输入'quit'退出")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("您的问题：").strip()
                if not query:
                    continue
                if query.lower() in ['quit', 'exit', '退出']:
                    print("再见！")
                    break
                result = self.generate(query)
                if "error" in result:
                    print(f"错误: {result['error']}")
                    continue
                print("\n" + "-"*60)
                print("📖 引用资料：")
                for doc in result['sources']:
                    ch = doc.metadata.get('chapter_title', '未知')
                    t = doc.metadata.get('type', '正文')
                    print(f"   • {ch}（{t}）")
                
                print("\n💬 专家解读：")
                print(result['answer'])
                print("-"*60 + "\n")
            
                if self.device == "cuda":
                    print(f"[显存占用: {self._get_gpu_memory():.1f}MB]")
                
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                print(f"错误: {e}")

def main():
    rag = HongLouRAG()
    rag.chat()

if __name__ == "__main__":
    main()