import streamlit as st
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from step3_rag_engine import HongLouRAG
import re
st.set_page_config(
    page_title="红楼RAG",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS（古风）
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;700&display=swap');
    
    .main {
        background-color: #f7f3e9;
        color: #2c241b;
        font-family: 'Noto Serif SC', serif;
    }
    
    .stTextInput>div>div>input {
        background-color: #fffef8;
        border: 2px solid #8b4513;
        border-radius: 8px;
        font-size: 16px;
    }
    
    .chat-bubble-user {
        background-color: #e8d5b5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #8b4513;
    }
    
    .chat-bubble-ai {
        background-color: #fffef8;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2c5f2d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.8;
    }
    
    .source-tag {
        display: inline-block;
        background-color: #d4c4a8;
        color: #5c4033;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
    }
    
    .chapter-title {
        color: #8b4513;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stSpinner > div {
        border-color: #8b4513;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag():
    """缓存RAG实例（避免重复加载模型）"""
    return HongLouRAG()

def clean_chapter_title(title):
    """清理回目显示（修复%{{}}问题）"""
    match = re.search(r'第[一二三四五六七八九十百千]+回', title)
    if match:
        return match.group(0)
    return title.replace('%{{', '').replace('}}', '').replace('}{', '')

def clean_answer(raw_answer):
    """
    清理模型输出（解决重复生成问题）
    """
    if not raw_answer:
        return raw_answer
    
    # 策略1：截断第一个【结束】之后的内容（防止重复循环）
    if '【结束】' in raw_answer:
        raw_answer = raw_answer[:raw_answer.index('【结束】')].strip()
    
    # 策略2：如果包含多个【回答】，只取最后一个
    if '【回答】' in raw_answer:
        parts = raw_answer.split('【回答】')
        if len(parts) > 1:
            raw_answer = parts[-1].strip()
    
    # 策略3：删除所有结构标记
    markers = ['【回答】', '【结束】', '专家解读：', '【任务】', '【检索资料】', '【用户问题】', '【红学分析】']
    for marker in markers:
        raw_answer = raw_answer.replace(marker, '').strip()
    
    # 策略4：如果回答以"以上回答"结尾（模型自我总结），删除
    if raw_answer.endswith('以上'):
        raw_answer = raw_answer[:-2].strip()
    
    # 策略5：删除"请随时告知"、"如有其他问题"等套话
    closing_phrases = [
        '如有其他问题或需要进一步探讨，请随时告知。',
        '如需进一步探讨，请随时告知。',
        '希望您满意。',
        '希望对您有所帮助。',
        '以上回答严格遵循了您的要求。',
    ]
    for phrase in closing_phrases:
        raw_answer = raw_answer.replace(phrase, '').strip()
    
    # 清理多余空行
    raw_answer = re.sub(r'\n\s*\n+', '\n\n', raw_answer)
    
    return raw_answer.strip()

def main():
    # 初始化
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'rag' not in st.session_state:
        with st.spinner("加载模型ing..."):
            st.session_state.rag = get_rag()
        st.success("Ready！")
    
    # 标题区
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #8b4513; font-size: 2.5em; margin-bottom: 0;"> 红楼夜话</h1>
            <p style="color: #666; font-style: italic; margin-top: 10px;">
                基于《红楼梦》脂评本与大语言模型的智能解读系统
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### 快捷导航")
        
        characters = ["贾宝玉", "林黛玉", "薛宝钗", "王熙凤", "史湘云", "贾探春"]
        selected = st.selectbox("选择人物直接询问", [""] + characters)
        if selected:
            st.session_state.quick_question = f"{selected}的人物性格分析"
            st.rerun()
        
        st.markdown("---")
        st.markdown("###  检索设置")
        k_value = st.slider("引用段落数", 2, 5, 3)
        
        st.markdown("---")
        st.markdown("### 系统状态")
        mem_usage = st.session_state.rag._get_gpu_memory()
        st.info(f"显存占用: {mem_usage:.1f}MB / 8192MB")
        if mem_usage > 7500:
            st.error(" 显存告警！请重启服务")
        elif mem_usage > 7000:
            st.warning("显存较高")
        else:
            st.success("运行正常")
        st.caption("本地部署 · 数据隐私安全")
    
    # 主界面：对话区
    st.markdown("---")
    
    # 显示历史对话
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="chat-bubble-user">
                <b> 问：</b>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            # 使用清理后的回答
            clean_content = clean_answer(msg['content'])
            
            st.markdown(f"""
            <div class="chat-bubble-ai">
                <b> 答：</b>{clean_content}
            </div>
            """, unsafe_allow_html=True)
            
            # 显示来源（可折叠）
            if msg.get('sources'):
                with st.expander("? 查看引用来源"):
                    for src in msg['sources']:
                        ch = clean_chapter_title(src['chapter'])
                        typ = src['type']
                        content = src['content'][:200]
                        st.markdown(f"""
                        <span class="source-tag">{ch}</span>
                        <span class="source-tag" style="background-color: #e8d5b5;">{typ}</span>
                        <div style="margin: 5px 0 15px 0; padding-left: 10px; border-left: 2px solid #ddd; color: #666; font-size: 0.9em;">
                            {content}...
                        </div>
                        """, unsafe_allow_html=True)
    
    # 输入区
    st.markdown("---")
    
    # 处理快捷问题
    default_value = st.session_state.get('quick_question', '')
    
    with st.form(key='query_form', clear_on_submit=True):
        cols = st.columns([4, 1])
        with cols[0]:
            user_input = st.text_input(
                "输入您的问题（如：分析黛玉葬花的象征意义）", 
                value=default_value,
                key="input",
                label_visibility="collapsed",
                placeholder="在此输入您关于《红楼梦》的疑问..."
            )
        with cols[1]:
            submit = st.form_submit_button(" 请教", use_container_width=True)
    
    if submit and user_input:
        # 清空快捷问题
        if 'quick_question' in st.session_state:
            del st.session_state.quick_question
            
        # 添加用户消息
        st.session_state.messages.append({
            'role': 'user', 
            'content': user_input
        })
        
        # 生成回答（带加载动画）
        with st.spinner("正在查阅典籍..."):
            try:
                result = st.session_state.rag.generate(user_input)
                
                # 关键：使用clean_answer清理重复内容
                raw_answer = result['answer']
                cleaned_answer = clean_answer(raw_answer)
                
                # 如果清理后内容太少（异常情况），使用原始内容
                if len(cleaned_answer) < 50 and len(raw_answer) > 100:
                    cleaned_answer = raw_answer[:500]  # 至少取前500字
                
                # 添加AI消息
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': cleaned_answer,  # 存储清理后的内容
                    'sources': [{
                        'chapter': d.metadata.get('chapter_title', '未知'),
                        'type': d.metadata.get('type', '正文'),
                        'content': d.page_content
                    } for d in result['sources']]
                })
                
                # 强制刷新显存显示
                st.rerun()
                
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': f"抱歉，查阅典籍时遇到阻碍：{str(e)}"
                })
    
    # 操作按钮
    col_clear, col_save = st.columns([1, 1])
    with col_clear:
        if st.button("清空对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_save:
        if st.button(" 导出记录", use_container_width=True) and st.session_state.messages:
            # 简单导出功能
            export_text = ""
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    export_text += f"问：{msg['content']}\n\n"
                else:
                    export_text += f"答：{clean_answer(msg['content'])}\n\n"
            st.download_button(
                label="下载TXT",
                data=export_text,
                file_name="红楼夜话记录.txt",
                mime="text/plain"
            )
    
    # 页脚
    st.markdown("---")
    st.caption("""
    <div style="text-align: center; color: #999; font-size: 0.8em;">
        基于 Qwen2.5-7B-Int4 + BGE-Embedding 构建 | 本地部署 | 无数据上传<br>
        引用脂评本及前八十回原文
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()