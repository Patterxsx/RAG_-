# -*- coding: utf-8 -*-
import os
import re
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

def extract_honglou(tex_folder: str, output_file: str = "honglou_data.jsonl"):
    """
    从LaTeX脂评本中提取结构化数据
    """
    tex_path = Path(tex_folder)
    tex_files = sorted(tex_path.glob("*.tex"))
    print(f"✅ 发现 {len(tex_files)} 个文件，开始处理...")
    all_documents = []
    for tex_file in tqdm(tex_files, desc="解析文件"):
        docs = parse_single_file(tex_file)
        all_documents.extend(docs)
    with open(output_file, 'w', encoding='utf-8') as f:       # 保存为JSONL（RAG标准格式）
        for doc in all_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    generate_report(all_documents, output_file)

def parse_single_file(file_path: Path) -> List[Dict]:
    """解析单个tex文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chapter_info = extract_chapter_info(content, file_path.name)
    raw_main_text, comments = split_content_and_comments(content)      # 分离正文和脂评
    main_text = clean_latex_content(raw_main_text)
    documents = []
    #  处理正文：按段落切分
    paragraphs = split_paragraphs(main_text)
    for idx, para in enumerate(paragraphs):
        if len(para.strip()) < 20:  # 过滤太短的
            continue
        documents.append({
            "id": f"{chapter_info['num']:03d}-{idx:03d}",
            "content": para.strip(),
            "metadata": {
                "chapter": chapter_info['num'],
                "chapter_title": chapter_info['title'],
                "type": "正文",
                "source_file": file_path.name
            }
        })
    #  处理脂评
    for idx, comment in enumerate(comments):
        if len(comment.strip()) < 5:
            continue
        documents.append({
            "id": f"{chapter_info['num']:03d}-z-{idx:03d}",
            "content": comment.strip(),
            "metadata": {
                "chapter": chapter_info['num'],
                "chapter_title": chapter_info['title'],
                "type": "脂评",
                "source_file": file_path.name
            }
        })
    
    return documents

def extract_chapter_info(content: str, filename: str) -> Dict:
    """提取第X回信息"""
    # 从文件名提取
    num_match = re.search(r'chapter(\d+)', filename)
    if num_match:
        num = int(num_match.group(1))
        lines = content.split('\n')[:10]
        for line in lines:
            if '第' in line and '回' in line:
                return {"num": num, "title": line.strip()}
        return {"num": num, "title": f"第{num}回"}
    
    return {"num": 0, "title": "未知回目"}

def clean_latex_content(text: str) -> str:
    """清理LaTeX标记，保留文本内容"""
    # 去除文档类定义等头部
    text = re.sub(r'\\documentclass[^}]*\}', '', text)
    text = re.sub(r'\\usepackage[^}]*\}', '', text)
    text = re.sub(r'\\begin\{document\}', '', text)
    text = re.sub(r'\\end\{document\}', '', text)
    # 去除注释（%开头）
    text = re.sub(r'^\s*%.*$', '', text, flags=re.MULTILINE)
    # 提取命令参数：\command{arg} -> arg
    # 重复多次处理嵌套
    for _ in range(3):
        text = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+\*?\[([^\]]*)\]', r'\1', text)
    # 去除剩余命令
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # 清理特殊符号
    text = text.replace('\\', '')
    text = text.replace('&', '')
    text = text.replace('#', '')
    text = text.replace('_', '')
    # 规范化空白
    text = re.sub(r'\s+', '\n', text)
    text = re.sub(r'\.5em', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # 清理残留的 includegraphics 参数块：{width=3mm{../Images/00004}
    text = re.sub(r'\{width=3mm(?:\{[^}]+\})+\}?', '', text)
    # 清理残留的 Images 路径（可能单独残留）
    text = re.sub(r'\{[^}]*Images/[^}]+\}\}?', '', text)
    # 清理残留的数学公式标记（如样本中的{$$}）
    text = re.sub(r'\{\$\$?\}', '', text)
    text = re.sub(r'width=3mm', '', text)
        # 清理夹批删除后残留的孤立 }（通常在句末或段末）
    text = re.sub(r'([。，！？；：、"”’])\s*\}\s*', r'\1', text)  # 标点后的 }
    text = re.sub(r'\s*\}\s*([。，！？；：、"”’])', r'\1', text)  # 标点前的 }
    text = re.sub(r'^\s*\}\s*', '', text, flags=re.MULTILINE)     # 行首的 }
    
    # 清理LaTeX方括号标记：{[}之{]} -> 之（保留内容，删除标记）
    text = re.sub(r'\{\[\}', '', text)  # 删除 {[
    text = re.sub(r'\{\]\}', '', text)  # 删除 ]}
    
    # 清理连续多余的空格
    text = re.sub(r' +', ' ', text)
    return text.strip()

def split_content_and_comments(text: str) -> tuple:
    """
    分离正文和脂评
    脂评常见格式：【批语内容】或(眉批：内容)或\footnote{内容}
    """
    comments = []
    
    footnote_pattern = r'\\footnote\{((?:[^{}]|\{[^{}]*\})*)\}'
    for match in re.findall(footnote_pattern, text):
        # 清理脚注内的LaTeX标记（如{[}之{]} -> 之）
        clean = re.sub(r'\{([^{}]*)\}', r'\1', match)
        clean = re.sub(r'\\[a-zA-Z]+', '', clean)  # 删除剩余命令
        comments.append(f"[回批] {clean.strip()}")
    text = re.sub(footnote_pattern, '', text)
    
    kaishu_pattern = r'\{\\includegraphics[^{}]*(?:\{[^}]*\})*\s*\\kaishu\s+((?:[^{}]|\{[^{}]*\})+?)\s*\}'
    for match in re.findall(kaishu_pattern, text):
        # 清理 {[}之{]} 或 {$\diamond$} 等
        clean = re.sub(r'\{([^{}]*)\}', r'\1', match)  # 去一层{}
        clean = re.sub(r'\\[a-zA-Z]+', '', clean)      # 去命令如\diamond
        clean = re.sub(r'\$\$\s*\$', '', clean)        # 去$$和$
        if len(clean.strip()) > 3:  # 过滤太短的
            comments.append(f"[夹批] {clean.strip()}")    # 删除整个 includegraphics+kaishu 组（避免污染正文）
    text = re.sub(r'\{\\includegraphics[^{}]*(?:\{[^}]*\})*\}', '', text)
    return text, comments

def split_paragraphs(text: str) -> List[str]:
    """智能分段：按空行分，但合并短行"""
    lines = text.split('\n')
    paragraphs = []
    current = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            if current:
                paragraphs.append(current)
                current = ""
        else:
            # 如果当前段落很长且以句号结束，先存起来
            if len(current) > 200 and current.endswith('。'):
                paragraphs.append(current)
                current = line
            else:
                current += line
    
    if current:
        paragraphs.append(current)
    
    return paragraphs

def chinese_to_number(chinese: str) -> int:
    """中文数字转阿拉伯数字（支持到千）"""
    num_map = {'一':1, '二':2, '三':3, '四':4, '五':5, 
               '六':6, '七':7, '八':8, '九':9, '十':10,
               '百':100, '千':1000, '零':0, '〇':0}
    
    result = 0
    temp = 0
    for char in chinese:
        if char in num_map:
            n = num_map[char]
            if n >= 10:
                if temp == 0:
                    temp = 1
                result += temp * n
                temp = 0
            else:
                temp = temp * 10 + n if temp > 0 else n
    result += temp
    return result if result > 0 else 0

def generate_report(docs: List[Dict], output_file: str):
    """生成提取报告"""
    total = len(docs)
    main_count = len([d for d in docs if d['metadata']['type'] == '正文'])
    comment_count = len([d for d in docs if d['metadata']['type'] == '脂评'])
    chapters = len(set(d['metadata']['chapter'] for d in docs))
    
    print(f"\n{'='*50}")
    print(f"✅ 提取完成！")
    print(f"{'='*50}")
    print(f"📄 输出文件：{output_file}")
    print(f"📚 覆盖回目：{chapters} 回")
    print(f"📝 正文段落：{main_count} 个")
    print(f"💬 脂评条目：{comment_count} 条")
    print(f"📊 总计文档：{total} 条")
    print(f"{'='*50}")
    
    # 显示样本
    print(f"\n📝 数据样本：")
    for i, doc in enumerate(docs[:3]):
        print(f"\n{i+1}. [{doc['metadata']['type']}] {doc['metadata']['chapter_title']}")
        print(f"   内容：{doc['content'][:-1]}...")

if __name__ == "__main__":
    # 交互式输入
    folder = input("📂 请输入LaTeX文件所在文件夹路径：").strip().strip('"').strip("'")
    output = input("💾 输出文件名（默认：honglou_data.jsonl）：").strip()
    if not output:
        output = "honglou_data.jsonl"
    
    extract_honglou(folder, output)
