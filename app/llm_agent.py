"""
LLM Agent - 自然语言理解和意图识别
"""
import os
import json
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import urllib.request
import urllib.error


class LLMAgent:
    """使用 LLM 理解用户意图并执行相应操作"""
    
    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai
        self.model = "llama3"
        self.ollama_host = "http://localhost:11434"
        self.ollama_url = f"{self.ollama_host}/api/generate"
        self._requests = None
        self.ollama_available = False
        
        if use_openai:
            # 使用 OpenAI API
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-3.5-turbo"
        else:
            # 使用本地 Ollama
            self._requests = self._safe_import_requests()
            if self._check_ollama_alive():
                if self._ensure_ollama_model():
                    self.ollama_available = True
                    print("✅ 连接到本地 Ollama")
                else:
                    self.ollama_url = None
            else:
                print("⚠️  未检测到 Ollama，将使用规则匹配模式")
                self.ollama_url = None
    
    def parse_intent(self, user_input: str) -> Dict:
        """
        解析用户意图
        返回: {
            'action': 'search_paper' | 'search_image' | 'add_paper' | 'organize_papers' | 'chat',
            'query': str,
            'params': dict
        }
        """
        # 如果有 LLM，使用 LLM 解析
        if self.use_openai or self.ollama_available:
            return self._parse_with_llm(user_input)
        else:
            # 否则使用规则匹配
            return self._parse_with_rules(user_input)
    
    def _parse_with_llm(self, user_input: str) -> Dict:
        """使用 LLM 解析意图"""
        
        system_prompt = """你是一个智能文献和图像管理助手。你需要理解用户的自然语言输入，并将其转换为结构化的操作指令。

可用的操作：
1. search_paper - 搜索论文（关键词：搜索、查找、找论文、论文、paper）
2. search_image - 搜索图片（关键词：图片、照片、图像、找图、image）
3. add_paper - 添加论文（关键词：添加、上传、导入论文）
4. organize_papers - 整理论文（关键词：整理、分类、归档）
5. chat - 普通对话

请分析用户输入，返回 JSON 格式：
{
    "action": "操作类型",
    "query": "搜索关键词或对话内容",
    "params": {"额外参数": "值"}
}

示例：
输入："帮我找关于 Transformer 的论文"
输出：{"action": "search_paper", "query": "Transformer", "params": {}}

输入："搜索海边日落的图片"
输出：{"action": "search_image", "query": "海边日落", "params": {}}
"""
        
        try:
            if self.use_openai:
                # OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                result = response.choices[0].message.content
            else:
                # Ollama API
                payload = {
                    "model": self.model,
                    "prompt": f"{system_prompt}\n\n用户输入：{user_input}\n\n请返回JSON：",
                    "stream": False,
                    "temperature": 0.3
                }
                response = self._http_post_json(self.ollama_url, payload, timeout=30)
                result = response.get('response', '')
            
            # 解析 JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"action": "chat", "query": user_input, "params": {}}
                
        except Exception as e:
            print(f"⚠️  LLM 解析失败: {e}，使用规则匹配")
            return self._parse_with_rules(user_input)
    
    def _parse_with_rules(self, user_input: str) -> Dict:
        """使用规则匹配解析意图（不依赖 LLM）"""
        user_input_lower = user_input.lower()
        
        # 搜索论文
        paper_keywords = ['论文', '文献', 'paper', '搜索', '查找', '找', 'search', 
                         'transformer', '深度学习', 'deep learning', 'neural']
        if any(kw in user_input_lower for kw in paper_keywords):
            if '图' not in user_input_lower and '照片' not in user_input_lower:
                # 提取关键词
                query = self._extract_query(user_input)
                return {
                    "action": "search_paper",
                    "query": query,
                    "params": {"top_k": 5}
                }
        
        # 搜索图片
        image_keywords = ['图片', '图像', '照片', 'image', 'photo', '找图', '日落', '山', '猫']
        if any(kw in user_input_lower for kw in image_keywords):
            query = self._extract_query(user_input)
            return {
                "action": "search_image",
                "query": query,
                "params": {"top_k": 5}
            }
        
        # 添加论文
        add_keywords = ['添加', '上传', '导入', 'add', '新增']
        if any(kw in user_input_lower for kw in add_keywords):
            return {
                "action": "add_paper",
                "query": user_input,
                "params": {"topics": "CV,NLP,RL"}
            }
        
        # 整理论文
        organize_keywords = ['整理', '分类', '归档', 'organize', '批量']
        if any(kw in user_input_lower for kw in organize_keywords):
            return {
                "action": "organize_papers",
                "query": user_input,
                "params": {"topics": "CV,NLP,RL"}
            }
        
        # 默认为对话
        return {
            "action": "chat",
            "query": user_input,
            "params": {}
        }
    
    def _extract_query(self, text: str) -> str:
        """从文本中提取搜索关键词"""
        # 移除常见的动词和介词
        stop_words = ['帮我', '请', '找', '搜索', '查找', '关于', '的', '图片', '照片', 
                     '论文', '文献', 'search', 'find', 'about', 'the', 'a', 'an']
        
        words = text.split()
        query_words = [w for w in words if w.lower() not in stop_words]
        return ' '.join(query_words) if query_words else text
    
    def _safe_import_requests(self):
        try:
            import requests  # type: ignore
            return requests
        except Exception:
            return None
    
    def _http_get_json(self, url: str, timeout: Optional[float] = None) -> Dict:
        if self._requests:
            response = self._requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        
        request = urllib.request.Request(url)
        if timeout is None:
            with urllib.request.urlopen(request) as resp:
                raw = resp.read()
        else:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}
    
    def _http_post_json(self, url: str, payload: Dict, timeout: Optional[float] = None) -> Dict:
        if self._requests:
            response = self._requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        if timeout is None:
            with urllib.request.urlopen(request) as resp:
                raw = resp.read()
        else:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}
    
    def _check_ollama_alive(self) -> bool:
        try:
            self._http_get_json(f"{self.ollama_host}/api/tags", timeout=3)
            return True
        except Exception:
            return False
    
    def _ensure_ollama_model(self) -> bool:
        try:
            tags = self._http_get_json(f"{self.ollama_host}/api/tags", timeout=5) or {}
            models = tags.get("models") or tags.get("data") or []
            if any(m.get("name") == self.model for m in models):
                return True
            
            print(f"⬇️  自动拉取模型 {self.model} ...")
            pulled = self._pull_model()
            if pulled:
                print(f"✅ 模型 {self.model} 已就绪")
            else:
                print(f"⚠️  拉取模型 {self.model} 失败，将使用规则匹配模式")
            return pulled
        except Exception as exc:
            print(f"⚠️  检查/拉取模型失败: {exc}")
            return False
    
    def _pull_model(self) -> bool:
        pull_url = f"{self.ollama_host}/api/pull"
        payload = {"model": self.model}
        
        try:
            if self._requests:
                with self._requests.post(pull_url, json=payload, stream=True, timeout=None) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                            data = json.loads(decoded)
                        except Exception:
                            continue
                        if data.get("status") == "success":
                            return True
                return False
            
            request = urllib.request.Request(
                pull_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request) as resp:
                raw = resp.read()
            for line in raw.splitlines()[::-1]:
                try:
                    decoded = line.decode("utf-8") if isinstance(line, (bytes, bytearray)) else line
                    data = json.loads(decoded)
                except Exception:
                    continue
                if data.get("status") == "success":
                    return True
            return False
        except Exception as exc:
            print(f"⚠️  自动拉取模型失败: {exc}")
            return False
    
    def generate_response(self, action: str, results: List, query: str) -> str:
        """生成友好的响应"""
        if action == "search_paper":
            if results:
                response = f"🔍 找到 {len(results)} 篇相关论文：\n\n"
                for i, result in enumerate(results[:3], 1):
                    filename = result['metadata']['filename']
                    topic = result['metadata'].get('topic', 'Unknown')
                    similarity = 1 - result.get('distance', 0)
                    response += f"{i}. 📄 {filename}\n"
                    response += f"   主题: {topic} | 相关度: {similarity:.1%}\n\n"
                return response
            else:
                return "😔 没有找到相关论文，试试其他关键词？"
        
        elif action == "search_image":
            if results:
                response = f"🖼️  找到 {len(results)} 张相关图片：\n\n"
                for i, result in enumerate(results[:3], 1):
                    filename = result['metadata']['filename']
                    similarity = 1 - result.get('distance', 0)
                    response += f"{i}. 🎨 {filename} (相关度: {similarity:.1%})\n"
                return response
            else:
                return "😔 没有找到相关图片，试试其他描述？"
        
        elif action == "chat":
            return "我是你的文献和图像管理助手！你可以：\n\n" \
                   "📄 搜索论文：「找关于 Transformer 的论文」\n" \
                   "🖼️  搜索图片：「找日落的照片」\n" \
                   "➕ 添加论文：「添加这篇论文 path/to/paper.pdf」\n" \
                   "📁 整理论文：「整理我的论文库」"
        
        return "操作完成！"


def interactive_chat():
    """交互式对话模式"""
    from app.embeddings import TextEmbedder, ImageEmbedder
    from app.chroma_store import ChromaStore
    
    print("🤖 智能助手启动...")
    print("=" * 50)
    
    # 初始化
    agent = LLMAgent(use_openai=False)
    text_embedder = TextEmbedder()
    image_embedder = ImageEmbedder()
    store = ChromaStore()
    
    print("\n💬 你可以用自然语言跟我对话！")
    print("示例:")
    print("  - 帮我找关于深度学习的论文")
    print("  - 搜索海边日落的图片")
    print("  - 找一下 Transformer 相关的文献")
    print("\n输入 'quit' 或 'exit' 退出\n")
    
    while True:
        try:
            user_input = input("👤 你: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', '退出', 'bye']:
                print("👋 再见！")
                break
            
            # 解析意图
            intent = agent.parse_intent(user_input)
            action = intent['action']
            query = intent['query']
            
            print(f"\n🤔 理解: [{action}] {query}")
            print("🔄 处理中...\n")
            
            # 执行操作
            if action == "search_paper":
                embedding = text_embedder.embed(query)
                results = store.search('papers', embedding, n_results=5)
                response = agent.generate_response(action, results, query)
                print(f"🤖 助手: {response}")
            
            elif action == "search_image":
                embedding = image_embedder.embed_text(query)
                results = store.search('images', embedding, n_results=5)
                response = agent.generate_response(action, results, query)
                print(f"🤖 助手: {response}")
            
            else:
                response = agent.generate_response(action, [], query)
                print(f"🤖 助手: {response}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}\n")


if __name__ == "__main__":
    interactive_chat()
