#!/bin/bash
echo "🚀 测试 AI Agent"
echo ""
echo "📚 整理论文..."
uv run python main.py organize-papers storage/papers --topics "CV,NLP,RL"

echo ""
echo "📁 查看分类结果："
find storage/papers -name "*.pdf" -type f

echo ""
echo "🔍 搜索测试："
uv run python main.py search-paper "deep learning"

echo ""
echo "✅ 测试完成！"