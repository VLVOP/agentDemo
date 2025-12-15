#!/bin/bash

echo "🖼️  准备图片测试"
echo ""

# 确保 images 目录存在
mkdir -p storage/images

echo "📥 下载测试图片..."

# 下载不同类型的图片
curl -L -o storage/images/sunset.jpg "https://source.unsplash.com/800x600/?sunset,beach" 2>/dev/null
echo "✓ 下载日落图片"

curl -L -o storage/images/mountain.jpg "https://source.unsplash.com/800x600/?mountain,landscape" 2>/dev/null
echo "✓ 下载山景图片"

curl -L -o storage/images/cat.jpg "https://source.unsplash.com/800x600/?cat,kitten" 2>/dev/null
echo "✓ 下载猫咪图片"

curl -L -o storage/images/city.jpg "https://source.unsplash.com/800x600/?city,urban" 2>/dev/null
echo "✓ 下载城市图片"

curl -L -o storage/images/ocean.jpg "https://source.unsplash.com/800x600/?ocean,sea" 2>/dev/null
echo "✓ 下载海洋图片"

echo ""
echo "✅ 图片准备完成！"
echo ""
echo "📁 当前图片："
ls -lh storage/images/

echo ""
echo "🔍 测试搜索..."
echo ""

# 测试不同的搜索查询
echo "1️⃣  搜索：sunset by the sea"
uv run python main.py search-image "sunset by the sea" --top-k 3

echo ""
echo "2️⃣  搜索：mountain landscape"
uv run python main.py search-image "mountain landscape" --top-k 3

echo ""
echo "3️⃣  搜索：cute cat"
uv run python main.py search-image "cute cat" --top-k 3

echo ""
echo "✨ 图片搜索测试完成！"
