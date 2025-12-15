#!/usr/bin/env python3
"""
下载测试图片
"""
import requests
import os
from pathlib import Path

# 确保目录存在
images_dir = Path("storage/images")
images_dir.mkdir(parents=True, exist_ok=True)

# 使用真实的图片 URL（来自 Picsum Photos - 免费图片服务）
images = {
    "sunset.jpg": "https://picsum.photos/id/1015/800/600.jpg",  # 海景
    "mountain.jpg": "https://picsum.photos/id/1018/800/600.jpg",  # 山景
    "nature.jpg": "https://picsum.photos/id/1020/800/600.jpg",  # 自然
    "city.jpg": "https://picsum.photos/id/1022/800/600.jpg",  # 城市
    "forest.jpg": "https://picsum.photos/id/1019/800/600.jpg",  # 森林
}

print("🖼️  开始下载测试图片...")
print("=" * 50)

for filename, url in images.items():
    filepath = images_dir / filename
    try:
        print(f"📥 下载 {filename}...", end=" ")
        
        # 禁用 SSL 验证以避免证书问题
        response = requests.get(url, timeout=30, verify=False)
        response.raise_for_status()
        
        # 保存图片
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        # 验证文件大小
        size = filepath.stat().st_size
        if size > 1000:
            print(f"✅ ({size // 1024} KB)")
        else:
            print(f"⚠️  文件太小 ({size} bytes)")
            
    except Exception as e:
        print(f"❌ 失败: {e}")

print()
print("✅ 下载完成！")
print()
print("📁 下载的图片:")
for img in sorted(images_dir.glob("*.jpg")):
    size = img.stat().st_size
    print(f"  {img.name}: {size // 1024} KB")

print()
print("🔍 测试图像搜索...")
print("-" * 50)
os.system('uv run python main.py search-image "beautiful landscape"')