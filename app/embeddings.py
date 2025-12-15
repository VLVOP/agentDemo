"""
Text and Image embedding helpers using SentenceTransformers and CLIP
"""
import os
# 设置 Hugging Face 镜像（解决国内网络问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List
from app.config import TEXT_MODEL_NAME, IMAGE_MODEL_NAME, DEVICE


class TextEmbedder:
    """Text embedding using SentenceTransformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or TEXT_MODEL_NAME
        print(f"Loading text model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=DEVICE)
    
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embedding for text or list of texts"""
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return embeddings
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        return self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True)


class ImageEmbedder:
    """Image and text embedding using CLIP"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or IMAGE_MODEL_NAME
        print(f"Loading CLIP model: {self.model_name}")
        
        # CLIP 需要特殊的加载方式
        try:
            # 尝试直接加载 CLIP 模型
            from transformers import CLIPProcessor, CLIPModel
            import torch
            
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.device = DEVICE
            self.model.to(self.device)
            self.use_clip = True
            print(f"✅ Loaded CLIP model with transformers")
        except Exception as e:
            # 如果失败，尝试使用 sentence-transformers 的 CLIP 模型
            print(f"Using sentence-transformers CLIP wrapper...")
            self.model = SentenceTransformer('clip-ViT-B-32', device=DEVICE)
            self.use_clip = False
    
    def embed_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Generate embedding for an image"""
        image = Image.open(image_path).convert('RGB')
        
        if self.use_clip:
            import torch
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            embedding = image_features.cpu().numpy().flatten()
        else:
            embedding = self.model.encode(image, convert_to_numpy=True)
        
        return embedding
    
    def embed_images(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """Generate embeddings for multiple images"""
        images = [Image.open(path).convert('RGB') for path in image_paths]
        
        if self.use_clip:
            import torch
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            embeddings = image_features.cpu().numpy()
        else:
            embeddings = self.model.encode(images, convert_to_numpy=True)
        
        return embeddings
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embedding for text using CLIP's text encoder"""
        if self.use_clip:
            import torch
            inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            embedding = text_features.cpu().numpy().flatten() if isinstance(text, str) else text_features.cpu().numpy()
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
        
        return embedding
    
    def compute_similarity(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> float:
        """Compute cosine similarity between image and text embeddings"""
        similarity = np.dot(image_embedding, text_embedding) / (
            np.linalg.norm(image_embedding) * np.linalg.norm(text_embedding)
        )
        return float(similarity)