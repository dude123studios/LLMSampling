from sentence_transformers import SentenceTransformer, util
import torch

class SemanticPruner:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)
        self.history_embeddings = None
        self.chunk_size = config.chunk_size
        self.threshold = config.similarity_threshold

    def is_redundant(self, current_text):
        if self.history_embeddings is None:
            return False
            
        current_emb = self.model.encode(current_text, convert_to_tensor=True)
        
        # Compute cosine similarities against all stored chunks
        cosine_scores = util.cos_sim(current_emb, self.history_embeddings)
        
        # Check if any score exceeds threshold
        return torch.any(cosine_scores > self.threshold).item()
        
    def add_chunk(self, text):
        new_emb = self.model.encode(text, convert_to_tensor=True)
        if new_emb.ndim == 1:
            new_emb = new_emb.unsqueeze(0)
            
        if self.history_embeddings is None:
            self.history_embeddings = new_emb
        else:
            self.history_embeddings = torch.cat([self.history_embeddings, new_emb], dim=0)
