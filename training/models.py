import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

class TrainableTextEncoder(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=128, output_dim=64, max_len=32):
        super().__init__()
        self.output_dim = output_dim
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.projector = nn.Linear(embed_dim, output_dim)
        self.vocab_size = vocab_size

    def _stable_hash(self, s):
        # Use MD5 for stable hashing across runs
        return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16)

    def _tokenize(self, texts):
        batch_ids = []
        for text in texts:
            words = str(text).lower().split()
            # Stable hash-based tokenization
            ids = [self._stable_hash(w) % self.vocab_size for w in words]
            if len(ids) > self.max_len:
                ids = ids[:self.max_len]
            else:
                ids = ids + [0] * (self.max_len - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long)

    def forward(self, texts):
        device = self.embedding.weight.device
        input_ids = self._tokenize(texts).to(device)
        x = self.embedding(input_ids)
        transformer_out = self.transformer(x)
        pooled_output = torch.mean(transformer_out, dim=1)
        projected = self.projector(pooled_output)
        return projected

class TimeXLModel(nn.Module):
    def __init__(self, num_classes, k=10, input_channels=5, time_seq_len=24, time_dim=256, text_dim=64):
        super().__init__()
        self.num_classes = num_classes
        
        # Time Encoder
        # Input shape: [B, 24, 5] -> Flatten -> [B, 120]
        self.time_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(time_seq_len * input_channels, time_dim), 
            nn.ReLU(),
            nn.Linear(time_dim, time_dim) # Extra layer for capacity
        )
        
        # Text Encoder
        self.text_encoder = TrainableTextEncoder(output_dim=text_dim)
        
        # Fusion Layer
        # Fusion: Time(C*K) + Text(C*K) -> NumClasses
        # PM outputs [B, C*K] for both modalities.
        fusion_input_dim = num_classes * k * 2
        self.fusion_layer = nn.Linear(fusion_input_dim, num_classes)

    def forward(self, z_time, z_text):
        # This forward is conceptual, usually called by components
        pass

    def fusion(self, sim_time, sim_text):
        # sim_time: [B, C*K], sim_text: [B, C*K]
        combined = torch.cat([sim_time, sim_text], dim=1) # [B, 2*C*K]
        return self.fusion_layer(combined)

class PrototypeManager(nn.Module):
    def __init__(self, num_classes=3, k=10, time_dim=256, text_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.time_dim = time_dim
        self.text_dim = text_dim
        
        # Class-specific prototypes: [C, K, D]
        self.P_time = nn.Parameter(torch.randn(num_classes, k, time_dim))
        self.P_text = nn.Parameter(torch.randn(num_classes, k, text_dim))

    def forward(self, z_time, z_text):
        # Calculate similarity with prototypes
        # z_time: [B, D]
        # P_time: [C, K, D]
        
        # [B, C, K]
        sim_time = torch.einsum('bd,ckd->bck', z_time, self.P_time)
        sim_text = torch.einsum('bd,ckd->bck', z_text, self.P_text)
        
        # Flatten to [B, C*K]
        sim_time_flat = sim_time.reshape(z_time.size(0), -1)
        sim_text_flat = sim_text.reshape(z_text.size(0), -1)
        
        return sim_time_flat, sim_text_flat
