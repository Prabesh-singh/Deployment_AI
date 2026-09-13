import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel

# ==========================================
# 1. Load Vocabulary Mappings
# ==========================================
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

# Ensure keys and values are correctly assigned
if isinstance(vocab_data, dict):
    stoi = vocab_data
    itos = {int(v): k for k, v in stoi.items()}
else:
    stoi = {word: i for i, word in enumerate(vocab_data)}
    itos = {i: word for i, word in enumerate(vocab_data)}

vocab_size = len(stoi)


# ==========================================
# 2. Transformer Architecture Definition
# ==========================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Enforce causal mask to prevent forward peeking during sequence generation
        attn_output, _ = self.attention(x, x, x, is_causal=True)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2, max_seq_len=32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        token_embed = self.token_embedding(x)
        position_embed = self.position_embedding(positions)
        x = self.blocks(token_embed + position_embed)
        x = self.norm(x)
        return self.head(x)


# ==========================================
# 3. Load Saved Weights
# ==========================================
model = MiniGPT(vocab_size=vocab_size)
model.load_state_dict(torch.load("mini_gpt.pth", map_location=torch.device('cpu')))
model.eval()


# ==========================================
# 4. Answer Generation Function (Sampling)
# ==========================================
def generate_reply(
    prompt: str,
    max_new_tokens: int = 25,
    temperature: float = 0.7,
    top_k: int = 5,
    repetition_penalty: float = 1.25
) -> str:
    model.eval()

    # Tokenize input prompt
    words = prompt.lower().strip().split()
    token_ids = [stoi[w] for w in words if w in stoi]

    if not token_ids:
        return "I do not recognize those words in my training vocabulary."

    input_tensor = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            cond_tensor = input_tensor[:, -32:] # Context window
            logits = model(cond_tensor)[:, -1, :]

            # Apply repetition penalty to prevent "is is is" word loops
            for token in set(input_tensor[0].tolist()):
                logits[0, token] /= repetition_penalty

            # Temperature scaling
            logits = logits / max(temperature, 1e-5)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Probabilistic sampling
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_tensor = torch.cat([input_tensor, next_token], dim=1)

    generated_tokens = input_tensor[0].tolist()
    return " ".join([itos.get(idx, "<UNK>") for idx in generated_tokens])


# ==========================================
# 5. FastAPI Service
# ==========================================
# ==========================================
# 5. FastAPI Service
# ==========================================
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Saved Model QA Service")

# Allow requests from your React Native/Expo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Expo, mobile devices, web browsers)
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 20
    temperature: float = 0.7


@app.post("/ask")
def ask_api(request: QueryRequest):
    answer = generate_reply(
        prompt=request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )
    return {
        "status": "success",
        "question": request.prompt,
        "answer": answer,
    }


# Bind to 0.0.0.0 so external mobile devices on your Wi-Fi can reach it
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)