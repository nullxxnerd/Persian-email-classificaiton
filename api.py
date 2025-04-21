from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import unicodedata
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Request schema
class TextRequest(BaseModel):
    text: str

# Load tokenizer and config
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('config.pkl', 'rb') as f:
    config = pickle.load(f)
max_len = config['max_len']
label_map_inv = config['label_map_inv']
stop_words = set(config['stop_words'])
model_config = config['model_config']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model architecture
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        convs = [F.relu(conv(x)) for conv in self.convs]
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in convs]
        out = torch.cat(pools, 1)
        out = self.dropout(out)
        return self.fc(out)

# Instantiate and load state
model = TextCNN(
    vocab_size=model_config['vocab_size'],
    embedding_dim=model_config['embedding_dim'],
    num_filters=model_config['num_filters'],
    filter_sizes=model_config['filter_sizes'],
    num_classes=model_config['num_classes'],
    dropout=model_config['dropout']
)
model.load_state_dict(torch.load('textcnn.pth', map_location=device))
model.to(device)
model.eval()

# Preprocessing function
punct_regex = re.compile(r"[\d%s]+" % re.escape(string.punctuation))
def preprocess_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text.lower())
    text = punct_regex.sub(' ', text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)

app = FastAPI()

@app.post('/predict')
def predict(req: TextRequest):
    cleaned = preprocess_text(req.text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    inp = torch.tensor(padded, dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    idx = int(torch.argmax(torch.tensor(probs)))
    return {
        'label': label_map_inv[idx],
        'probabilities': {label_map_inv[i]: float(probs[i]) for i in range(len(probs))}
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
