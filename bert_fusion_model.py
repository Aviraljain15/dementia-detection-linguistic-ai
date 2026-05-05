import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ================= MODEL ================= #
class DementiaFusionClassifier(nn.Module):
    def __init__(self, n_linguistic_features: int, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        bert_dim = 768

        self.feat_proj = nn.Sequential(
            nn.Linear(n_linguistic_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
        )
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
    def forward(self, input_ids, attention_mask, linguistic_feats):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = bert_out.last_hidden_state[:, 0, :]
        ling_emb = self.feat_proj(linguistic_feats)
        combined = torch.cat([cls_emb, ling_emb], dim=-1)
        return self.classifier(combined)

# ================= DATASET ================= #
class DementiaDataset(Dataset):
    def __init__(self, texts, features, labels, tokenizer, max_len=256):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'features': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
# ================= TRAIN FUNCTION ================= #
def run_bert_fusion(samples, feature_matrix, epochs=3, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---------------- PREPARE DATA ---------------- #
    texts = [s.clean_text for s in samples]
    labels = [s.label for s in samples]

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    X_train, X_test, f_train, f_test, y_train, y_test = train_test_split(
        texts,
        feature_matrix,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )
    # ---------------- DATASETS ---------------- #
    train_dataset = DementiaDataset(X_train, f_train, y_train, tokenizer)
    test_dataset = DementiaDataset(X_test, f_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # ---------------- MODEL ---------------- #
    model = DementiaFusionClassifier(n_linguistic_features=len(feature_matrix[0]))
    model.to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': 2e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
        {'params': model.feat_proj.parameters(), 'lr': 1e-3},
    ], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2]).to(device))
    # ================= TRAIN ================= #
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels_batch = batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    # ================= EVALUATION ================= #
    model.eval()
    preds = []
    true = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels_batch = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, features)
            pred = torch.argmax(outputs, dim=1)

            preds.extend(pred.cpu().numpy())
            true.extend(labels_batch.cpu().numpy())

    print("\n===== BERT FUSION RESULTS =====\n")
    print(classification_report(true, preds))

    return model