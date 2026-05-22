"""用户与物品嵌入生成模型。"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class EmbeddingModel(nn.Module):
    """基于BERT的文本嵌入生成模型。"""

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.max_length = max_length

    def forward(self, texts: list[str]) -> list[torch.Tensor]:
        """为文本列表生成CLS嵌入向量。"""
        self.bert.eval()
        embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for text in texts:
                encoded_input = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.bert.device)

                output = self.bert(**encoded_input)
                cls_embedding = output.last_hidden_state[:, 0, :].squeeze(0)
                embeddings.append(cls_embedding.cpu())

        return embeddings


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingModel().to(device)

    texts = [
        "This is a great book about artificial intelligence.",
        "Language models like BERT are widely used for embeddings.",
        "Embedding generation is crucial for recommendation systems.",
    ]

    embeddings = model(texts)

    print("=== 嵌入生成结果 ===")
    for i, (text, embed) in enumerate(zip(texts, embeddings)):
        print(f"文本 {i+1}: {text}")
        print(f"嵌入向量 (前10维): {embed.numpy()[:10]}...")
        print(f"嵌入维度: {embed.shape[0]}")
        print()


if __name__ == "__main__":
    main()
