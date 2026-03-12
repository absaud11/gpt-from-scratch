import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.transformer import GPTLanguageModel
from src.tokenizer.bpe_tokenizer import BPETokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device)
    checkpoint_vocab_size = state_dict["token_embedding_table.weight"].shape[0]

    model = GPTLanguageModel(
        vocab_size=checkpoint_vocab_size,
        block_size=128,
        n_embd=128,
        n_head=4,
        n_layer=4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate(model, idx, max_new_tokens, context_size, temperature=0.8, top_k=20):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            out = model(idx_cond)

        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out

        logits = logits[:, -1, :]

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            top_logits, _ = torch.topk(logits, k)
            min_val = top_logits[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_val,
                torch.full_like(logits, float("-inf")),
                logits
            )

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8, top_k=20):
    input_ids = tokenizer.encode(prompt)

    if len(input_ids) == 0:
        input_ids = [0]

    x = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        generated_ids = generate(
            model=model,
            idx=x,
            max_new_tokens=max_new_tokens,
            context_size=128,
            temperature=temperature,
            top_k=top_k,
        )[0].tolist()

    valid_ids = [tid for tid in generated_ids if 0 <= tid < len(tokenizer.id_to_token)]
    return tokenizer.decode(valid_ids)


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load("checkpoints/tokenizer")

    model = load_model("checkpoints/pretrained/model 2.pt")

    prompt = "في ليلة هادئة"
    output = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=120,
        temperature=0.8,
        top_k=20,
    )

    print("=" * 60)
    print("PRETRAINED MODEL OUTPUT")
    print("=" * 60)
    print(output)