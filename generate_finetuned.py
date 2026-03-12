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


def generate(model, idx, max_new_tokens, context_size, temperature=0.8, top_k=20, eos_id=None):
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

        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def clean_generated_output(decoded_text):
    if "الاستجابة:" in decoded_text:
        decoded_text = decoded_text.split("الاستجابة:", 1)[-1].strip()

    decoded_text = decoded_text.replace("التعليمات:", "").replace("المدخل:", "").strip()
    return decoded_text


def generate_text(model, tokenizer, prompt, max_new_tokens=120, temperature=0.8, top_k=20):
    input_ids = tokenizer.encode(prompt)

    if len(input_ids) == 0:
        input_ids = [0]

    checkpoint_vocab_size = model.token_embedding_table.weight.shape[0]
    input_ids = [tid for tid in input_ids if 0 <= tid < checkpoint_vocab_size]

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
            eos_id=None,
        )[0].tolist()

    valid_ids = [tid for tid in generated_ids if 0 <= tid < len(tokenizer.id_to_token)]
    decoded = tokenizer.decode(valid_ids)
    return clean_generated_output(decoded)


if __name__ == "__main__":
    tokenizer = BPETokenizer()
    tokenizer.load("checkpoints/tokenizer")

    story_model = load_model("checkpoints/finetuned/story_completion 2.pt")
    poetry_model = load_model("checkpoints/finetuned/poetry 2.pt")

    story_prompt = (
        "التعليمات:\n"
        "اكمل القصة التالية.\n\n"
        "المدخل:\n"
        "في ليلة هادئة وقف الطفل أمام الباب القديم.\n\n"
        "الاستجابة:\n"
    )

    poetry_prompt = (
        "التعليمات:\n"
        "اكتب أبياتا شعرية عن المطر.\n\n"
        "المدخل:\n"
        "المطر\n\n"
        "الاستجابة:\n"
    )

    story_output = generate_text(
        story_model,
        tokenizer,
        story_prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=30,
    )

    poetry_output = generate_text(
        poetry_model,
        tokenizer,
        poetry_prompt,
        max_new_tokens=60,
        temperature=0.7,
        top_k=20,
    )

    print("=" * 60)
    print("STORY MODEL OUTPUT")
    print("=" * 60)
    print(story_output)

    print("\n" + "=" * 60)
    print("POETRY MODEL OUTPUT")
    print("=" * 60)
    print(poetry_output)