import json
import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

import re
from bisect import bisect_right


def get_answer_spans(contract):
    answer_spans = []

    for paragraph_index, paragraph in enumerate(contract.get("paragraphs", [])):
        for qa in paragraph.get("qas", []):
            answers = qa.get("answers") or []
            if not answers:
                continue

            question = str(qa.get("question", "")).strip()
            category_match = re.search(
                r'related to\s+"([^"]+)"',
                question,
                flags=re.IGNORECASE
            )
            category = (
                category_match.group(1).strip().casefold()
                if category_match
                else question.casefold()
            )

            for answer in answers:
                answer_text = str(answer.get("text", "") or "")
                answer_start = int(answer.get("answer_start", 0))
                answer_spans.append({
                    "paragraph_index": paragraph_index,
                    "category": category,
                    "question": question,
                    "text": answer_text,
                    "start": answer_start,
                    "end": answer_start + len(answer_text)
                })

    return answer_spans


def chunk_text(text, tokenizer, max_tokens=450, overlap=75, answer_spans=None):
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be between 0 and max_tokens - 1")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if not token_ids:
        return []

    token_ends = [end for _, end in offsets]
    boundary_chars = [
        match.end()
        for match in re.finditer(r"\n+|(?<=[.!?])\s+", text)
    ]
    preferred_ends = sorted({
        bisect_right(token_ends, char_position)
        for char_position in boundary_chars
    })

    chunks = []
    start_token = 0
    minimum_chunk_size = max(1, max_tokens // 2)
    answer_spans = answer_spans or []

    while start_token < len(token_ids):
        end_token = min(start_token + max_tokens, len(token_ids))

        if end_token < len(token_ids):
            boundary_index = bisect_right(preferred_ends, end_token) - 1
            while boundary_index >= 0:
                candidate_end = preferred_ends[boundary_index]
                if candidate_end < start_token + minimum_chunk_size:
                    break
                if candidate_end > start_token:
                    end_token = candidate_end
                    break
                boundary_index -= 1

        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        categories = sorted({
            span["category"]
            for span in answer_spans
            if span["start"] < char_end and span["end"] > char_start
        })
        chunks.append({
            "text": text[char_start:char_end],
            "start_token": start_token,
            "end_token": end_token,
            "char_start": char_start,
            "char_end": char_end,
            "num_tokens": end_token - start_token,
            "categories": categories
        })

        if end_token >= len(token_ids):
            break
        start_token = max(start_token + 1, end_token - overlap)

    return chunks

def json_to_df_new(path, tokenizer):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for contract in data.get("data", []):
        answer_spans = get_answer_spans(contract)
        answers_by_paragraph = {}
        for answer in answer_spans:
            answers_by_paragraph.setdefault(answer["paragraph_index"], []).append(answer)

        for paragraph_index, paragraph in enumerate(contract.get("paragraphs", [])):
            text = paragraph.get("context", "")
            paragraph_answers = answers_by_paragraph.get(paragraph_index, [])
            paragraph_chunks = chunk_text(
                text,
                tokenizer,
                answer_spans=paragraph_answers
            )
            chunks.extend(paragraph_chunks)

    return pd.DataFrame(chunks)