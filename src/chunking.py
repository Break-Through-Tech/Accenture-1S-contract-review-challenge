import json
import re
from bisect import bisect_left, bisect_right
import pandas as pd

def get_tokenizer(name="bert-base-uncased"):
    """Load the tokenizer on demand (not at import time)."""
    from transformers import AutoTokenizer
 
    return AutoTokenizer.from_pretrained(name)

ABBREVIATIONS = [
    "Inc", "Corp", "Ltd", "Co", "LLC", "L.P", "No", "Nos", "Sec", "Secs",
    "Art", "Mr", "Mrs", "Ms", "Dr", "St", "vs", "etc", "e.g", "i.e", "Jr", "Sr",
]

_NOT_ABBREVIATION = "".join(
    rf"(?<!\b{re.escape(a)}\.)" for a in ABBREVIATIONS
)
BOUNDARY_RE = re.compile(rf"\n+|{_NOT_ABBREVIATION}(?<=[.!?])\s+")


HEADING_RE = re.compile(
    r"(?:^[ \t]*|\n[ \t]*|(?<=[.!?;:])[ \t]+)"
    r"(?P<h>(?:(?:ARTICLE|Article|SECTION|Section)[ \t]+[\dIVXLC]+\b"
    r"|\d{1,2}(?:\.\d{1,2}){0,3}\.?(?=[ \t]+[A-Z])))"
)


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
                flags=re.IGNORECASE,
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
                    "end": answer_start + len(answer_text),
                })
 
    return answer_spans


def _tokenize(text, tokenizer):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return encoded["input_ids"], encoded["offset_mapping"]

def _sentence_starts(text, token_ends):
    """Token indices where a new sentence/line begins (valid chunk ends AND starts)."""
    return sorted({
        bisect_right(token_ends, m.end())
        for m in BOUNDARY_RE.finditer(text)
    })

def _build_chunk(text, offsets, start_token, end_token, answer_spans):
    """One chunk dict covering tokens [start_token, end_token).
 
    Label fields:
      categories          - categories with a gold span fully inside the chunk
      partial_categories  - categories with a span the chunk cuts (and no
                            complete span of that category inside)
      label_status        - positive: holds a whole target clause
                            partial : holds only a fragment (NOT a clean negative)
                            negative: holds no part of any target clause
    """
    char_start = offsets[start_token][0]
    char_end = offsets[end_token - 1][1]
 
    inside = {
        s["category"] for s in answer_spans
        if s["start"] >= char_start and s["end"] <= char_end
    }
    touching = {
        s["category"] for s in answer_spans
        if s["start"] < char_end and s["end"] > char_start
    }
 
    if inside:
        label_status = "positive"
    elif touching:
        label_status = "partial"
    else:
        label_status = "negative"
 
    return {
        "text": text[char_start:char_end],
        "start_token": start_token,
        "end_token": end_token,
        "char_start": char_start,
        "char_end": char_end,
        "num_tokens": end_token - start_token,
        "categories": sorted(inside),
        "partial_categories": sorted(touching - inside),
        "label_status": label_status,
    }

def chunk_text(text, tokenizer, max_tokens=450, overlap=75, answer_spans=None):
    """Cut `text` into sentence-aware windows of at most `max_tokens` tokens.
 
    Both ends of each window snap to sentence/line boundaries when possible.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be between 0 and max_tokens - 1")
 
    token_ids, offsets = _tokenize(text, tokenizer)
    if not token_ids:
        return []
 
    token_ends = [end for _, end in offsets]
    preferred_ends = _sentence_starts(text, token_ends)
 
    chunks = []
    start_token = 0
    minimum_chunk_size = max(1, max_tokens // 2)
    answer_spans = answer_spans or []
 
    while start_token < len(token_ids):
        end_token = min(start_token + max_tokens, len(token_ids))
 
        if end_token < len(token_ids):
            idx = bisect_right(preferred_ends, end_token) - 1
            if idx >= 0 and preferred_ends[idx] >= start_token + minimum_chunk_size:
                end_token = preferred_ends[idx]
 
        chunks.append(
            _build_chunk(text, offsets, start_token, end_token, answer_spans)
        )
 
        if end_token >= len(token_ids):
            break

        target = end_token - overlap
        next_start = max(start_token + 1, target)
        idx = bisect_right(preferred_ends, target) - 1
        if idx >= 0:
            snapped = preferred_ends[idx]
            if snapped > start_token and snapped >= target - overlap:
                next_start = snapped
        start_token = next_start
 
    return chunks

def _split_long_span(a, b, max_tokens, sentence_starts):
    """Split tokens [a, b) into <= max_tokens pieces at sentence boundaries.
 
    No overlap. Falls back to a hard cut when no sentence boundary leaves a
    piece at least half the window long.
    """
    minimum = max(1, max_tokens // 2)
    pieces = []
    start = a
    while start < b:
        end = min(start + max_tokens, b)
        if end < b:
            idx = bisect_right(sentence_starts, end) - 1
            if idx >= 0 and sentence_starts[idx] >= start + minimum:
                end = sentence_starts[idx]
        pieces.append((start, end))
        start = end
    return pieces

def section_chunk_text(text, tokenizer, max_tokens=450, answer_spans=None):
    """Section-aware chunks of <= max_tokens tokens, with no overlap.
 
    1. Cut the text into sections at numbered headings (HEADING_RE).
    2. Pack consecutive sections greedily until the next one would overflow.
    3. A section longer than max_tokens is split at sentence boundaries.
    If no headings are found this reduces to plain sentence packing.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
 
    token_ids, offsets = _tokenize(text, tokenizer)
    if not token_ids:
        return []
 
    n = len(token_ids)
    token_starts = [s for s, _ in offsets]
    token_ends = [e for _, e in offsets]
    sentence_starts = _sentence_starts(text, token_ends)
 
    # Section start positions in token space (always include 0 and n).
    cuts = {0, n}
    for m in HEADING_RE.finditer(text):
        idx = bisect_left(token_starts, m.start("h"))
        if 0 < idx < n:
            cuts.add(idx)
    cuts = sorted(cuts)
    sections = list(zip(cuts[:-1], cuts[1:]))
 
    # Greedy packing.
    spans = []
    cur = None  # [start, end] of the chunk being built
    for a, b in sections:
        if b - a > max_tokens:
            if cur is not None:
                spans.append(tuple(cur))
                cur = None
            spans.extend(_split_long_span(a, b, max_tokens, sentence_starts))
        elif cur is None:
            cur = [a, b]
        elif b - cur[0] <= max_tokens:
            cur[1] = b
        else:
            spans.append(tuple(cur))
            cur = [a, b]
    if cur is not None:
        spans.append(tuple(cur))
 
    answer_spans = answer_spans or []
    return [
        _build_chunk(text, offsets, a, b, answer_spans) for a, b in spans
    ]

STRATEGIES = ("sliding", "section")
 
 
def json_to_df_new(path, tokenizer, max_tokens=450, overlap=75,
                   strategy="sliding", verify=True):
    """Chunk every contract in a CUAD JSON file into one DataFrame.
 
    strategy: "sliding" (default, 1S original) or "section".
    `overlap` is used by "sliding" only.
    Adds `contract` (title) and `paragraph_index` so chunks can be grouped
    back to contracts for splitting and contract-level scoring.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
 
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
 
    for contract in data.get("data", []):
        title = contract.get("title")
        answer_spans = get_answer_spans(contract)
        answers_by_paragraph = {}
        for answer in answer_spans:
            answers_by_paragraph.setdefault(answer["paragraph_index"], []).append(answer)
 
        for paragraph_index, paragraph in enumerate(contract.get("paragraphs", [])):
            text = paragraph.get("context", "")
            spans_here = answers_by_paragraph.get(paragraph_index, [])
 
            if strategy == "section":
                paragraph_chunks = section_chunk_text(
                    text, tokenizer, max_tokens=max_tokens,
                    answer_spans=spans_here,
                )
            else:
                paragraph_chunks = chunk_text(
                    text, tokenizer, max_tokens=max_tokens, overlap=overlap,
                    answer_spans=spans_here,
                )
 
            if verify:
                # M2 done-criterion: every chunk is an exact slice of the text.
                for c in paragraph_chunks:
                    assert text[c["char_start"]:c["char_end"]] == c["text"], (
                        f"Chunk is not an exact slice in contract {title!r}"
                    )
 
            for c in paragraph_chunks:
                c["contract"] = title
                c["paragraph_index"] = paragraph_index
            rows.extend(paragraph_chunks)
 
    df = pd.DataFrame(rows)
    if not df.empty:
        cols = ["contract", "paragraph_index"] + [
            c for c in df.columns if c not in ("contract", "paragraph_index")
        ]
        df = df[cols]
    return df


def prepare_training_chunks(df, neg_ratio=3, seed=42):
    positives = df[df["label_status"] == "positive"]
    negatives = df[df["label_status"] == "negative"]
 
    if neg_ratio is not None:
        n_keep = min(len(negatives), int(neg_ratio * len(positives)))
        negatives = negatives.sample(n=n_keep, random_state=seed)
 
    return (
        pd.concat([positives, negatives])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

def get_chunks(path, tokenizer, relevant_categories):
    chunks = json_to_df_new(path, tokenizer)
    chunks['categories'] = chunks['categories'].apply(
        lambda labels: [label for label in labels if label in relevant_categories] or ["None"]
    )
    return chunks

def remove_partial(chunks):
    """Remove chunks with label_status == "partial"."""
    if "label_status" not in chunks.columns:
        return chunks
    return chunks[chunks["label_status"] != "partial"]