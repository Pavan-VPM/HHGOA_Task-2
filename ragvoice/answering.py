from __future__ import annotations

from ragvoice.embed import tokenize


def answer_from_context(question: str, retrievals: list[dict]) -> str:
    if not retrievals:
        return "I could not find grounded context to answer that."
    keywords = set(tokenize(question))
    best_sentence = ""
    best_score = -1
    for hit in retrievals:
        for sentence in hit["text"].split(". "):
            tokens = set(tokenize(sentence))
            score = len(tokens & keywords)
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
    if not best_sentence:
        best_sentence = retrievals[0]["text"].strip()
    sources = ", ".join(sorted({hit["doc_id"] for hit in retrievals[:3]}))
    return f"{best_sentence}. Sources: {sources}"
