"""Phase 8 - teacher-candidate screening by Hindi/Bengali tokenizer fragmentation.

Why screen tokenizers before fine-tuning anything: on a 2-vCPU / 1.9 GB box with
`max_length: 64`, a tokenizer that needs 2x more subword units per sentence does
not merely cost latency, it silently truncates the stance-bearing clause. Token
fragmentation is therefore a *modelling* decision, not a housekeeping metric.

All metrics are computed from real tokenizer runs over real Devanagari and
Bengali bytes; nothing is estimated from parameter counts or model cards.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Protocol


class TokenizerLike(Protocol):
    def tokenize(self, text: str) -> list[str]: ...
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


UNK_STRINGS = {"[UNK]", "<unk>", "<UNK>", "[unk]", "<UNK_TOKEN>"}


@dataclass
class FragmentationReport:
    model: str
    tokenizer_class: str
    vocab_size: int
    language: str
    n_examples: int
    n_words: int
    n_tokens: int
    fertility: float
    chars_per_token: float
    unk_rate: float
    single_char_share: float
    p95_tokens: int
    mean_tokens: float
    truncation_rate_at_64: float
    max_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> list[str]:
    return [w for w in str(text).split() if w.strip()]


def screen_tokenizer(
    model_name: str,
    tokenizer: TokenizerLike,
    texts: Iterable[str],
    language: str,
    max_length: int = 64,
) -> FragmentationReport:
    """Compute fragmentation statistics for one tokenizer over one corpus.

    fertility          = subword tokens per whitespace-delimited word
                         (the standard intrinsic tokenizer metric; lower is better)
    chars_per_token    = characters consumed per subword token (higher is better
                         for a fixed vocabulary - the byte/char side of the same coin)
    unk_rate           = share of emitted tokens that are the unknown token
    single_char_share  = share of tokens of length 1, i.e. the model must rebuild
                         a word from single graphemes - the classic failure mode
                         of a Latin-centric tokenizer on Devanagari/Bengali
    truncation_rate    = share of examples whose token length exceeds max_length
                         at the project's CPU budget
    """
    texts = [t for t in texts if str(t).strip()]
    n_examples = len(texts)
    n_words = n_tokens = n_chars = 0
    unk = singles = 0
    lengths: list[int] = []

    for t in texts:
        toks = tokenizer.tokenize(str(t))
        n_tok = len(toks)
        n_tokens += n_tok
        n_words += len(_words(t))
        n_chars += len(str(t))
        lengths.append(n_tok)
        unk += sum(1 for x in toks if x in UNK_STRINGS)
        singles += sum(1 for x in toks if len(x.replace("##", "").replace("▁", "")) <= 1)

    lengths.sort()
    p95_idx = min(n_examples - 1, int(round(0.95 * (n_examples - 1)))) if n_examples else 0

    return FragmentationReport(
        model=model_name,
        tokenizer_class=type(tokenizer).__name__,
        vocab_size=int(getattr(tokenizer, "vocab_size", 0) or 0),
        language=language,
        n_examples=n_examples,
        n_words=n_words,
        n_tokens=n_tokens,
        fertility=round(n_tokens / n_words, 4) if n_words else 0.0,
        chars_per_token=round(n_chars / n_tokens, 4) if n_tokens else 0.0,
        unk_rate=round(unk / n_tokens, 6) if n_tokens else 0.0,
        single_char_share=round(singles / n_tokens, 4) if n_tokens else 0.0,
        p95_tokens=lengths[p95_idx] if lengths else 0,
        mean_tokens=round(sum(lengths) / n_examples, 2) if n_examples else 0.0,
        truncation_rate_at_64=round(sum(1 for x in lengths if x > max_length) / n_examples, 4) if n_examples else 0.0,
        max_tokens=lengths[-1] if lengths else 0,
    )


def rank_candidates(rows: list[FragmentationReport], languages: tuple[str, str] = ("Hindi", "Bengali")) -> list[dict]:
    """Rank by mean fertility across the two target languages, ascending.

    The project has no prior ranking rule for tokenizers, so this is the stated
    criterion: mean fertility over Hindi + Bengali, lower is better. Ties break on
    mean chars-per-token (higher better), then on mean truncation rate (lower).
    """
    by_model: dict[str, dict] = {}
    for r in rows:
        bucket = by_model.setdefault(r.model, {"model": r.model, "tokenizer_class": r.tokenizer_class,
                                               "vocab_size": r.vocab_size})
        bucket[r.language] = r

    ranked = []
    for model, b in by_model.items():
        lang_rows = [b[l] for l in languages if l in b]
        if not lang_rows:
            continue
        ranked.append({
            "model": model,
            "tokenizer_class": b["tokenizer_class"],
            "vocab_size": b["vocab_size"],
            "fertility_hi": b.get("Hindi").fertility if b.get("Hindi") else None,
            "fertility_bn": b.get("Bengali").fertility if b.get("Bengali") else None,
            "fertility_mean": round(sum(x.fertility for x in lang_rows) / len(lang_rows), 4),
            "chars_per_token_mean": round(sum(x.chars_per_token for x in lang_rows) / len(lang_rows), 4),
            "unk_rate_mean": round(sum(x.unk_rate for x in lang_rows) / len(lang_rows), 6),
            "single_char_share_mean": round(sum(x.single_char_share for x in lang_rows) / len(lang_rows), 4),
            "trunc_rate_mean": round(sum(x.truncation_rate_at_64 for x in lang_rows) / len(lang_rows), 4),
            "mean_tokens_hi": b.get("Hindi").mean_tokens if b.get("Hindi") else None,
            "mean_tokens_bn": b.get("Bengali").mean_tokens if b.get("Bengali") else None,
            "p95_tokens_hi": b.get("Hindi").p95_tokens if b.get("Hindi") else None,
            "p95_tokens_bn": b.get("Bengali").p95_tokens if b.get("Bengali") else None,
            "fertility_en": b.get("English").fertility if b.get("English") else None,
        })

    ranked.sort(key=lambda d: (d["fertility_mean"], -d["chars_per_token_mean"], d["trunc_rate_mean"]))
    for i, d in enumerate(ranked, 1):
        d["rank"] = i
    return ranked
