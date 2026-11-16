import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.models.tokenizer_screening import FragmentationReport, rank_candidates, screen_tokenizer  # noqa: E402


class CharTokenizer:
    """1 token per character - the worst realistic case."""

    vocab_size = 100

    def tokenize(self, text):
        return list(text.replace(" ", "▁"))

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % self.vocab_size for c in text]


class WordTokenizer:
    """1 token per word - the best realistic case."""

    vocab_size = 50_000

    def tokenize(self, text):
        return text.split()

    def encode(self, text, add_special_tokens=False):
        return [len(w) for w in text.split()]


def test_fertility_of_perfect_word_tokenizer_is_one():
    rep = screen_tokenizer("word", WordTokenizer(), ["जलवायु परिवर्तन गंभीर है"], "Hindi")
    assert rep.fertility == 1.0
    assert rep.n_words == 4 and rep.n_tokens == 4


def test_character_tokenizer_has_high_fertility_and_single_chars():
    rep = screen_tokenizer("char", CharTokenizer(), ["जलवायु परिवर्तन"], "Hindi")
    assert rep.fertility > 5
    assert rep.single_char_share == 1.0


def test_chars_per_token_is_inverse_of_fragmentation():
    texts = ["জলবায়ু পরিবর্তন একটি সমস্যা"]
    word = screen_tokenizer("w", WordTokenizer(), texts, "Bengali")
    char = screen_tokenizer("c", CharTokenizer(), texts, "Bengali")
    assert char.chars_per_token < word.chars_per_token


def test_truncation_rate_at_budget():
    long_text = " ".join(["शब्द"] * 100)
    rep = screen_tokenizer("w", WordTokenizer(), [long_text], "Hindi", max_length=64)
    assert rep.truncation_rate_at_64 == 1.0
    short = screen_tokenizer("w", WordTokenizer(), ["छोटा वाक्य है"], "Hindi", max_length=64)
    assert short.truncation_rate_at_64 == 0.0


class UnkTokenizer:
    vocab_size = 10

    def tokenize(self, text):
        return ["[UNK]"] * len(text.split())

    def encode(self, text, add_special_tokens=False):
        return [0]


def test_unk_rate_detected():
    rep = screen_tokenizer("unk", UnkTokenizer(), ["जलवायु परिवर्तन है"], "Hindi")
    assert rep.unk_rate == 1.0


def test_empty_corpus_is_safe():
    rep = screen_tokenizer("w", WordTokenizer(), [], "Hindi")
    assert rep.n_examples == 0 and rep.fertility == 0.0


def test_ranking_is_ascending_on_mean_fertility():
    rows = [
        FragmentationReport("bad", "T", 100, "Hindi", 1, 10, 40, 4.0, 2.0, 0.0, 1.0, 40, 40.0, 0.0, 40),
        FragmentationReport("bad", "T", 100, "Bengali", 1, 10, 40, 4.0, 2.0, 0.0, 1.0, 40, 40.0, 0.0, 40),
        FragmentationReport("good", "T", 100, "Hindi", 1, 10, 12, 1.2, 6.0, 0.0, 0.0, 12, 12.0, 0.0, 12),
        FragmentationReport("good", "T", 100, "Bengali", 1, 10, 13, 1.3, 5.5, 0.0, 0.0, 13, 13.0, 0.0, 13),
    ]
    ranked = rank_candidates(rows)
    assert [r["model"] for r in ranked] == ["good", "bad"]
    assert ranked[0]["rank"] == 1


def test_ranking_keeps_single_language_candidate():
    rows = [FragmentationReport("hi_only", "T", 100, "Hindi", 1, 5, 10, 2.0, 3.0, 0.0, 0.0, 10, 10.0, 0.0, 10)]
    ranked = rank_candidates(rows)
    assert ranked[0]["fertility_bn"] is None
    assert ranked[0]["fertility_mean"] == 2.0
