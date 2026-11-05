import re


class WhitespaceTokenizer:
    """Each word is one token, with the same output format as a Hugging Face tokenizer."""

    def __call__(self, text, **kwargs):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}
