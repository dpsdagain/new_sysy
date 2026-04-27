import re
from typing import Optional

class FuzzyMatcher:
    """
    Anthropic-Grade Fuzzy String Matcher (F-05).
    Ported from FileEditTool/utils.ts.
    
    Ensures file edits succeed even if the LLM makes minor 
    formatting/typographic errors.
    """
    
    # 🧬 Anthropic Constants (utils.ts:21-24)
    CURLY_QUOTES = {
        '‘': "'", '’': "'",
        '“': '"', '”': '"'
    }

    @staticmethod
    def normalize_quotes(text: str) -> str:
        """Ported from utils.ts:31."""
        for curly, straight in FuzzyMatcher.CURLY_QUOTES.items():
            text = text.replace(curly, straight)
        return text

    @staticmethod
    def strip_trailing_whitespace(text: str) -> str:
        """Ported from utils.ts:44."""
        lines = text.splitlines(keepends=True)
        return "".join([re.sub(r'[ \t]+$', '', line) for line in lines])

    @classmethod
    def find_actual_string(cls, content: str, search: str) -> Optional[str]:
        """
        Ported from utils.ts:73.
        Tries exact match, then normalized match.
        """
        # 1. Exact try
        if search in content:
            return search

        # 2. Normalized try (Quotes + Whitespace)
        n_content = cls.normalize_quotes(cls.strip_trailing_whitespace(content))
        n_search = cls.normalize_quotes(cls.strip_trailing_whitespace(search))

        index = n_content.find(n_search)
        if index != -1:
            # Success! Extract the ORIGINAL content segment to return
            # (Matches utils.ts:89 logic)
            return content[index : index + len(search)]
            
        return None
