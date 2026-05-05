import re
import langid
import pysbd
from gruut import sentences as gruut_sentences
from zh_normalization import TextNormalizer as ZhNormalizer
from num2words import num2words
from cleantext import clean

# --- TTS Chunking ---
# Vocoder OOMs at ~1862 frames (~2900 chars) due to O(n²) attention VRAM.
# 1500 chars ≈ ~10s audio — safe headroom while avoiding excessive splitting.
MAX_CHUNK_CHARS = 1500

class TextNormalizer:
    def __init__(self, default_lang="en"):
        self.default_lang = default_lang
        # Initialize pysbd segmenters for common languages
        self.segmenters = {
            "en": pysbd.Segmenter(language="en", clean=False),
            "zh": pysbd.Segmenter(language="zh", clean=False),
            "fr": pysbd.Segmenter(language="fr", clean=False),
            "de": pysbd.Segmenter(language="de", clean=False),
            "es": pysbd.Segmenter(language="es", clean=False),
        }
        self.zh_norm = ZhNormalizer()
        
    def detect_language(self, text):
        """Detect language using langid."""
        lang, _ = langid.classify(text)
        return lang

    def clean_text(self, text):
        """Basic cleanup: unicode, URLs, etc.  Preserves newlines."""
        return clean(text,
            fix_unicode=True,
            to_ascii=False,
            lower=False,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=False,
            no_numbers=False,
            no_digits=False,
            no_currency_symbols=False,
            normalize_whitespace=False,  # preserve newlines for model prosody
        )

    # Languages that use dot as thousands separator and comma as decimal
    # e.g. 25.000,50 instead of 25,000.50
    DOT_THOUSANDS_LANGS = {"es", "fr", "de", "it", "pt", "ru"}
    
    def _normalize_number(self, num_str: str, lang: str) -> str:
        """Convert a locale-formatted number string to a plain float string.
        
        English-style: 25,000.50 → 25000.50
        Spanish-style: 25.000,50 → 25000.50
        """
        if lang in self.DOT_THOUSANDS_LANGS:
            # dot = thousands, comma = decimal
            num_str = num_str.replace(".", "").replace(",", ".")
        else:
            # comma = thousands, dot = decimal (English, Chinese, etc.)
            num_str = num_str.replace(",", "")
        return num_str
    
    def _num2words_lang(self, lang: str) -> str:
        """Map detected language to num2words language code."""
        mapping = {"en": "en", "es": "es", "fr": "fr", "de": "de", 
                   "it": "it", "pt": "pt_BR", "ru": "ru", "zh": "zh",
                   "ja": "ja", "ko": "ko"}
        return mapping.get(lang, "en")

    def process(self, text):
        """Full normalization pipeline.  Preserves newlines."""
        text = self.clean_text(text)
        lang = self.detect_language(text)
        
        # 1. Number & Currency Expansion
        n2w_lang = self._num2words_lang(lang)
        
        try:
            if lang in self.DOT_THOUSANDS_LANGS:
                text = re.sub(
                    r"[\$€£]([\d.]+(?:,\d+)?)",
                    lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang=n2w_lang)} {self._currency_word(m.group(0)[0], lang)}",
                    text
                )
                text = re.sub(
                    r"([\d.]+(?:,\d+)?)%",
                    lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang=n2w_lang)} {self._percent_word(lang)}",
                    text
                )
                text = re.sub(
                    r"\b(\d{1,3}(?:\.\d{3})+(?:,\d+)?)\b",
                    lambda m: num2words(self._normalize_number(m.group(0), lang), lang=n2w_lang),
                    text
                )
                text = re.sub(
                    r"\b(\d{2,}(?:,\d+)?)\b",
                    lambda m: num2words(self._normalize_number(m.group(0), lang), lang=n2w_lang),
                    text
                )
            elif lang == "zh":
                text = re.sub(r"\$([\d,]+(?:\.\d+)?)", lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang='zh')}美元", text)
                text = re.sub(r"￥([\d,]+(?:\.\d+)?)", lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang='zh')}元", text)
                text = re.sub(r"([\d,]+(?:\.\d+)?)%", lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang='zh')}百分之", text)
                text = re.sub(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", lambda m: num2words(self._normalize_number(m.group(0), lang), lang='zh'), text)
            else:
                text = re.sub(
                    r"\$([\d,]+(?:\.\d+)?)",
                    lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang=n2w_lang)} dollars",
                    text
                )
                text = re.sub(
                    r"([\d,]+(?:\.\d+)?)%",
                    lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang=n2w_lang)} percent",
                    text
                )
                text = re.sub(
                    r"\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b",
                    lambda m: num2words(self._normalize_number(m.group(0), lang), lang=n2w_lang),
                    text
                )
                text = re.sub(
                    r"\b(\d{2,}(?:\.\d+)?)\b",
                    lambda m: num2words(m.group(0), lang=n2w_lang),
                    text
                )
        except:
            pass

        # 2. Language-Specific Text Normalization
        if lang == "zh":
            try:
                text = "".join(self.zh_norm.normalize(text))
            except:
                pass

        # 3. Punctuation Hygiene
        text = re.sub(r'[—–]', ', ', text)        # em/en-dash → comma
        text = re.sub(r'\.{3,}', '...', text)     # normalize ellipsis
        text = re.sub(r'([.!?])\1+', r'\1', text) # dedup terminal punct

        # 4. Whitespace: collapse spaces/tabs but PRESERVE newlines.
        # The model tokenizer maps \n to Ċ (GPT-2 BPE byte 0x0A) — a prosody cue.
        text = re.sub(r'[^\S\n]+', ' ', text)     # spaces/tabs → single space
        text = re.sub(r' *\n *', '\n', text)       # clean space around newlines
        text = re.sub(r'\n{3,}', '\n\n', text)     # cap at double-newline
        text = text.strip()

        return text, lang
    
    def _currency_word(self, symbol: str, lang: str) -> str:
        """Return the spoken currency name for a symbol."""
        currencies = {
            "$": {"es": "dólares", "fr": "dollars", "de": "Dollar", "it": "dollari", "pt": "dólares", "ru": "долларов"},
            "€": {"es": "euros", "fr": "euros", "de": "Euro", "it": "euro", "pt": "euros", "ru": "евро"},
            "£": {"es": "libras", "fr": "livres", "de": "Pfund", "it": "sterline", "pt": "libras", "ru": "фунтов"},
        }
        return currencies.get(symbol, {}).get(lang, "dollars")
    
    def _percent_word(self, lang: str) -> str:
        """Return the spoken word for percent."""
        words = {"es": "por ciento", "fr": "pour cent", "de": "Prozent", 
                 "it": "per cento", "pt": "por cento", "ru": "процентов"}
        return words.get(lang, "percent")

    def split_sentences(self, text, lang=None):
        """Split text into sentences using pysbd.
        
        NOTE: For TTS chunking, use chunk_for_tts() instead.
        Kept for backward compatibility (transcript trimming in server.py).
        """
        if not lang:
            lang = self.detect_language(text)
        
        segmenter = self.segmenters.get(lang, self.segmenters["en"])
        return segmenter.segment(text)

    # --- TTS Chunking ---

    def chunk_for_tts(self, text, lang=None, max_chars=None):
        """Split text into chunks sized for model capacity.
        
        The model generates 4096 audio tokens at 12Hz = 341 seconds of audio.
        We only chunk when text exceeds that capacity (~3000 chars conservative).
        
        When we must split:
          1. Break at paragraph boundaries (\\n\\n)
          2. If a paragraph is still too long, break at sentence boundaries
        
        Newlines are preserved — the model uses them as prosody cues.
        """
        if max_chars is None:
            max_chars = MAX_CHUNK_CHARS

        if not text or not text.strip():
            return []
        
        if not lang:
            lang = self.detect_language(text)
        
        # Fits in one chunk — just pass it through
        if len(text) <= max_chars:
            return [self._ensure_terminal_punct(text)]
        
        # --- Split at paragraph boundaries (double newline) ---
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Greedily pack paragraphs into chunks
        chunks = []
        current = ""
        
        for para in paragraphs:
            if not current:
                current = para
            elif len(current) + 2 + len(para) <= max_chars:
                current = current + "\n\n" + para
            else:
                chunks.append(current)
                current = para
        
        if current:
            chunks.append(current)
        
        # If any chunk is still over max, split at sentence boundaries
        final = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final.append(chunk)
            else:
                final.extend(self._split_at_sentences(chunk, lang, max_chars))
        
        return [self._ensure_terminal_punct(c) for c in final]

    def _split_at_sentences(self, text, lang, max_chars):
        """Split oversized text at sentence boundaries."""
        segmenter = self.segmenters.get(lang, self.segmenters["en"])
        sentences = segmenter.segment(text)
        
        if not sentences:
            return [text]
        
        chunks = []
        current = ""
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            candidate = (current + " " + sent).strip() if current else sent
            
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = sent
            else:
                current = candidate
        
        if current:
            chunks.append(current)
        
        return chunks if chunks else [text]

    @staticmethod
    def _ensure_terminal_punct(text):
        """Ensure chunk ends with terminal punctuation.
        
        The model uses this as a completion cue.
        """
        text = text.rstrip()
        if text and text[-1] not in '.!?':
            text += '.'
        return text

# Singleton instance
normalizer = TextNormalizer()
