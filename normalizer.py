import re
import langid
import pysbd
from gruut import sentences as gruut_sentences
from zh_normalization import TextNormalizer as ZhNormalizer
from num2words import num2words
from cleantext import clean

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
        """Basic cleanup: whitespace, unicode, etc."""
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
            normalize_whitespace=True,
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
        """Full normalization pipeline using professional packages."""
        text = self.clean_text(text)
        lang = self.detect_language(text)
        
        # 1. Number & Currency Expansion
        # Qwen3-TTS handles most text natively, but struggles with raw digits.
        # Number formats vary by locale:
        #   English:  25,000.50  (comma=thousands, dot=decimal)
        #   Spanish:  25.000,50  (dot=thousands, comma=decimal)
        n2w_lang = self._num2words_lang(lang)
        
        try:
            if lang in self.DOT_THOUSANDS_LANGS:
                # Dot-thousands locale (es, fr, de, it, pt, ru)
                # Currency: $25.000,50 or €25.000,50
                text = re.sub(
                    r"[\$€£]([\d.]+(?:,\d+)?)",
                    lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang=n2w_lang)} {self._currency_word(m.group(0)[0], lang)}",
                    text
                )
                # Percentages: 99,5%
                text = re.sub(
                    r"([\d.]+(?:,\d+)?)%",
                    lambda m: f"{num2words(self._normalize_number(m.group(1), lang), lang=n2w_lang)} {self._percent_word(lang)}",
                    text
                )
                # Numbers with dot-thousands: 25.000 / 1.000.000
                text = re.sub(
                    r"\b(\d{1,3}(?:\.\d{3})+(?:,\d+)?)\b",
                    lambda m: num2words(self._normalize_number(m.group(0), lang), lang=n2w_lang),
                    text
                )
                # Plain numbers (2+ digits): 42 → cuarenta y dos
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
                # English and other comma-thousands locales
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
        # Note: gruut removed — Qwen3-TTS handles English abbreviations natively.
            
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
        """Split text into sentences using pysbd."""
        if not lang:
            lang = self.detect_language(text)
        
        segmenter = self.segmenters.get(lang, self.segmenters["en"])
        return segmenter.segment(text)

# Singleton instance
normalizer = TextNormalizer()
