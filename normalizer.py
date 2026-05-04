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

    def process(self, text):
        """Full normalization pipeline using professional packages."""
        text = self.clean_text(text)
        lang = self.detect_language(text)
        
        # 1. Number & Currency Expansion (num2words is the industry standard here)
        try:
            if lang == "en":
                text = re.sub(r"\$(\d+(?:\.\d+)?)", lambda m: f"{num2words(m.group(1), lang='en')} dollars", text)
                text = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: f"{num2words(m.group(1), lang='en')} percent", text)
                # Standard numbers
                text = re.sub(r"\b\d+(?:\.\d+)?\b", lambda m: num2words(m.group(0), lang='en'), text)
            elif lang == "zh":
                text = re.sub(r"\$(\d+(?:\.\d+)?)", lambda m: f"{num2words(m.group(1), lang='zh')}美元", text)
                text = re.sub(r"￥(\d+(?:\.\d+)?)", lambda m: f"{num2words(m.group(1), lang='zh')}元", text)
                text = re.sub(r"(\d+(?:\.\d+)?)%", lambda m: f"{num2words(m.group(1), lang='zh')}百分之", text)
                # Standard numbers
                text = re.sub(r"\b\d+(?:\.\d+)?\b", lambda m: num2words(m.group(0), lang='zh'), text)
        except:
            pass

        # 2. Language-Specific Text Normalization
        if lang == "en":
            # Gruut handles abbreviations and complex English text expansion
            try:
                sents = []
                for sent in gruut_sentences(text, lang="en-us"):
                    sents.append(" ".join(word.text for word in sent))
                text = " ".join(sents)
            except:
                pass
        
        elif lang == "zh":
            # zh_normalization is the industry standard for Chinese TTS
            try:
                # zh_normalization.normalize returns a list of sentences
                text = "".join(self.zh_norm.normalize(text))
            except:
                pass
            
        return text, lang

    def split_sentences(self, text, lang=None):
        """Split text into sentences using pysbd."""
        if not lang:
            lang = self.detect_language(text)
        
        segmenter = self.segmenters.get(lang, self.segmenters["en"])
        return segmenter.segment(text)

# Singleton instance
normalizer = TextNormalizer()
