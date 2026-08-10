"""
NLP preprocessing helpers.

Collects the reusable functions written in LAB 0 (file handling + regex) and
LAB 1 (NLTK text preprocessing) so they can be imported into later labs:

    from preprocessing import clean_text, tokenize, preprocess

NLTK resources are downloaded lazily (only when a function that needs them is
actually called), so importing this module stays fast and works offline.
"""

import math
import re
import string
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
from nltk.tokenize import word_tokenize as _nltk_word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# ----------------------------------------------------------------------
# NLTK setup (lazy)
# ----------------------------------------------------------------------

_NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
_nltk_ready = False

# cached objects, built on first use
_stop_words = None
_stemmer = None
_lemmatizer = None


def ensure_nltk_data(quiet=True):
    """Download the NLTK corpora the rest of this module needs (once)."""
    global _nltk_ready
    if _nltk_ready:
        return
    for package in _NLTK_PACKAGES:
        nltk.download(package, quiet=quiet)
    _nltk_ready = True


def get_stopwords(language="english"):
    """Return the NLTK stopword set for a language."""
    global _stop_words
    if _stop_words is None:
        ensure_nltk_data()
        _stop_words = set(stopwords.words(language))
    return _stop_words


def get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = PorterStemmer()
    return _stemmer


def get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        ensure_nltk_data()
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


# ----------------------------------------------------------------------
# LAB 0 - File handling
# ----------------------------------------------------------------------

def read_file(path, n_chars=None, encoding="utf-8"):
    """Read a whole file, or only the first n_chars characters."""
    with open(path, "r", encoding=encoding) as file:
        return file.read() if n_chars is None else file.read(n_chars)


def read_lines(path, keep_newlines=False, encoding="utf-8"):
    """Read a file as a list of lines (\\n kept only if keep_newlines=True)."""
    with open(path, "r", encoding=encoding) as file:
        lines = file.readlines()
    return lines if keep_newlines else [line.rstrip("\n") for line in lines]


def read_line(path, line_number, encoding="utf-8"):
    """Read one specific line, counted from 1."""
    lines = read_lines(path, encoding=encoding)
    return lines[line_number - 1]


def read_pdf(path):
    """Read all text from a PDF file, page by page (needs the pypdf package)."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def write_file(path, text, append=False, encoding="utf-8"):
    """Write a string to a file. append=True adds to the end instead."""
    with open(path, "a" if append else "w", encoding=encoding) as file:
        file.write(text)


def write_lines(path, lines, append=False, encoding="utf-8"):
    """Write a list of strings to a file, one per line."""
    with open(path, "a" if append else "w", encoding=encoding) as file:
        for line in lines:
            file.write(str(line) + "\n")


# ----------------------------------------------------------------------
# LAB 0 - Regex extraction and validation
# ----------------------------------------------------------------------

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.-]+"
PHONE_PATTERN = r"(?:\+91[-\s]?)?\b\d{10}\b"
URL_PATTERN = r"(?:https?://)?(?:www\.)?\S+\.(?:com|in|org|net)\S*"
PROFILE_PATTERN = r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/[\w\-./]+"

# domain may have any number of sub-levels, so name@nmims.edu.in is accepted
VALID_EMAIL_PATTERN = r"^[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}$"


def is_valid_email(email):
    """True if the string is a properly formatted email address."""
    return bool(re.match(VALID_EMAIL_PATTERN, email))


def extract_emails(text):
    """All email addresses in the text."""
    return re.findall(EMAIL_PATTERN, text)


def extract_phones(text):
    """All 10 digit phone numbers (optional +91 country code)."""
    return re.findall(PHONE_PATTERN, text)


def extract_urls(text):
    """All URLs in the text."""
    return re.findall(URL_PATTERN, text)


def extract_first(text, pattern, default="Not found", group=0, flags=0):
    """First regex match in the text, or default if there is none."""
    match = re.search(pattern, text, flags)
    return match.group(group) if match else default


# ----------------------------------------------------------------------
# LAB 0 - Regex based normalisation and tokenisation
# ----------------------------------------------------------------------

def remove_urls(text, replacement=" "):
    return re.sub(URL_PATTERN, replacement, text)


def remove_emails(text, replacement=" "):
    return re.sub(r"[\w.+-]+@[\w.-]+", replacement, text)


def remove_numbers(text, replacement=" "):
    return re.sub(r"\d+", replacement, text)


def collapse_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text, lowercase=True):
    """
    Normalise raw text with regex substitutions.

    Order matters: URLs and emails are stripped first, because once the dots
    and slashes are gone there is nothing left to recognise them by.
    """
    text = remove_urls(text)
    text = remove_emails(text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)   # drop digits and punctuation
    text = collapse_whitespace(text)
    return text.lower() if lowercase else text


def tokenize(text):
    """Regex word tokeniser (expects text already lowercased by clean_text)."""
    return re.findall(r"[a-z]+", text)


# ----------------------------------------------------------------------
# LAB 1 - Level one preprocessing operations
# ----------------------------------------------------------------------

def to_lower(text):
    return text.lower()


def to_upper(text):
    return text.upper()


def sentence_tokenize(text):
    """Split text into sentences using NLTK."""
    ensure_nltk_data()
    return _nltk_sent_tokenize(text)


def word_tokenize(text):
    """Split text into word tokens using NLTK."""
    ensure_nltk_data()
    return _nltk_word_tokenize(text)


def remove_punctuation(words):
    """Strip punctuation off each token and drop tokens that become empty."""
    stripped = [w.strip(string.punctuation) for w in words]
    return [w for w in stripped if w]


def remove_punctuation_text(text):
    """Remove every punctuation character from a raw string."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_stopwords(words, language="english", extra=None):
    """Drop stopwords from a token list. extra adds more words to remove."""
    stop_words = set(get_stopwords(language))
    if extra:
        stop_words |= {w.lower() for w in extra}
    return [w for w in words if w.lower() not in stop_words]


def stem_words(words):
    """Porter stemming - fast, rule based, output may not be a real word."""
    stemmer = get_stemmer()
    return [stemmer.stem(w) for w in words]


def lemmatize_words(words, pos="n"):
    """WordNet lemmatisation - slower, dictionary based, output is a real word."""
    lemmatizer = get_lemmatizer()
    return [lemmatizer.lemmatize(w, pos=pos) for w in words]


# ----------------------------------------------------------------------
# LAB 1 - Tokenisation without built-in NLTK functions
# ----------------------------------------------------------------------

def custom_sent_tokenize(text):
    """Split after a sentence ending mark (. ! ?) followed by whitespace."""
    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s]


def custom_word_tokenize(text):
    """
    Match words (keeping contractions like "it's" together) or a single
    punctuation character, as separate tokens.
    """
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


# ----------------------------------------------------------------------
# LAB 1 - Word frequency
# ----------------------------------------------------------------------

def word_frequency(words):
    """Counter of how often each token appears."""
    return Counter(words)


def most_common_words(words, n=10):
    """Top n (word, count) pairs."""
    return word_frequency(words).most_common(n)


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------

def preprocess(text,
               lowercase=True,
               use_custom_tokenizer=False,
               strip_punctuation=True,
               drop_stopwords=True,
               stem=False,
               lemmatize=False,
               extra_stopwords=None,
               language="english"):
    """
    Run the standard LAB 1 pipeline and return a list of tokens.

    lowercase -> tokenize -> remove punctuation -> remove stopwords
              -> stem and/or lemmatize
    """
    if lowercase:
        text = text.lower()

    words = custom_word_tokenize(text) if use_custom_tokenizer else word_tokenize(text)

    if strip_punctuation:
        words = remove_punctuation(words)
    if drop_stopwords:
        words = remove_stopwords(words, language=language, extra=extra_stopwords)
    if stem:
        words = stem_words(words)
    if lemmatize:
        words = lemmatize_words(words)

    return words


def preprocess_regex(text):
    """Lighter pipeline with no NLTK dependency: clean_text then tokenize."""
    return tokenize(clean_text(text))


# ----------------------------------------------------------------------
# LAB 2 - Label encoding and one hot encoding (no sklearn)
# ----------------------------------------------------------------------

def preprocess_documents(documents, **kwargs):
    """Run preprocess() on every document and return a list of token lists."""
    return [preprocess(document, **kwargs) for document in documents]


def build_vocabulary(tokens):
    """Sorted list of the unique tokens. Sorting keeps the ids reproducible."""
    return sorted(set(tokens))


def build_vocab_index(tokens):
    """Map each unique token to an integer id: {word: id}."""
    return {word: index for index, word in enumerate(build_vocabulary(tokens))}


def label_encode(tokens, vocab_index=None):
    """
    Replace every token by its integer id.

    One number per token, so the output is as long as the input. The ids are
    only labels - a larger id does not mean a "bigger" word.
    """
    if vocab_index is None:
        vocab_index = build_vocab_index(tokens)
    return [vocab_index[token] for token in tokens]


def label_decode(codes, vocab_index):
    """Turn integer ids back into words."""
    id_to_word = {index: word for word, index in vocab_index.items()}
    return [id_to_word[code] for code in codes]


def one_hot_vector(token, vocab_index):
    """Vector of zeros with a single 1 at the token's position in the vocabulary."""
    vector = [0] * len(vocab_index)
    vector[vocab_index[token]] = 1
    return vector


def one_hot_encode(tokens, vocab_index=None):
    """One hot vector per token, so the result is a len(tokens) x vocab matrix."""
    if vocab_index is None:
        vocab_index = build_vocab_index(tokens)
    return [one_hot_vector(token, vocab_index) for token in tokens]


def one_hot_document(tokens, vocab_index):
    """One binary vector per document, marking every vocabulary word present in it."""
    vector = [0] * len(vocab_index)
    for token in tokens:
        if token in vocab_index:
            vector[vocab_index[token]] = 1
    return vector


def one_hot_documents(documents_tokens, vocab_index=None):
    """Document x vocabulary binary matrix, one row per document."""
    if vocab_index is None:
        all_tokens = [token for tokens in documents_tokens for token in tokens]
        vocab_index = build_vocab_index(all_tokens)
    return [one_hot_document(tokens, vocab_index) for tokens in documents_tokens]


# ----------------------------------------------------------------------
# LAB 2 - Bag of Words (no sklearn)
# ----------------------------------------------------------------------

def bag_of_words(tokens, vocab_index):
    """
    Count vector for one document.

    Same shape as one_hot_document(), but it keeps counting instead of stopping
    at 1, so a word used three times scores 3.
    """
    vector = [0] * len(vocab_index)
    for token in tokens:
        if token in vocab_index:
            vector[vocab_index[token]] += 1
    return vector


def bag_of_words_matrix(documents_tokens, vocab_index=None):
    """Document x vocabulary count matrix, one row per document."""
    if vocab_index is None:
        all_tokens = [token for tokens in documents_tokens for token in tokens]
        vocab_index = build_vocab_index(all_tokens)
    return [bag_of_words(tokens, vocab_index) for tokens in documents_tokens]


def document_frequency(documents_tokens):
    """How many documents each word appears in (not how many times in total)."""
    counts = Counter()
    for tokens in documents_tokens:
        counts.update(set(tokens))
    return counts


def sparsity(matrix):
    """Fraction of the matrix that is zero - the main cost of these representations."""
    cells = [value for row in matrix for value in row]
    return cells.count(0) / len(cells) if cells else 0.0


# ----------------------------------------------------------------------
# LAB 2 - TF-IDF (no sklearn)
# ----------------------------------------------------------------------

def term_frequency(tokens, vocab_index, normalize=True):
    """
    TF vector for one document.

    Raw counts (the BoW row) divided by the document length, so a word used
    twice in a 20 word paragraph does not outrank the same word used twice in a
    200 word one. normalize=False keeps the plain counts, which is the tf that
    sklearn's TfidfVectorizer feeds into its formula.
    """
    counts = bag_of_words(tokens, vocab_index)
    total = sum(counts)
    if not normalize or total == 0:
        return [float(count) for count in counts]
    return [count / total for count in counts]


def term_frequency_matrix(documents_tokens, vocab_index=None, normalize=True):
    """Document x vocabulary TF matrix, one row per document."""
    if vocab_index is None:
        all_tokens = [token for tokens in documents_tokens for token in tokens]
        vocab_index = build_vocab_index(all_tokens)
    return [term_frequency(tokens, vocab_index, normalize) for tokens in documents_tokens]


def inverse_document_frequency(documents_tokens, vocab_index=None, scheme="standard"):
    """
    IDF value for every vocabulary word, as a list indexed by word id.

    A word in every document carries no information, so it should score near
    zero; a word in one document only should score high.

        scheme="standard" : log(N / df)             - the textbook formula
        scheme="smooth"   : log((1 + N)/(1 + df)) + 1  - what sklearn uses

    The smooth version pretends there is one extra document containing every
    word (so df is never 0) and adds 1 at the end, which stops a word present in
    all N documents from being wiped out entirely.
    """
    if vocab_index is None:
        all_tokens = [token for tokens in documents_tokens for token in tokens]
        vocab_index = build_vocab_index(all_tokens)

    n_docs = len(documents_tokens)
    doc_freq = document_frequency(documents_tokens)

    idf = [0.0] * len(vocab_index)
    for word, index in vocab_index.items():
        df = doc_freq.get(word, 0)
        if scheme == "smooth":
            idf[index] = math.log((1 + n_docs) / (1 + df)) + 1
        else:
            idf[index] = math.log(n_docs / df) if df else 0.0
    return idf


def l2_normalize(vector):
    """Scale a vector to unit length, so long and short documents stay comparable."""
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else list(vector)


def tfidf_vector(tokens, vocab_index, idf, normalize_tf=True, normalize=False):
    """TF-IDF vector for one document: its TF row multiplied element wise by IDF."""
    tf = term_frequency(tokens, vocab_index, normalize=normalize_tf)
    weights = [t * i for t, i in zip(tf, idf)]
    return l2_normalize(weights) if normalize else weights


def tfidf_matrix(documents_tokens,
                 vocab_index=None,
                 scheme="standard",
                 normalize_tf=True,
                 normalize=False):
    """
    Document x vocabulary TF-IDF matrix.

    Defaults give the textbook version: TF as a within document proportion and
    IDF as log(N / df). Passing scheme="smooth", normalize_tf=False and
    normalize=True reproduces sklearn's TfidfVectorizer exactly.
    """
    if vocab_index is None:
        all_tokens = [token for tokens in documents_tokens for token in tokens]
        vocab_index = build_vocab_index(all_tokens)

    idf = inverse_document_frequency(documents_tokens, vocab_index, scheme=scheme)
    return [tfidf_vector(tokens, vocab_index, idf, normalize_tf, normalize)
            for tokens in documents_tokens]


def top_terms(vector, vocabulary, n=5):
    """
    The n highest scoring words in a document vector, as (word, score) pairs.

    Ties are broken alphabetically so the output is reproducible.
    """
    scored = [(vocabulary[index], score)
              for index, score in enumerate(vector) if score > 0]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:n]


# ----------------------------------------------------------------------
# LAB 1 - Rule based keyword classification
# ----------------------------------------------------------------------

def keyword_scores(document, category_keywords):
    """How many times each category's keywords appear in the document."""
    document = document.lower()
    return {
        category: sum(document.count(keyword) for keyword in keywords)
        for category, keywords in category_keywords.items()
    }


def rule_based_classify(document, category_keywords, fallback=None):
    """
    Pick the category whose keywords appear most often in the document.
    If nothing matches at all, return the fallback category.
    """
    scores = keyword_scores(document, category_keywords)
    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return fallback
    return best_category


# ----------------------------------------------------------------------
# LAB 0 - Resume screening helpers (application specific)
# ----------------------------------------------------------------------

def parse_job_description(path):
    """Read a job description line by line and pull out the requirements."""
    requirements = {}
    for line in read_lines(path):
        line = line.strip()
        if re.match(r"^Job Role:", line, re.IGNORECASE):
            requirements["role"] = re.sub(r"^Job Role:\s*", "", line, flags=re.IGNORECASE)
        elif re.match(r"^Required Skills:", line, re.IGNORECASE):
            value = re.sub(r"^Required Skills:\s*", "", line, flags=re.IGNORECASE)
            requirements["skills"] = [s.strip().lower() for s in value.split(",")]
        elif re.match(r"^Minimum CGPA:", line, re.IGNORECASE):
            requirements["min_cgpa"] = float(re.search(r"\d+\.?\d*", line).group())
        elif re.match(r"^Minimum Graduation Year:", line, re.IGNORECASE):
            requirements["min_year"] = int(re.search(r"20\d{2}", line).group())
    return requirements


def extract_details(text):
    """Pull name, email, phone, profile, CGPA and graduation year out of a resume."""
    details = {}
    lines = text.strip().split("\n")
    details["name"] = lines[0].strip()

    details["email"] = extract_first(text, r"[\w.+-]+@\S+")
    details["email_valid"] = is_valid_email(details["email"])
    details["phone"] = extract_first(text, PHONE_PATTERN)
    details["profile"] = extract_first(text, PROFILE_PATTERN)

    cgpa = re.search(r"CGPA\s*:?\s*(\d+\.?\d*)", text, re.IGNORECASE)
    details["cgpa"] = float(cgpa.group(1)) if cgpa else 0.0

    year = re.search(r"Graduation\s*:?\s*(20\d{2})", text, re.IGNORECASE)
    details["year"] = int(year.group(1)) if year else 0

    return details


def score_resume(text, skills):
    """
    Score text against a list of required skills.

    Single word skills are matched against the token set (exact match, so
    "sql" does not match inside "mysql"). Multi word skills are matched as a
    substring of the cleaned text, since a set of single tokens cannot hold them.
    """
    cleaned = clean_text(text)
    tokens = set(tokenize(cleaned))
    matched = []
    for skill in skills:
        if " " in skill:
            if skill in cleaned:
                matched.append(skill)
        elif skill in tokens:
            matched.append(skill)
    missing = [s for s in skills if s not in matched]
    percent = round(len(matched) / len(skills) * 100, 2) if skills else 0.0
    return percent, matched, missing


def screen(files, requirements, min_match_percent=50):
    """Screen a list of resume files and return them ranked by skill match."""
    results = []
    for path in files:
        text = read_file(path)

        details = extract_details(text)
        percent, matched, missing = score_resume(text, requirements["skills"])

        reasons = []
        if details["cgpa"] < requirements.get("min_cgpa", 0):
            reasons.append("CGPA below cutoff")
        if details["year"] < requirements.get("min_year", 0):
            reasons.append("graduated before cutoff year")
        if not details["email_valid"]:
            reasons.append("invalid email format")
        if percent < min_match_percent:
            reasons.append(f"skill match below {min_match_percent} percent")

        details["file"] = path
        details["score"] = percent
        details["matched"] = matched
        details["missing"] = missing
        details["reasons"] = reasons
        details["status"] = "SHORTLISTED" if not reasons else "REJECTED"
        results.append(details)

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


# ----------------------------------------------------------------------
# LAB 3 - NLTK corpus access
# ----------------------------------------------------------------------

def ensure_corpus(name, quiet=True):
    """Download an NLTK corpus by name (safe to call more than once)."""
    nltk.download(name, quiet=quiet)


def text_statistics(fileid):
    """
    Basic statistics for one Gutenberg text, using the corpus reader's
    raw() (characters), words() (tokens) and sents() (sentences) views.

    TTR (Type Token Ratio) = unique words / total words. Words are lowercased
    first so "The" and "the" count as one type.
    """
    ensure_corpus("gutenberg")
    ensure_nltk_data()   # sents() needs the punkt sentence tokenizer
    from nltk.corpus import gutenberg

    raw = gutenberg.raw(fileid)
    words = gutenberg.words(fileid)
    sents = gutenberg.sents(fileid)

    n_chars = len(raw)
    n_words = len(words)
    n_sents = len(sents)
    n_types = len(set(word.lower() for word in words))

    return {
        "text": fileid,
        "words": n_words,
        "characters": n_chars,
        "sentences": n_sents,
        "avg_chars_per_word": round(n_chars / n_words, 2),
        "ttr": round(n_types / n_words, 4),
    }


def brown_category_statistics(category, top=5):
    """
    Compare one Brown corpus category (genre) against the others:
    total words, total sentences, and the most common words found by
    nltk.FreqDist() (a counter of how often every word appears).
    """
    ensure_corpus("brown")
    from nltk.corpus import brown

    words = brown.words(categories=category)
    freq_dist = nltk.FreqDist(words)

    return {
        "category": category,
        "words": len(words),
        "sentences": len(brown.sents(categories=category)),
        "most_common": freq_dist.most_common(top),
    }
