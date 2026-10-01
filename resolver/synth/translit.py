"""Script and spelling noise for Indian names.

`to_devanagari` writes a romanised name in Devanagari with a small greedy syllable parser. The
model later sees it through `anyascii`, which drops inherent vowels ("ganesh" -> "गणेश" -> "gnes"),
reproducing the non-Latin records of the real data. `phonetic_romanise` produces the other
pattern seen there: Indian spellings of English words ("sunrise technology" -> "sanraij teknolji").
"""
import random
import re

_CONS = [("ksh", "क्ष"), ("chh", "छ"), ("kh", "ख"), ("gh", "घ"), ("ch", "च"), ("jh", "झ"), ("th", "थ"),
         ("dh", "ध"), ("ph", "फ"), ("bh", "भ"), ("sh", "श"), ("k", "क"), ("g", "ग"), ("c", "क"), ("j", "ज"),
         ("t", "ट"), ("d", "ड"), ("n", "न"), ("p", "प"), ("b", "ब"), ("m", "म"), ("y", "य"), ("r", "र"),
         ("l", "ल"), ("v", "व"), ("w", "व"), ("s", "स"), ("h", "ह"), ("z", "ज़"), ("f", "फ़"), ("q", "क"),
         ("x", "क्स")]
_VOW = [("aa", "आ", "ा"), ("ai", "ऐ", "ै"), ("au", "औ", "ौ"), ("ee", "ई", "ी"), ("oo", "ऊ", "ू"),
        ("a", "अ", ""), ("i", "इ", "ि"), ("u", "उ", "ु"), ("e", "ए", "े"), ("o", "ओ", "ो")]
_VIRAMA = "्"
# Names of Indian origin keep dental t/d; English loanwords use retroflex (as Hindi spelling does).
_DENTAL = {"t": "त", "d": "द", "th": "थ", "dh": "ध"}
_INDIC = re.compile(r"(sharma|gupta|patel|singh|verma|mehta|shah|reddy|nath|datt|pandit|trivedi|tiwari|"
                    r"chaturvedi|dutta|aditya|amrit|anand|kalpataru|samruddhi|pratham|gayatri|siddhi|durga)")


def _word(w: str) -> str:
    out, i, prev_cons = [], 0, False
    dental = bool(_INDIC.search(w))
    while i < len(w):
        for lat, ind, matra in _VOW:
            if w.startswith(lat, i):
                if prev_cons:
                    # word-final short 'a' after a consonant is written long (sharma -> शर्मा)
                    out.append("ा" if lat == "a" and i + 1 == len(w) else matra)
                else:
                    out.append(ind)
                i += len(lat)
                prev_cons = False
                break
        else:
            for lat, ind in _CONS:
                if w.startswith(lat, i):
                    if prev_cons:
                        out.append(_VIRAMA)
                    out.append(_DENTAL.get(lat, ind) if dental else ind)
                    i += len(lat)
                    prev_cons = True
                    break
            else:                                   # digits, punctuation
                out.append(w[i])
                i += 1
                prev_cons = False
    return "".join(out)


# Conventional Hindi spellings of frequent words (the parser handles everything else).
_KNOWN = {"private": "प्राइवेट", "pvt": "प्राइवेट", "limited": "लिमिटेड", "ltd": "लिमिटेड", "llp": "एलएलपी",
          "shree": "श्री", "sri": "श्री", "shri": "श्री", "ganesh": "गणेश", "krishna": "कृष्णा", "laxmi": "लक्ष्मी",
          "technology": "टेक्नोलॉजी", "traders": "ट्रेडर्स", "enterprises": "एंटरप्राइजेज", "industries": "इंडस्ट्रीज",
          "electricals": "इलेक्ट्रिकल्स", "sons": "संस", "and": "एंड", "sweets": "स्वीट्स", "jewellers": "ज्वैलर्स",
          "medicals": "मेडिकल्स", "hardware": "हार्डवेयर", "textiles": "टेक्सटाइल्स", "motors": "मोटर्स"}


def to_devanagari(name: str) -> str:
    words = re.sub(r"[.&()]", " ", name.lower()).replace(" & ", " and ").split()
    return " ".join(_KNOWN.get(w) or (_word(w) if w.isalpha() else w) for w in words)


_PHONETIC = [("ph", "f"), ("ee", "i"), ("oo", "u"), ("ck", "k"), ("c", "k"), ("y", "i"), ("w", "v"),
             ("ou", "au"), ("ise", "aij"), ("ology", "olji"), ("tion", "shan"), ("th", "t"), ("x", "ks"),
             ("q", "k"), ("ate", "et"), ("ea", "i")]


def phonetic_romanise(name: str, rng: random.Random) -> str:
    """Apply 2-4 phonetic substitutions to the whole name ("sunrise" -> "sunraij")."""
    out = name.lower()
    for a, b in rng.sample(_PHONETIC, rng.randint(2, 4)):
        out = out.replace(a, b)
    return out
