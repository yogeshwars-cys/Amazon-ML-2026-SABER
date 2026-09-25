import re, unicodedata
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
BLOCKS=[(0x0900,sanscript.DEVANAGARI),(0x0980,sanscript.BENGALI),(0x0A00,sanscript.GURMUKHI),(0x0A80,sanscript.GUJARATI),
        (0x0B00,sanscript.ORIYA),(0x0B80,sanscript.TAMIL),(0x0C00,sanscript.TELUGU),(0x0C80,sanscript.KANNADA),(0x0D00,sanscript.MALAYALAM)]
def _script(ch):
    o=ord(ch)
    for b,s in BLOCKS:
        if b<=o<b+0x80: return s
    return None
_run=re.compile(r"[\u0900-\u0D7F\u200c\u200d]+")
def _translit(m):
    t=m.group(0).replace("\u200c","").replace("\u200d","")
    sc=next((_script(c) for c in t if _script(c)),None)
    if sc is None: return t
    out=transliterate(t,sc,sanscript.ITRANS)
    out=re.sub(r"(?<=[^aeiouAEIOU\s])a\b","",out)          # schwa deletion at word end
    out=out.replace("~N","n").replace(".N","n").replace("M","n").replace("JN","gy")
    return out
_LIG=str.maketrans({"œ":"oe","Œ":"OE","æ":"ae","Æ":"AE","ß":"ss","ø":"o","Ø":"O","ł":"l","Ł":"L","đ":"d","’":"'"})
def norm(s):
    if not s: return ""
    s=_run.sub(_translit,s)
    s=s.translate(_LIG)
    s=unicodedata.normalize("NFKD",s); s="".join(c for c in s if not unicodedata.combining(c))
    s=s.lower().replace("&"," and ")
    s=re.sub(r"[^a-z0-9/ ,.-]"," ",s)
    s=re.sub(r"(?<![a-z0-9])[-.,/]+|[-.,/]+(?![a-z0-9])"," ",s)
    return re.sub(r"\s+"," ",s).strip()
