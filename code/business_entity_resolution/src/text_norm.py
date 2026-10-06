"""Text normalisation for business names and addresses (vectorised with polars).

Design notes (from EDA):
* Indic-script names are word-by-word transliterations of the English name, so a
  dictionary learned from training pairs (learn_translit.py) maps them back
  exactly; anyascii is the fallback for unseen tokens.
* Accents are pure noise ("Límited", "Fóod") -> NFKD + drop combining marks.
* Legal forms, honorifics (mr/dr/sri/shri/smt), alias markers (dba/aka/fka) and
  filler words (center/services/the) are injected by the noise generator, so they
  are canonicalised and removed from the "core" name.
* Addresses: abbreviations are canonicalised to one short form (street->st,
  rue->r, boulevard/bd->blvd ...), ordinals/leading zeros stripped, digits split
  from letters ("419Bonny" -> "419 bonny", "218C" -> "218 c"), null tokens removed.
  Country-specific state tables are applied only where the country label matches;
  any other country (e.g. France, unseen in training) goes through the generic path.
"""
import json
import os
import re

import polars as pl
from anyascii import anyascii

from config import WORK

INDIC_RE = re.compile(r"[ऀ-෿]")
INDIC_PL = r"[ऀ-෿]"

# ----------------------------------------------------------------------------- names
NAME_CANON = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "incorporated": "inc", "inc": "inc", "lnc": "inc",
    "llc": "llc", "lc": "llc",
    "centre": "center", "ctr": "center",
    "service": "services", "svcs": "services", "svc": "services",
    "shree": "shree", "sree": "shree",
    "and": "and", "et": "and", "und": "and",
    "intl": "international", "int": "international",
    "mfg": "manufacturing", "tech": "tech", "techs": "tech",
    "bros": "brothers", "assoc": "associates", "assocs": "associates",
    "mgmt": "management", "dept": "department", "univ": "university",
    "ets": "etablissements", "etablissement": "etablissements",
    "ste": "societe", "sté": "societe", "societe": "societe", "soc": "societe",
}
LEGAL = {"pvt", "ltd", "corp", "co", "inc", "llc", "lp", "llp", "pc", "plc", "pllc", "lllp", "ltda",
         "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "scp", "selarl", "sca", "scs", "ei", "eirl",
         "gmbh", "ag", "bv", "nv", "pty", "public", "opc", "huf"}
HONORIFIC = {"mr", "mrs", "ms", "dr", "smt", "sri", "shri", "messrs", "ms", "m/s", "the"}
ALIAS = {"dba", "aka", "fka", "formerly", "known", "doing", "as", "also"}
FILLER = {"center", "services", "and", "of", "associes"}
NAME_STOP = LEGAL | HONORIFIC | ALIAS | FILLER
# France only (unseen in training): mined from the test inputs themselves (token substitutions in
# confident pairs) - the generator injects "& Fils", "Groupe", "Developpement", "France" the way it
# injects "Services"/"Center" elsewhere, and abbreviates Freres -> Frs, Saint -> St.
FR_NAME_CANON = {"frs": "freres", "st": "saint"}
FR_FILLER = {"fils", "groupe", "developpement", "france"}
FR_ADDR_CANON = {"bis": "b", "ter": "t", "psg": "pass", "app": "apt"}
# Disabled: a public-leaderboard comparison showed that these France-only rules lower the score
# (0.97666 -> 0.97591) - the injected French words also occur on planted look-alikes. Kept for reference.
APPLY_FR_RULES = False

# ----------------------------------------------------------------------------- addresses
ADDR_CANON = {
    # street types (EN)
    "street": "st", "str": "st", "st": "st", "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "road": "rd", "rd": "rd", "drive": "dr", "dr": "dr", "drv": "dr", "lane": "ln", "ln": "ln",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd", "bvd": "blvd",
    "court": "ct", "ct": "ct", "place": "pl", "pl": "pl", "circle": "cir", "cir": "cir",
    "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy", "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "square": "sq", "sq": "sq", "point": "pt", "pt": "pt",
    "expressway": "expy", "freeway": "fwy", "heights": "hts", "junction": "jct", "mount": "mt",
    "mountain": "mtn", "route": "rte", "rte": "rte", "rt": "rte", "turnpike": "tpke", "crossing": "xing",
    "alley": "aly", "creek": "crk", "crescent": "cres", "grove": "grv", "lake": "lk", "landing": "lndg",
    "manor": "mnr", "meadows": "mdws", "plaza": "plz", "ridge": "rdg", "springs": "spgs",
    "valley": "vly", "view": "vw", "village": "vlg", "vill": "vlg", "vil": "vlg", "cove": "cv",
    "canyon": "cyn", "glen": "gln", "harbor": "hbr", "hollow": "holw", "station": "sta",
    "extension": "extn", "extn": "extn", "ext": "extn", "gardens": "gdns", "estates": "ests",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # units
    "suite": "ste", "ste": "ste", "apartment": "apt", "apt": "apt", "appartement": "apt", "appt": "apt",
    "apts": "apt", "floor": "fl", "flr": "fl", "fl": "fl", "etage": "fl", "building": "bldg",
    "bldg": "bldg", "bldng": "bldg", "batiment": "bldg", "bat": "bldg", "room": "rm", "rm": "rm",
    # FR street types
    "rue": "r", "r": "r", "allee": "all", "all": "all", "chemin": "ch", "che": "ch", "chem": "ch",
    "impasse": "imp", "imp": "imp", "quai": "quai", "qu": "quai", "cours": "crs", "crs": "crs",
    "faubourg": "fbg", "fbg": "fbg", "fg": "fbg", "residence": "res", "res": "res",
    "saint": "st", "sainte": "ste", "passage": "pass", "lotissement": "lot", "hameau": "ham",
    "rond": "rpt", "rpt": "rpt", "promenade": "prom", "esplanade": "esp", "chaussee": "chs",
    # IN
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "sector": "sec", "sec": "sec",
    "block": "blk", "blk": "blk", "phase": "ph", "ph": "ph", "district": "dist", "dist": "dist",
    "distt": "dist", "tehsil": "tq", "teh": "tq", "taluk": "tq", "taluka": "tq", "tal": "tq", "tq": "tq",
    "colony": "colony", "col": "colony", "layout": "layout", "lyt": "layout", "industrial": "indl",
    "indl": "indl", "complex": "complex", "cmplx": "complex", "shop": "shop", "shp": "shop",
    "flat": "flat", "flt": "flat", "falt": "flat", "plot": "plot", "plt": "plot",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "ground": "gf", "gf": "gf",
    # alternate city spellings (transliteration variants)
    "bengaluru": "bangalore", "calcutta": "kolkata", "bombay": "mumbai", "madras": "chennai",
    "gurugram": "gurgaon", "trivandrum": "thiruvananthapuram", "visakhapatnam": "vishakhapatnam",
    "poona": "pune", "mysuru": "mysore", "baroda": "vadodara", "cochin": "kochi",
    "pondicherry": "puducherry", "mangaluru": "mangalore", "belagavi": "belgaum", "orissa": "odisha",
    "keralam": "kerala", "ahmadabad": "ahmedabad",
}
ADDR_DROP = {"no", "nos", "hn", "hno", "number", "null", "na", "n/a", "nil", "none", "unknown", "numero", "nº", "cdp"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "ladakh": "la",
    "orissa": "od", "keralam": "kl",
}
STATE_TABLES = {"US": US_STATES, "India": IN_STATES}

_TRANSLIT = None


def load_translit():
    global _TRANSLIT
    if _TRANSLIT is None:
        with open(os.path.join(WORK, "translit.json"), encoding="utf-8") as f:
            d = json.load(f)
        d["addr"] = {k: v for k, v in d["addr"].items() if '"' not in k}
        _TRANSLIT = d
    return _TRANSLIT


def translit_name(s):
    m = load_translit()["name"]
    out = []
    for t in s.split():
        if INDIC_RE.search(t):
            v = m.get(t) or m.get(t.strip(".,()"))
            out.append(v if v is not None else anyascii(t))
        else:
            out.append(t)
    return " ".join(out)


def translit_addr(s):
    m = load_translit()["addr"]
    out = []
    for seg in s.split(","):
        st = seg.strip()
        if INDIC_RE.search(st):
            v = m.get(st)
            out.append(v if v is not None else anyascii(st))
        else:
            out.append(seg)
    return ",".join(out)


def _fold(e):
    """Unicode fold: NFKD, drop combining marks, map a few special letters, lowercase."""
    e = e.str.normalize("NFKD").str.replace_all(r"\p{M}", "")
    for a, b in [("ß", "ss"), ("æ", "ae"), ("Æ", "ae"), ("ø", "o"), ("Ø", "o"), ("ł", "l"), ("Ł", "l"),
                 ("đ", "d"), ("ı", "i"), ("œ", "oe"), ("Œ", "oe"), ("’", "'"), ("`", "'"), ("´", "'")]:
        e = e.str.replace_all(a, b, literal=True)
    return e.str.to_lowercase()


def _map_tokens(tok_expr, mapping):
    keys = list(mapping.keys())
    vals = [mapping[k] for k in keys]
    return tok_expr.list.eval(pl.element().replace(keys, vals))


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """df: entity_id, business_name, business_address, country -> normalised columns."""
    # --- Indic transliteration (python, only rows that need it)
    name = df["business_name"].fill_null("")
    addr = df["business_address"].fill_null("")
    has_ind_n = name.str.contains(INDIC_PL)
    has_ind_a = addr.str.contains(INDIC_PL)
    if has_ind_n.any():
        idx = has_ind_n.arg_true()
        vals = [translit_name(s) for s in name.gather(idx).to_list()]
        name = name.scatter(idx, pl.Series(vals, dtype=pl.String))
    if has_ind_a.any():
        idx = has_ind_a.arg_true()
        vals = [translit_addr(s) for s in addr.gather(idx).to_list()]
        addr = addr.scatter(idx, pl.Series(vals, dtype=pl.String))
    df = df.with_columns(name.alias("_n"), addr.alias("_a"), has_ind_n.alias("name_indic"))

    # --- names
    n = _fold(pl.col("_n"))
    n = n.str.replace_all(r"&", " and ", literal=True)
    n = n.str.replace_all(r"\+", " ")
    n = n.str.replace_all(r"#\s*\d+", " ")
    n = n.str.replace_all(r"\bm\s*/\s*s\b", " ")
    n = n.str.replace_all(r"\bd\s*/\s*b\s*/\s*a\b", " dba ")
    n = n.str.replace_all(r"https?://|www\.", " ")
    is_domain = n.str.contains(r"[a-z0-9]\.(com|net|org|in|fr|biz|info|io|co|us)\b")
    n = n.str.replace_all(r"([a-z0-9])\.(com|net|org|in|fr|biz|info|io|co|us)\b", "$1 ")
    n = n.str.replace_all(r"\b([a-z])\.", "$1")          # L.L.C. -> llc, P.C. -> pc
    n = n.str.replace_all(r"'", "")                          # eve's -> eves
    n = n.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    df = df.with_columns(is_domain.alias("is_domain"), n.alias("_n2"))
    # OCR-style digit noise inside words (learned from training pairs: 0>o 1>l 5>s 8>b 6>g),
    # applied only to tokens mixing letters and digits ("financia1", "5hree", "a6ro")
    ocr = pl.element()
    for d, l in (("0", "o"), ("1", "l"), ("5", "s"), ("8", "b"), ("6", "g")):
        ocr = ocr.str.replace_all(d, l, literal=True)
    mixed = pl.element().str.contains(r"[a-z]") & pl.element().str.contains(r"[0-9]")
    toks = pl.col("_n2").str.split(" ").list.eval(pl.element().filter(pl.element() != ""))
    toks = toks.list.eval(pl.when(mixed).then(ocr).otherwise(pl.element()))
    df = df.with_columns(_map_tokens(toks, NAME_CANON).alias("name_tok"))
    is_fr = (pl.col("country") == "France") & pl.lit(APPLY_FR_RULES)
    df = df.with_columns(pl.when(is_fr).then(_map_tokens(pl.col("name_tok"), FR_NAME_CANON)).otherwise(pl.col("name_tok")).alias("name_tok"))
    stop = list(NAME_STOP)
    fr_stop = list(NAME_STOP | FR_FILLER)
    df = df.with_columns(
        pl.when(is_fr).then(pl.col("name_tok").list.eval(pl.element().filter(~pl.element().is_in(fr_stop))))
          .otherwise(pl.col("name_tok").list.eval(pl.element().filter(~pl.element().is_in(stop)))).alias("name_core_tok"),
        pl.col("name_tok").list.eval(pl.element().filter(pl.element().is_in(list(LEGAL)))).list.unique().list.sort().list.join(" ").alias("legal"),
        pl.col("name_tok").list.eval(pl.element().is_in(list(ALIAS - {"as", "also"}))).list.any().alias("has_alias"),
    )
    # if the core would be empty (name made only of legal words etc.), fall back to all tokens
    df = df.with_columns(pl.when(pl.col("name_core_tok").list.len() == 0).then(pl.col("name_tok")).otherwise(pl.col("name_core_tok")).alias("name_core_tok"))
    df = df.with_columns(
        pl.col("name_tok").list.join(" ").alias("name_full"),
        pl.col("name_core_tok").list.join(" ").alias("name_core"),
        pl.col("name_core_tok").list.join("").alias("name_join"),
        pl.col("name_core_tok").list.sort().list.join(" ").alias("name_sorted"),
    )

    # --- addresses
    a = _fold(pl.col("_a"))
    a = a.str.replace_all(r"<null>|\bn/a\b|\bnull\b", " ")
    # injected mailbox components ("PO Box 8046", "PMB 9531") and French "N°" prefixes
    a = a.str.replace_all(r"\b(p\s*\.?\s*o\s*\.?\s*box|box|pmb)\s*#?\s*\d+\b", " ")
    a = a.str.replace_all(r"\bn\s*°", " ")
    a = a.str.replace_all(r"\bc\s*/\s*o\b", " co ")
    a = a.str.replace_all(r"\b1\s*/\s*2\b", " ")
    a = a.str.replace_all(r"\b(\d+)(st|nd|rd|th|er|eme)\b", "$1")
    # hyphens injected inside numbers ("554" -> "5-54", "103" -> "1-03"): re-join digit-digit
    a = a.str.replace_all(r"(\d)-(\d)", "$1$2").str.replace_all(r"(\d)-(\d)", "$1$2")
    a = a.str.replace_all(r"(\d)([a-z])", "$1 $2").str.replace_all(r"([a-z])(\d)", "$1 $2")
    a = a.str.replace_all(r"[^a-z0-9]+", " ")
    a = a.str.replace_all(r"\b0+(\d)", "$1")
    a = a.str.strip_chars()
    df = df.with_columns(a.alias("_a2"))
    # country-specific state names -> codes (only where the country label matches;
    # any other country label goes through the generic path unchanged)
    for ctry, table in STATE_TABLES.items():
        expr = pl.col("_a2")
        for full in [k for k in table if " " in k]:
            expr = expr.str.replace_all(r"\b" + full + r"\b", table[full])
        df = df.with_columns(pl.when(pl.col("country") == ctry).then(expr).otherwise(pl.col("_a2")).alias("_a2"))
    toks = pl.col("_a2").str.split(" ").list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(list(ADDR_DROP))))
    df = df.with_columns(toks.alias("addr_tok"))
    for ctry, table in STATE_TABLES.items():
        single = {k: v for k, v in table.items() if " " not in k}
        df = df.with_columns(pl.when(pl.col("country") == ctry).then(_map_tokens(pl.col("addr_tok"), single)).otherwise(pl.col("addr_tok")).alias("addr_tok"))
    df = df.with_columns(_map_tokens(pl.col("addr_tok"), ADDR_CANON).alias("addr_tok"))
    df = df.with_columns(pl.when((pl.col("country") == "France") & pl.lit(APPLY_FR_RULES)).then(_map_tokens(pl.col("addr_tok"), FR_ADDR_CANON))
                           .otherwise(pl.col("addr_tok")).alias("addr_tok"))
    df = df.with_columns(
        pl.col("addr_tok").list.join(" ").alias("addr_full"),
        pl.col("addr_tok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))).alias("addr_num"),
        (pl.col("addr_tok").list.len() == 0).alias("addr_empty"),
    )
    return df.drop(["_n", "_a", "_n2", "_a2"])
