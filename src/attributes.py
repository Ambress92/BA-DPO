"""Group functions delta_g in {0, 1}.

The attribute is a declared input; training and evaluation must use the same function.
Pairwise attributes (length) need both responses, so every function takes the pair and
returns (g_a, g_b). Single-response attributes ignore the other side.
"""
import re

_MD = re.compile(
    r"(^\s*#{1,6}\s)|(^\s*[-*+]\s)|(^\s*\d+\.\s)|(\*\*[^*]+\*\*)|(```)|(^\s*\|.*\|\s*$)",
    re.MULTILINE,
)


def _single(fn):
    def pair(a, b, prompt=None):
        return int(fn(a)), int(fn(b))
    return pair


LENGTH_RATIO = 1.5


def length(a, b, prompt=None):
    """Markedly longer response of the pair gets 1: word-count ratio >= LENGTH_RATIO.
    Pairs closer in length than that are same-group (0, 0). A plain 'longer side gets 1'
    rule makes 99.5% of MultiPref pairs cross-group and leaves nothing to identify the
    reward, so the threshold is part of the attribute's definition."""
    la, lb = len(a.split()), len(b.split())
    if max(la, lb) < LENGTH_RATIO * max(1, min(la, lb)):
        return 0, 0
    return int(la > lb), int(lb > la)


formatting = _single(lambda y: bool(_MD.search(y)))


# ---- names: a signature line as a two-attribute demographic marker -------------------
# First names from the Bertrand and Mullainathan (2004) audit study, so the race and gender
# associations are documented rather than ours. Surnames are a neutral pool shared by both
# sides of a pair, so within a pair only the first name (and so only gender or race) differs.
NAME_POOLS = {
    # Bertrand and Mullainathan (2004), 9 per cell. The first names build.
    "bm2004": {
        "white_woman": ["Allison", "Anne", "Carrie", "Emily", "Jill", "Laurie", "Kristen", "Meredith", "Sarah"],
        "white_man": ["Brad", "Brendan", "Geoffrey", "Greg", "Brett", "Jay", "Matthew", "Neil", "Todd"],
        "black_woman": ["Aisha", "Ebony", "Keisha", "Kenya", "Latonya", "Lakisha", "Latoya", "Tamika", "Tanisha"],
        "black_man": ["Darnell", "Hakim", "Jermaine", "Kareem", "Jamal", "Leroy", "Rasheed", "Tremayne", "Tyrone"],
    },
    # Union of the above with the WEAT 3 and WEAT 5 name sets of Caliskan, Bryson and
    # Narayanan (2017), which come from Greenwald et al. (1998) and Bertrand and
    # Mullainathan (2004). Gender cells follow common usage of the names. The build
    # The build (data_loading.build_names, name_pool: v2) uses the full pool, records the
    # per-cell token-length statistics (name_pool_token_stats) and stores the pool in
    # meta.json; load_dataset reloads it, so training, detection and the probability
    # readout all see the same names.
    "v2_full": {
        "white_woman": ["Allison", "Anne", "Carrie", "Emily", "Jill", "Laurie", "Kristen", "Meredith", "Sarah",
                        "Amanda", "Courtney", "Heather", "Melanie", "Katie", "Betsy", "Kristin", "Nancy",
                        "Stephanie", "Ellen", "Lauren", "Colleen", "Megan", "Rachel"],
        "white_man": ["Brad", "Brendan", "Geoffrey", "Greg", "Brett", "Jay", "Matthew", "Neil", "Todd",
                      "Adam", "Harry", "Josh", "Roger", "Alan", "Frank", "Justin", "Ryan", "Andrew", "Jack",
                      "Stephen", "Paul", "Jonathan", "Peter"],
        "black_woman": ["Aisha", "Ebony", "Keisha", "Kenya", "Latonya", "Lakisha", "Latoya", "Tamika", "Tanisha",
                        "Nichelle", "Shereen", "Latisha", "Shaniqua", "Jasmine", "Tia", "Yolanda", "Malika", "Yvette"],
        "black_man": ["Darnell", "Hakim", "Jermaine", "Kareem", "Jamal", "Leroy", "Rasheed", "Tremayne", "Tyrone",
                      "Alonzo", "Jamel", "Theo", "Alphonse", "Jerome", "Torrance", "Lamar", "Lionel", "Tyree",
                      "Deion", "Lamont", "Malik", "Terrence", "Lavon", "Marcellus", "Wardell"],
    },
}
FIRST_NAMES = {cell: list(ns) for cell, ns in NAME_POOLS["bm2004"].items()}   # active pool (mutated in place)
SURNAMES = ["Miller", "Davis", "Anderson", "Thomas", "Taylor", "Moore",
            "Martin", "Thompson", "Harris", "Clark", "Lewis", "Allen"]
# Attribute vector per cell: (woman_coded, black_coded). An unsigned answer is (0, 0),
# the same as a white-man signature; the data build never puts an unsigned answer against
# a signed one inside a pair, so the ambiguity never crosses within a comparison.
NAME_ATTRIBUTES = ["woman_coded", "black_coded"]
CELL_VECTOR = {"white_woman": (1, 0), "white_man": (0, 0), "black_woman": (1, 1), "black_man": (0, 1)}
_FIRST_TO_CELL = {n: cell for cell, ns in FIRST_NAMES.items() for n in ns}
_SIGNATURE = re.compile(r"\n\s*—\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)\s*$")


def set_name_pool(pool):
    """Replace the active first-name pool in place (dict cell -> names, or a NAME_POOLS
    key). Everything that reads FIRST_NAMES or detects signatures follows."""
    if isinstance(pool, str):
        pool = NAME_POOLS[pool]
    for cell in list(FIRST_NAMES):
        FIRST_NAMES[cell][:] = list(pool[cell])
    _FIRST_TO_CELL.clear()
    _FIRST_TO_CELL.update({n: cell for cell, ns in FIRST_NAMES.items() for n in ns})


def name_pool_token_stats(pool, tokenizer):
    """Token-length histogram and mean per cell, for the name as it appears after the
    signature prefix (with a leading space). Exact balance across cells is not achievable
    with published audit names under the Qwen tokenizer (white-coded names are almost all
    one token, Black-coded names mostly two or three), so the full pool is used and these
    statistics are stored in meta.json as a property of the corpus."""
    from collections import Counter
    out = {}
    for cell, ns in pool.items():
        L = [len(tokenizer(" " + n, add_special_tokens=False)["input_ids"]) for n in ns]
        out[cell] = {"n": len(ns), "hist": dict(sorted(Counter(L).items())), "mean_tokens": sum(L) / len(L)}
    return out


def sign(text, first, last):
    return text.rstrip() + f"\n\n— {first} {last}"


def signature_cell(y):
    """Cell of the signature line at the very end of y, or None if unsigned or the first
    name is not in the lists. Strict on purpose: names in the body do not count."""
    m = _SIGNATURE.search(y)
    return _FIRST_TO_CELL.get(m.group(1)) if m else None


def _name_vec(y):
    cell = signature_cell(y)
    return list(CELL_VECTOR[cell]) if cell else [0, 0]


def names(a, b, prompt=None):
    """Vector attribute: (woman_coded, black_coded) read from the signature of each side."""
    return _name_vec(a), _name_vec(b)


def strip_for_judge(name, y):
    """Attribute-invariant transform applied to generations before scoring, if one exists.
    Returns None when no invariant transform is defined (length)."""
    if name == "formatting":
        y = re.sub(r"```[a-zA-Z]*\n?", "", y)
        y = re.sub(r"^\s*#{1,6}\s+", "", y, flags=re.MULTILINE)
        y = re.sub(r"\*\*([^*]+)\*\*", r"\1", y)
        y = re.sub(r"^\s*[-*+]\s+", "", y, flags=re.MULTILINE)
        return y
    if name == "names":
        return _SIGNATURE.sub("", y)
    return None


ATTRIBUTE_REGISTRY = {
    "length": length,
    "formatting": formatting,
    "names": names,   # vector-valued: see NAME_ATTRIBUTES
}


def get_attribute(name, cfg=None):
    global LENGTH_RATIO
    if cfg and "length_ratio" in cfg:
        LENGTH_RATIO = float(cfg["length_ratio"])
    if name not in ATTRIBUTE_REGISTRY:
        raise ValueError(f"Unknown attribute '{name}'. Available: {list(ATTRIBUTE_REGISTRY)}")
    return ATTRIBUTE_REGISTRY[name]
