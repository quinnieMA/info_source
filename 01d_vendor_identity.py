# -*- coding: utf-8 -*-
"""
01d_vendor_identity.py  —  Vendor identity classification (v4, 2026-09-18)
================================================================================
Input  : data/cleaned/01_deal_vendor_name.csv   (01a VENDOR AGG output)
Output : data/cleaned/01_deal_vendor_type.csv
         data/cleaned/01_deal_vendor_type_profile.csv
         data/cleaned/01_vendor_ner_cache.csv   (NER cache)

────────────────────────────────────────────────────────────────────────────────
Core rules (introduced in v2, retained in v3/v4)
────────────────────────────────────────────────────────────────────────────────
  [R1] If a deal has NAMED-ENTITY vendors, primary is chosen ONLY from named
       entities; generic descriptors (SHAREHOLDERS/EMPLOYEES/MANAGEMENT...) do
       not compete.
  [R2] If a deal has NO named entities at all, primary is chosen from generic
       descriptors; in that case generic words represent the true vendor.

Rationale (empirical frequency diagnosis):
  SHAREHOLDERS  alone  1,309 | with entity  2,941  <- 69% redundant tagging
  EMPLOYEES     alone      9 | with entity     72  <- 89% redundant tagging
  MANAGEMENT    alone     26 | with entity    178  <- 87% redundant tagging
  RECEIVER      alone    219 | with entity     79  <- 74% alone, true vendor

────────────────────────────────────────────────────────────────────────────────
v4 changes from v3
────────────────────────────────────────────────────────────────────────────────
  [1] Bug fix: 3 generic words were treated as NAMED ENTITIES, falling into
       UNMAPPED and polluting ven_disclosure=named
         CREDITORS / CREDITOR        -> financial_inst
         MANAGERS / MANAGER          -> management
         INSTITUTIONAL INVESTORS     -> financial_inst
       These are not company names but "who is the seller" descriptors and
       must go into GENERIC_MAP.
  [2] GENERIC_PRIORITY adds government / financial_inst two tiers
       (v3 was missing these; generic words hitting them would fall back to default order)
  [3] ENTITY_RULES adds [country company suffixes] — the only effective lever
       for long-tail names. Diagnosis shows 88.1% of vendor names appear only
       once; enumerating each is futile. Suffix rules hit thousands of one-off
       names at once.
       government: FGUP / ministerstvo / konsolidacni / agentura /
                   autoritatea / valorificarea / activelor statului /
                   imushchestvennykh
       corporate : OOO / ZAO / PJSC / PAO / DD / D.O.O. / SP ZOO /
                   S.R.O. / KFT / ZRT / LTDA / SLU / TBK / JSC / EAD /
                   A.S. / ENTERPRISES
       financial : well-known institution names (MERRILL / GOLDMAN / MORGAN STANLEY /
                   UBS / BARCLAYS / HSBC / ...) plus Slovenian insurance/pension
                   (zavarovanj / pokojninsk / odskodninska / kapitalska)
  [4] pe_vc adds \binvestors?\b
       so PE firm names like "ENTERPRISE INVESTORS SP ZOO" are correctly classified as pe_vc
       (INSTITUTIONAL INVESTORS is already routed to financial_inst by GENERIC_MAP,
        unaffected by this rule)

Category system (identity dimension, MECE, primary values)
────────────────────────────────────────────────────────────────────────────────
  government          government / state-owned
  pe_vc               PE / VC / investment firms
  financial_inst      banks / insurance / securities
  corporate           generic company
  individual_natural  natural person (named individual / INDIVIDUALS / FOUNDERS / PROMOTERS / DIRECTORS)
  shareholder_group    shareholder group (SHAREHOLDERS / MINORITY SHAREHOLDERS appearing alone)
  insolvency          bankruptcy receivership (RECEIVER / LIQUIDATOR)
  management          management buyout (MANAGEMENT appearing alone)
  undisclosed         explicitly marked undisclosed
  UNMAPPED            has named entity but no rule matched
  (empty)             Zephyr has no vendor record (empty string after 08b left join)

Disclosure dimension (ven_disclosure): named / generic / undisclosed / no_record

────────────────────────────────────────────────────────────────────────────────
NER cache cross-machine reuse (empirically verified)
────────────────────────────────────────────────────────────────────────────────
  Cache = 01_vendor_ner_cache.csv (two columns: name, ner), a name->tag lookup.
  CoreNLP NER is deterministic for the same string, so the cache is machine-independent.
  On a new machine, just copy this CSV to D:/MA/data/cleaned/; no Java/CoreNLP needed.
  Verify: run this script and check "[NER ] pending X / total Y" — X=0 means full cache hit.
  Empirical: after switching machines, "pending 0 / total 43,473", cross-machine reuse confirmed.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = r"D:\MA"
CLEANED = os.path.join(BASE, "data", "cleaned")
os.makedirs(CLEANED, exist_ok=True)

# CoreNLP is optional: if absent, fall back to lexicon-only, pipeline continues
STANFORD_CORENLP_PATH = r"D:/MA/NLP/stanford-corenlp-4.5.10"
JAVA_BIN_PATH = r"C:\Program Files\Java\jdk1.8.0_202\bin"
if os.path.isdir(JAVA_BIN_PATH):
    os.environ["PATH"] = JAVA_BIN_PATH + os.pathsep + os.environ.get("PATH", "")

VEN_FP   = os.path.join(CLEANED, "01_deal_vendor_name.csv")
CACHE_FP = os.path.join(CLEANED, "01_vendor_ner_cache.csv")
LEX_FP   = os.path.join(BASE, "data", "vendor_lexicon.csv")
OUT_FP   = os.path.join(CLEANED, "01_deal_vendor_type.csv")


def safe_to_csv(df, path, **kw):
    """Atomic write: write .tmp then replace, avoids truncation when file is locked by Excel"""
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False, encoding=kw.pop("encoding", "utf-8-sig"), **kw)
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 1. Generic descriptor list — exact match (upper + strip)
#    These are not specific entity names but generic "who is the seller" descriptors
# ════════════════════════════════════════════════════════════════════════════
GENERIC_MAP = {
    # -- Shareholder group --
    "SHAREHOLDERS":               "shareholder_group",
    "SHAREHOLDER":                "shareholder_group",
    "MINORITY SHAREHOLDERS":      "shareholder_group",
    "MINORITY SHAREHOLDER":       "shareholder_group",
    "HOLDERS":                    "shareholder_group",
    "HOLDER":                     "shareholder_group",
    "SHAREHOLDERS (UNSPECIFIED)": "shareholder_group",
    "MAJORITY SHAREHOLDERS":      "shareholder_group",
    "SELLING SHAREHOLDERS":       "shareholder_group",
    # -- Natural persons / management groups --
    "INDIVIDUALS":                "individual_natural",
    "INDIVIDUAL":                 "individual_natural",
    "PRIVATE INDIVIDUALS":        "individual_natural",
    "FOUNDERS":                   "individual_natural",
    "FOUNDER":                    "individual_natural",
    "PROMOTERS":                  "individual_natural",
    "PROMOTER":                   "individual_natural",
    "EMPLOYEES":                  "individual_natural",
    "EMPLOYEE":                   "individual_natural",
    "INVESTORS":                  "individual_natural",
    "INVESTOR":                   "individual_natural",
    "DIRECTORS":                  "individual_natural",
    "DIRECTOR":                   "individual_natural",
    "OFFICERS":                   "individual_natural",
    "OFFICER":                    "individual_natural",
    # -- Management --
    "MANAGEMENT":                 "management",
    "MANAGEMENTS":                "management",
    "MANAGEMENT TEAM":            "management",
    "MBO":                        "management",
    "MANAGERS":                   "management",          # v4
    "MANAGER":                    "management",          # v4
    # -- Creditors / institutional investors --
    "CREDITORS":                  "financial_inst",      # v4
    "CREDITOR":                   "financial_inst",      # v4
    "INSTITUTIONAL INVESTORS":    "financial_inst",      # v4
    "INSTITUTIONAL INVESTOR":     "financial_inst",      # v4
    # -- Insolvency receivership --
    "RECEIVER":                   "insolvency",
    "RECEIVERS":                  "insolvency",
    "LIQUIDATOR":                 "insolvency",
    "LIQUIDATORS":                "insolvency",
    "ADMINISTRATOR":              "insolvency",
    "OFFICIAL RECEIVER":          "insolvency",
    # -- Undisclosed --
    "UNDISCLOSED":                "undisclosed",
    "NOT DISCLOSED":              "undisclosed",
    "UNDISCLOSED VENDOR":         "undisclosed",
    "UNDISCLOSED VENDORS":        "undisclosed",
    "UNDISCLOSED OWNERS":         "undisclosed",
    "UNDISCLOSED SELLERS":        "undisclosed",
    "UNDISCLOSED SHAREHOLDERS":   "undisclosed",
    "UNKNOWN":                    "undisclosed",
    "NOT AVAILABLE":              "undisclosed",
    "N/A":                        "undisclosed",
    # 1) GENERIC_MAP adds 2 more
    "UNDISCLOSED VENDOR(S)": "undisclosed",
    "ASSOCIATES":            "individual_natural",
}
GENERIC_EXACT = set(GENERIC_MAP.keys())

# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# 2. Entity lexicon — classification for NAMED-ENTITY names
#    Generic words do not appear here (handled by GENERIC_MAP)
# ════════════════════════════════════════════════════════════════════════════
ENTITY_RULES = [
    # -- Government / state-owned --
    ("government", r"\bgovernment\b"),
    ("government", r"\bministry\b"),
    ("government", r"\bministerstwo\b"),
    ("government", r"\bskarbu\s+panstwa\b"),
    ("government", r"\bfederalnoe\s+agentstvo\b"),
    ("government", r"\bgosudarstven"),
    ("government", r"\bfederalnym\s+imu"),
    ("government", r"\bfond\s+derzhavnoho\s+mayna\b"),
    ("government", r"\ballami\s+privatizacios\b"),
    ("government", r"\bvagyonkezelo\b"),
    ("government", r"\bsociedad\s+estatal\b"),
    ("government", r"\bparticipaciones\s+patrimoniales\b"),
    ("government", r"\bstate[- ]owned\b"),
    ("government", r"\bstate\s+property\b"),
    ("government", r"\bmunicipal\w*\b"),
    ("government", r"\bnationali[sz]ed?\b"),
    ("government", r"\bfederal\b"),
    ("government", r"\bcity\s+of\b"),
    ("government", r"\bprovince\b"),
    ("government", r"\bSASAC\b"),
    ("government", r"\bprivatisation\b"),
    ("government", r"\bprivatization\b"),
    ("government", r"\bstate\s+assets\b"),
    ("government", r"\bprivatizaci"),
    ("government", r"\bprivatisat"),
    ("government", r"\bagencij"),
    ("government", r"\bagentsiy"),
    ("government", r"republika srbija"),
    # v4: country SOE / ministry / state assets bureau
    ("government", r"\bFGUP\b"),                    # RU federal state unitary enterprise
    ("government", r"\bministerstvo\b"),            # RU/RS ministry
    ("government", r"\bimushchestvennykh\b"),       # RU property relations ministry
    ("government", r"\bkonsolidacni\b"),            # CZ consolidation agency
    ("government", r"\bagentura\b"),                # CZ agency
    ("government", r"\bautoritatea\b"),             # RO authority
    ("government", r"\bvalorificarea\b"),           # RO asset recovery
    ("government", r"activelor statului"),          # RO state assets
    ("government", r"国有资产"), ("government", r"国有"), ("government", r"国资"),
    ("government", r"政府"), ("government", r"财政"), ("government", r"国资委"),
    ("government", r"人民政府"), ("government", r"开发区"), ("government", r"管委会"),
    # -- PE / VC / investment firms --
    ("pe_vc", r"capital partners?"),
    ("pe_vc", r"private equity"),
    ("pe_vc", r"venture capital"),
    ("pe_vc", r"\bventures?\b"),
    ("pe_vc", r"\bbuyout\b"),
    ("pe_vc", r"equity partners?"),
    ("pe_vc", r"\bKKR\b"), ("pe_vc", r"\bBain Capital\b"),
    ("pe_vc", r"\bCarlyle\b"), ("pe_vc", r"\bCVC\b"), ("pe_vc", r"\bEQT\b"),
    ("pe_vc", r"\bTPG\b"), ("pe_vc", r"\bPermira\b"),
    ("pe_vc", r"\bBlackstone\b"), ("pe_vc", r"\bApollo\b"),
    ("pe_vc", r"\bAdvent\b"), ("pe_vc", r"\bWarburg\b"), ("pe_vc", r"\bPincus\b"),
    ("pe_vc", r"\bL\.?P\.?\b"),
    ("pe_vc", r"\binvestment\b"), ("pe_vc", r"\binvestments\b"),
    ("pe_vc", r"\binvestors?\b"),                   # v4
    ("pe_vc", r"\bfund\b"), ("pe_vc", r"\bfunds\b"),
    ("pe_vc", r"股权投资基金"), ("pe_vc", r"创业投资"), ("pe_vc", r"创投"),
    ("pe_vc", r"投资基金"), ("pe_vc", r"投资合伙"), ("pe_vc", r"私募"),
    ("pe_vc", r"股权投资"), ("pe_vc", r"投资管理"), ("pe_vc", r"投资"),
    ("pe_vc", r"基金"), ("pe_vc", r"资本"),
    # -- Financial institutions --
    ("financial_inst", r"\bbank\b"), ("financial_inst", r"\bbancorp\b"),
    ("financial_inst", r"\bbanca\b"),
    ("financial_inst", r"\bbanco\b"),
    ("financial_inst", r"\bbanque\b"),
    ("financial_inst", r"\binsurance\b"), ("financial_inst", r"\bassurance\b"),
    ("financial_inst", r"\bsecurities\b"), ("financial_inst", r"\btrust\b"),
    ("financial_inst", r"\basset management\b"),
    # v4: well-known financial institution names (no bank/insurance keyword, must name explicitly)
    ("financial_inst", r"\bmerrill\b"),
    ("financial_inst", r"\bgoldman\b"),
    ("financial_inst", r"\bmorgan stanley\b"),
    ("financial_inst", r"\bjp\s?morgan\b"),
    ("financial_inst", r"\bUBS\b"),
    ("financial_inst", r"\bcredit suisse\b"),
    ("financial_inst", r"\bbarclays\b"),
    ("financial_inst", r"\bHSBC\b"),
    ("financial_inst", r"\bcitigroup\b"),
    ("financial_inst", r"\bnomura\b"),
    ("financial_inst", r"\blloyds?\b"),
    ("financial_inst", r"\brothschild\b"),
    ("financial_inst", r"\blazard\b"),
    ("financial_inst", r"\bjefferies\b"),
    ("financial_inst", r"\bevercore\b"),
    ("financial_inst", r"\bmoelis\b"),
    ("financial_inst", r"\bmacquarie\b"),
    ("financial_inst", r"\bstandard chartered\b"),
    ("financial_inst", r"\bsociete generale\b"),
    ("financial_inst", r"\bBNP\b"), ("financial_inst", r"\bING\b"),
    ("financial_inst", r"\baviva\b"), ("financial_inst", r"\bprudential\b"),
    ("financial_inst", r"\bAXA\b"), ("financial_inst", r"\ballianz\b"),
    ("financial_inst", r"\bzurich\b"), ("financial_inst", r"\bmetlife\b"),
    ("financial_inst", r"\bAIG\b"),
    # v4: Slovenian insurance / pension / claims
    ("financial_inst", r"\bzavarovanj"),
    ("financial_inst", r"\bpokojninsk"),
    ("financial_inst", r"\bodskodninska\b"),
    ("financial_inst", r"\bkapitalska\b"),
    # -- Insolvency (entity form) --
    ("financial_inst", r"银行"), ("financial_inst", r"保险"),
    ("financial_inst", r"证券"), ("financial_inst", r"信托"),
    ("financial_inst", r"资产管理"),
    # -- Insolvency (entity form) --
    ("insolvency", r"\breceivership\b"),
    ("insolvency", r"\binsolvency\b"),
    ("insolvency", r"\bliquidation\b"),
    ("insolvency", r"\bbankrupt"),
    # -- Natural persons / family --
    ("individual_natural", r"\bfamily\b"),
    ("individual_natural", r"\bfamily[- ]owned\b"),
    ("individual_natural", r"家族"),
    # -- Corporate (company suffixes, fallback) --
    ("corporate", r"\bInc\.?\b"), ("corporate", r"\bLtd\.?\b"),
    ("corporate", r"\bLLC\b"), ("corporate", r"\bGmbH\b"), ("corporate", r"\bAG\b"),
    ("corporate", r"\bS\.?p\.?A\.?\b"), ("corporate", r"\bSPA\b"),
    ("corporate", r"\bSAS\b"), ("corporate", r"\bN\.?V\.?\b"),
    ("corporate", r"\bB\.?V\.?\b"), ("corporate", r"\bPLC\b"),
    ("corporate", r"\bCo\.,? ?Ltd\b"), ("corporate", r"\bGroup\b"),
    ("corporate", r"\bHoldings?\b"), ("corporate", r"\bIndustries\b"),
    ("corporate", r"\bCorporation\b"), ("corporate", r"\bCompany\b"),
    ("corporate", r"\bSA\b"), ("corporate", r"\bOAO\b"), ("corporate", r"\bOJSC\b"),
    ("corporate", r"\bASA\b"), ("corporate", r"\bAS\b"), ("corporate", r"\bOY\b"),
    ("corporate", r"\bOYJ\b"), ("corporate", r"\bAB\b"), ("corporate", r"\bBHD\b"),
    ("corporate", r"\bRT\b"), ("corporate", r"\bSCA\b"), ("corporate", r"\bSCS\b"),
    ("corporate", r"\bPTE\b"), ("corporate", r"\bPTY\b"), ("corporate", r"\bKK\b"),
    ("corporate", r"\bSRL\b"), ("corporate", r"\bSARL\b"),
    ("corporate", r"\bLIMITED\b"), ("corporate", r"\bPCL\b"),
    ("corporate", r"\bA/S\b"), ("corporate", r"\bSCPA\b"), ("corporate", r"\bSCARL\b"),
    # v4: country suffixes (key long-tail lever)
    ("corporate", r"\bOOO\b"),                     # RU LLC
    ("corporate", r"\bZAO\b"),                     # RU closed joint-stock
    ("corporate", r"\bPJSC\b"), ("corporate", r"\bPAO\b"),   # RU public joint-stock
    ("corporate", r"\bDD\b"),                      # SI/HR joint-stock
    ("corporate", r"\bD\.?O\.?O\.?\b"),            # HR/RS LLC
    ("corporate", r"\bSP\.?\s?Z\.?\s?O\.?\s?O\.?\b"),  # PL LLC
    ("corporate", r"\bSP\s?ZOO\b"),
    ("corporate", r"\bspolka\b"),
    ("corporate", r"\bS\.?R\.?O\.?\b"),            # CZ/SK LLC
    ("corporate", r"\bA\.S\.?\b"),                 # CZ joint-stock
    ("corporate", r"\bKFT\b"), ("corporate", r"\bZRT\b"),   # HU
    ("corporate", r"\bLTDA\b"),                    # BR/ES
    ("corporate", r"\bSLU\b"),                     # ES
    ("corporate", r"\bTBK\b"),                     # ID
    ("corporate", r"\bJSC\b"),                     # VN/BG
    ("corporate", r"\bEAD\b"),                     # BG
    ("corporate", r"\bENTERPRISES?\b"),
    ("corporate", r"公司"), ("corporate", r"集团"), ("corporate", r"控股"),
    ("corporate", r"股份"), ("corporate", r"有限"), ("corporate", r"实业"),
    # 2) ENTITY_RULES adds 5 more (all patterns, not bare words)
    ("individual_natural", r"\bMR\b"), ("individual_natural", r"\bMRS\b"),
    ("individual_natural", r"\bMS\b"),  ("individual_natural", r"\bDR\b"),
    ("government", r"\bimushchestv"),              # RU property ministry (imushchestva/imushchestvennykh)
    ("government", r"\bkommune?\b"),               # Nordic municipality (Oslo Kommune)
    ("government", r"\bmajetku\b"),                # SK state asset fund
    ("corporate",  r"\bSL\b"),                     # ES LLC (GRUPO INTERCOM FACTORY SL)

]

# Old lexicon CSV type names -> new system (idempotent compatibility)
LEGACY_MAP = {
    "pe_vc_strong": "pe_vc", "pe_vc_weak": "pe_vc",
    "corporate_suffix": "corporate",
    "family": "individual_natural",
    "family_person": "individual_natural",
    "placeholder_individuals": "individual_natural",
    "placeholder_insolvency": "insolvency",
    "placeholder_management": "management",
    # Handled by GENERIC_MAP below, excluded from entity classification
    "placeholder_shareholders": "__DROP__",
    "shareholders_undisclosed": "__DROP__",
    "shareholder_group": "__DROP__",
    "undisclosed": "__DROP__",
}

# Entity classification priority (lower number = higher priority)
REAL_PRIORITY = {
    "government": 0,
    "insolvency": 1,
    "pe_vc": 2,
    "financial_inst": 3,
    "corporate": 4,
    "individual_natural": 5,
}
# Generic word classification priority (only used when all names are generic)
# v4: adds government / financial_inst two tiers
GENERIC_PRIORITY = {
    "insolvency": 0,
    "government": 1,
    "financial_inst": 2,
    "management": 3,
    "individual_natural": 4,
    "shareholder_group": 5,
    "undisclosed": 6,
}


def build_lexicon():
    """Merge CSV legacy rules + ENTITY_RULES, deduplicate, drop generic classes"""
    rows = []
    if os.path.exists(LEX_FP):
        try:
            cur = pd.read_csv(LEX_FP, encoding="utf-8-sig")
            cur = cur.dropna(subset=["type", "pattern"])
            for t, p in zip(cur["type"], cur["pattern"]):
                t = LEGACY_MAP.get(str(t), str(t))
                if t != "__DROP__":
                    rows.append((t, str(p)))
        except Exception as e:
            print("  [warn] Lexicon read failed (%s), using built-in rules only" % type(e).__name__)
    rows.extend(ENTITY_RULES)
    lex = pd.DataFrame(rows, columns=["type", "pattern"]) \
        .drop_duplicates(subset=["type", "pattern"]).reset_index(drop=True)
    safe_to_csv(lex, LEX_FP)
    print("[lexicon] %s  total %d entity rules" % (LEX_FP, len(lex)))
    d = {}
    for t, p in zip(lex["type"], lex["pattern"]):
        d.setdefault(t, []).append(p)
    return d


LEX_DICT = build_lexicon()
REAL_CHECK_ORDER = sorted(REAL_PRIORITY, key=lambda k: REAL_PRIORITY[k])


def lex_hit_entity(name):
    """Lexicon classification for named entities"""
    for t in REAL_CHECK_ORDER:
        for p in LEX_DICT.get(t, []):
            try:
                if re.search(p, name, flags=re.IGNORECASE):
                    return t
            except re.error:
                continue
    return None


# ════════════════════════════════════════════════════════════════════════════
# 3. NER (cache first -> CoreNLP optional -> graceful fallback)
# ════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("01d — Vendor identity")
print("=" * 70)

df = pd.read_csv(VEN_FP, low_memory=False)
df["deal_num"] = pd.to_numeric(df["deal_num"], errors="coerce").astype("Int64")
print("[data] vendor table %s deals" % format(len(df), ","))

_unique = set()
for s in df["ven_name_all"].dropna():
    for x in str(s).split("|"):
        x = x.strip()
        if x and x.lower() != "nan":
            _unique.add(x)
_unique = sorted(_unique)

_nerd = {}
if os.path.exists(CACHE_FP):
    try:
        c = pd.read_csv(CACHE_FP, encoding="utf-8-sig")
        _nerd = dict(zip(c["name"].astype(str), c["ner"].astype(str)))
        print("[cache] %s entries already (cross-machine reusable)" % format(len(_nerd), ","))
    except Exception:
        pass

_todo = [n for n in _unique if n not in _nerd]
print("[NER ] pending %s / total %s" % (format(len(_todo), ","), format(len(_unique), ",")))

if _todo:
    _nlp = None
    try:
        from stanfordcorenlp import StanfordCoreNLP
        if os.path.isdir(STANFORD_CORENLP_PATH):
            _nlp = StanfordCoreNLP(STANFORD_CORENLP_PATH, lang="en")
            print("  [ok] CoreNLP started")
    except Exception as e:
        print("  [warn] CoreNLP unavailable (%s)" % type(e).__name__)

    if _nlp is not None:
        for i, nm in enumerate(_todo):
            try:
                tags = [t for _, t in _nlp.ner(nm) if t != "O"]
                if any(t == "PERSON" for t in tags):
                    _nerd[nm] = "PERSON"
                elif any(t == "ORGANIZATION" for t in tags):
                    _nerd[nm] = "ORGANIZATION"
                elif tags:
                    _nerd[nm] = tags[0]
                else:
                    _nerd[nm] = "NONE"
            except Exception:
                _nerd[nm] = "NONE"
            if (i + 1) % 5000 == 0:
                print("      %s/%s" % (format(i + 1, ","), format(len(_todo), ",")))
        try:
            _nlp.close()
        except Exception:
            pass
    else:
        for nm in _todo:
            _nerd[nm] = "NONE"
        print("  [warn] %s names have no NER tag, lexicon-only mode"
              % format(len(_todo), ","))

    safe_to_csv(pd.DataFrame({"name": list(_nerd.keys()),
                              "ner": list(_nerd.values())}), CACHE_FP)
    print("  cache updated -> %s" % CACHE_FP)
else:
    print("  all cache hits, skipping CoreNLP (no Java needed on this machine)")


# ════════════════════════════════════════════════════════════════════════════
# 4. Classification
# ════════════════════════════════════════════════════════════════════════════
def classify_one(name):
    """Single vendor name -> (type, is_generic, ner_tag)"""
    s = str(name).strip()
    if not s or s.lower() == "nan":
        return "missing", True, "NONE"
    up = s.upper()
    if up in GENERIC_EXACT:                      # Generic descriptor
        return GENERIC_MAP[up], True, "GENERIC"
    ner = _nerd.get(s, "NONE")                   # Named entity
    t = lex_hit_entity(s)
    if t:
        return t, False, ner
    if ner == "PERSON":
        return "individual_natural", False, ner
    if ner == "ORGANIZATION":
        return "corporate", False, ner
    return "UNMAPPED", False, ner


def classify_deal(name_all):
    """-> (primary, all_types, disclosure, mixed, n_names, rescued)"""
    if pd.isna(name_all):
        return (np.nan, "", "no_record", "", 0, False)
    names = [x.strip() for x in str(name_all).split("|")
             if x.strip() and x.strip().lower() != "nan"]
    if not names:
        return (np.nan, "", "no_record", "", 0, False)

    types, isgen = [], []
    for nm in names:
        t, g, _ = classify_one(nm)
        types.append(t)
        isgen.append(g)

    real_types = [t for t, g in zip(types, isgen) if (not g) and t != "missing"]
    gen_types = [t for t, g in zip(types, isgen) if g and t != "missing"]

    # [R1] Has named entities -> choose from entities only; generic words are overridden, mark rescued
    if real_types:
        prim = min(real_types, key=lambda t: REAL_PRIORITY.get(t, 9))
        disc = "named"
        rescued = bool(gen_types)
    # [R2] All generic -> use generic words
    elif gen_types:
        prim = min(gen_types, key=lambda t: GENERIC_PRIORITY.get(t, 9))
        disc = "undisclosed" if prim == "undisclosed" else "generic"
        rescued = False
    else:
        prim, disc, rescued = "UNMAPPED", "named", False

    uniq = sorted(set(t for t in types if t != "missing"))
    mixed = "mixed" if len(uniq) > 1 else ""
    return (prim, "|".join(uniq), disc, mixed, len(names), rescued)


print("\n[classify] Applying rules (named entities first, generic words do not compete) ...")
res = df["ven_name_all"].apply(classify_deal)
df["ven_type_primary"] = [r[0] for r in res]
df["ven_type_all"]     = [r[1] for r in res]
df["ven_disclosure"]   = [r[2] for r in res]
df["ven_type_mixed"]   = [r[3] for r in res]
df["ven_n_names"]      = [r[4] for r in res]
_rescued = [r[5] for r in res]

# -- MBO / management participation flag (overlapping, does not compete for primary) --
_MBO = r"\bMBO\b|\bmanagement buyout\b|\bmanagement\b"
df["ven_has_mbo"] = df["ven_name_all"].apply(
    lambda s: int(bool(re.search(_MBO, str(s), flags=re.IGNORECASE)))
    if pd.notna(s) else 0)

# -- Distribution --
print("\n[Vendor type distribution] (primary, MECE)")
vc = df["ven_type_primary"].value_counts(dropna=False)
for k, n in vc.items():
    lab = "(no vendor record)" if pd.isna(k) else k
    print("    %-22s %8s  (%.1f%%)" % (lab, format(n, ","), n / len(df) * 100))

print("\n[Disclosure status distribution]")
for k, n in df["ven_disclosure"].value_counts().items():
    print("    %-16s %8s  (%.1f%%)" % (k, format(n, ","), n / len(df) * 100))

# -- Impact of this fix --
n_rescued = int(sum(_rescued))
print("\n* Generic words overridden, now replaced by named entity: %s (%.1f%%)"
      % (format(n_rescued, ","), n_rescued / len(df) * 100))
if n_rescued:
    _sub = pd.DataFrame({"t": [r[0] for r in res], "f": _rescued})
    print("   New primary distribution for these deals:")
    for k, n in _sub[_sub["f"]]["t"].value_counts().items():
        print("      %-22s %8s" % (str(k), format(n, ",")))

print("  ven_has_mbo = 1: %s (%.1f%%)"
      % (format(int(df["ven_has_mbo"].sum()), ","), df["ven_has_mbo"].mean() * 100))
print("  mixed vendors: %s (%.1f%%)"
      % (format(int((df["ven_type_mixed"] == "mixed").sum()), ","),
         (df["ven_type_mixed"] == "mixed").mean() * 100))

safe_to_csv(df, OUT_FP)
print("\n  Saved -> %s" % OUT_FP)

# ════════════════════════════════════════════════════════════════════════════
# 5. UNMAPPED diagnosis — exclude generic words, count only true entity names
# ════════════════════════════════════════════════════════════════════════════
_unm = df[df["ven_type_primary"] == "UNMAPPED"]
print("\n  UNMAPPED: %s (%.1f%%)" % (format(len(_unm), ","), len(_unm) / len(df) * 100))
if len(_unm):
    cnt_real, cnt_gen = {}, {}
    for s in _unm["ven_name_all"].dropna():
        for x in str(s).split("|"):
            x = x.strip()
            if not x or x.lower() == "nan":
                continue
            if x.upper() in GENERIC_EXACT:
                cnt_gen[x] = cnt_gen.get(x, 0) + 1
            else:
                cnt_real[x] = cnt_real.get(x, 0) + 1
    top = sorted(cnt_real.items(), key=lambda x: -x[1])[:100]
    safe_to_csv(pd.DataFrame(top, columns=["vendor_name", "freq"]),
                os.path.join(CLEANED, "01_vendor_unmapped.csv"))
    print("  [true-entity UNMAPPED, generic words excluded] top 15:")
    for nm, f in top[:15]:
        print("      %-52s %6s" % (nm[:52], format(f, ",")))
    if cnt_gen:
        print("  [reference] accompanying generic words (not misses, for info only):")
        for nm, f in sorted(cnt_gen.items(), key=lambda x: -x[1])[:6]:
            print("      %-52s %6s" % (nm[:52], format(f, ",")))

# ════════════════════════════════════════════════════════════════════════════
# 6. Profile: vendor type x information environment
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Vendor type x information environment profile")
print("=" * 70)

prof = df.copy()
src_fp = os.path.join(CLEANED, "01_deal_info_source_count.csv")
val_fp = os.path.join(CLEANED, "01_deal_value.csv")
if os.path.exists(src_fp):
    s = pd.read_csv(src_fp, low_memory=False)
    s["deal_num"] = pd.to_numeric(s["deal_num"], errors="coerce").astype("Int64")
    prof = prof.merge(s, on="deal_num", how="left")
if os.path.exists(val_fp):
    v = pd.read_csv(val_fp, low_memory=False)
    v["deal_num"] = pd.to_numeric(v["deal_num"], errors="coerce").astype("Int64")
    prof = prof.merge(v[["deal_num", "deal_value"]], on="deal_num", how="left")

_08b = os.path.join(BASE, "data", "merged", "08b_deal_firm_analysis.dta")
if os.path.exists(_08b):
    try:
        d8 = pd.read_stata(_08b, columns=["deal_num", "ln_days",
                                          "ln_mul_rev", "num_Advisor_Submission"])
        d8["deal_num"] = pd.to_numeric(d8["deal_num"], errors="coerce").astype("Int64")
        d8["has_adv"] = (pd.to_numeric(d8["num_Advisor_Submission"],
                                       errors="coerce").fillna(0) > 0).astype(int)
        prof = prof.merge(d8[["deal_num", "ln_days", "ln_mul_rev", "has_adv"]],
                          on="deal_num", how="left")
        print("  merged 08b (ln_days / ln_mul_rev / has_adv)")
    except Exception as e:
        print("  08b merge failed: %s" % type(e).__name__)

CH = [c for c in ["num_Advisor_Submission", "num_Stock_Exchange",
                  "num_Company_Press_Release", "num_Website",
                  "num_Electronic_Publication", "num_Miscellaneous"]
      if c in prof.columns]
EXTRA = [c for c in ["ln_days", "ln_mul_rev", "has_adv"] if c in prof.columns]

print("\n  %-22s %7s %8s %8s %8s %11s"
      % ("vendor type", "N", "advsub", "exch", "press", "median val"))
for t, g in prof.groupby("ven_type_primary", dropna=False):
    if len(g) < 20:
        continue
    lab = "(no record)" if pd.isna(t) else t
    print("  %-22s %7s %8.3f %8.3f %8.3f %11.0f" % (
        lab, format(len(g), ","),
        g["num_Advisor_Submission"].mean() if "num_Advisor_Submission" in g else np.nan,
        g["num_Stock_Exchange"].mean() if "num_Stock_Exchange" in g else np.nan,
        g["num_Company_Press_Release"].mean() if "num_Company_Press_Release" in g else np.nan,
        g["deal_value"].median() if "deal_value" in g else np.nan))

if EXTRA:
    print("\n  %-22s %7s %9s %11s %8s"
          % ("vendor type", "N", "ln_days", "ln_mul_rev", "has_adv"))
    for t, g in prof.groupby("ven_type_primary", dropna=False):
        if len(g) < 20:
            continue
        lab = "(no record)" if pd.isna(t) else t
        row = [lab, format(len(g), ",")]
        for c in ["ln_days", "ln_mul_rev", "has_adv"]:
            row.append("%.3f" % pd.to_numeric(g[c], errors="coerce").mean()
                       if c in g.columns else "n/a")
        print("  %-22s %7s %9s %11s %8s" % tuple(row))

rows = []
for t, g in prof.groupby("ven_type_primary", dropna=False):
    r = {"ven_type": "(no_record)" if pd.isna(t) else t, "N": len(g)}
    for c in CH + EXTRA:
        if c in g.columns:
            r[c + "_mean"] = pd.to_numeric(g[c], errors="coerce").mean()
    if "deal_value" in g.columns:
        r["deal_value_median"] = pd.to_numeric(g["deal_value"], errors="coerce").median()
    rows.append(r)
safe_to_csv(pd.DataFrame(rows),
            os.path.join(CLEANED, "01_deal_vendor_type_profile.csv"))
print("\n  Saved -> 01_deal_vendor_type_profile.csv")
print("\n=== 01d complete ===")
#（注：内容由AI生成）
