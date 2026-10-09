"""Conservative, deterministic query hints and explicit corpus-scope checks.

Hints broaden lexical retrieval only. They are not accounting identities and
never replace the user's question for embeddings, reranking, or generation.
An absent report is a hard coverage failure only when that report is explicitly
requested: a historical metric can also appear in a later comparative filing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping


POLICY_VERSION = "financial-query-policy-v1"
_AMBIGUOUS_TICKERS = {"NOW", "ON", "IT", "CAT", "COST", "LOW", "HD", "KO", "PG", "PH", "CL", "DE", "GIS"}
_KNOWN_COMPANIES = {"AAPL": "Apple", "TSLA": "Tesla"}
_ALIASES = {
    "DE": ("John Deere", "Deere"), "EMR": ("Emerson",),
    "MU": ("Micron",), "CDNS": ("Cadence",), "MDLZ": ("Mondelez", "Mondelēz"),
    "PG": ("P&G", "Procter and Gamble"), "PH": ("Parker Hannifin",),
    "LOW": ("Lowes",),
}
_YEAR = r"(?:19|20)\d{2}"
_FISCAL_YEAR = re.compile(rf"\b(?:FY\s*|fiscal\s+(?:year\s+)?)({_YEAR})\b", re.I)
_REPORT = r"(?:annual\s+reports?|(?:form\s+)?10[-\s]?Ks?|reports?)"
_REPORT_BEFORE = re.compile(rf"\b{_REPORT}\s+(?:for\s+|of\s+|covering\s+)?(?:FY\s*|fiscal\s+year\s+)?({_YEAR})\b", re.I)
_REPORT_AFTER = re.compile(rf"\b(?:FY\s*|fiscal\s+year\s+)?({_YEAR})\s+(?:annual\s+|fiscal\s+)?{_REPORT}\b", re.I)
_REPORT_YEAR_LIST = re.compile(rf"\b((?:FY\s*|fiscal\s+year\s+)?{_YEAR}(?:\s*(?:,\s*(?:and\s+)?|and\s+|[-]|to\s+|through\s+)\s*(?:FY\s*|fiscal\s+year\s+)?{_YEAR})+)\s+{_REPORT}\b", re.I)
_REPORT_YEAR_LIST_AFTER = re.compile(rf"\b{_REPORT}\s+(?:for\s+|of\s+|covering\s+)?((?:FY\s*|fiscal\s+year\s+)?{_YEAR}(?:\s*(?:,\s*(?:and\s+)?|and\s+|[-]|to\s+|through\s+)\s*(?:FY\s*|fiscal\s+year\s+)?{_YEAR})+)\b", re.I)
_FORECAST = re.compile(r"\b(?:project(?:ed|ion|ions)?|forecast(?:s|ed|ing)?|guidance|outlook|expect(?:ed|ations)?|future|predict(?:ed|ion|ions)?|hypothetical|estimat(?:e|ed|es))\b", re.I)
_METRICS = (
    ("gross_margin", r"\bgross\s+(?:profit\s+)?margin\b|\bgross\s+profit\s+as\s+a\s+percentage\b",
     ("gross margin", "gross profit as a percentage of net sales")),
    ("operating_cash_flow", r"\b(?:operating\s+cash\s+flow|cash\s+flows?\s+from\s+operating\s+activities|net\s+cash\s+(?:provided\s+by|used\s+in)\s+operating\s+activities)\b",
     ("net cash provided by operating activities", "cash flows from operating activities")),
    ("operating_income", r"\b(?:operating\s+(?:income|profit|loss)|income\s+from\s+operations)\b",
     ("income from operations", "operating income")),
    ("net_income", r"\bnet\s+(?:income|earnings|profit|loss)\b",
     ("net income", "net earnings")),
    ("revenue", r"\b(?:revenues?|net\s+sales|total\s+sales)\b",
     ("total revenue", "net sales")),
    ("gross_profit", r"\bgross\s+profit\b", ("gross profit",)),
)


def _normalized(text: str) -> str:
    return text.replace("’", "'").replace("–", "-").replace("—", "-")


def _inventory(records: Iterable[Mapping]) -> dict[str, dict]:
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("inventory must be an iterable of company/ticker/fiscal_year records")
    catalog = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("inventory entries must be mappings")
        if record.get("status", "validated") != "validated":
            continue
        ticker, company, year = record.get("ticker"), record.get("company"), record.get("fiscal_year")
        if (not isinstance(ticker, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker)
                or not isinstance(company, str) or not company.strip()
                or isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2099):
            raise ValueError("inventory entries need a ticker, company, and integer fiscal_year")
        item = catalog.setdefault(ticker, {"company": company, "years": set()})
        if item["company"] != company:
            raise ValueError(f"Conflicting company names for ticker {ticker}")
        item["years"].add(year)
    return catalog


def _company_mentions(query: str, catalog: dict[str, dict]) -> list[dict]:
    names = {ticker: item["company"] for ticker, item in catalog.items()}
    for ticker, name in _KNOWN_COMPANIES.items():
        names.setdefault(ticker, name)
    mentions = []
    for ticker, company in names.items():
        aliases = (company, *_ALIASES.get(ticker, ()))
        for alias in aliases:
            pattern = r"(?<!\w)" + re.escape(_normalized(alias)) + r"(?!\w)"
            for match in re.finditer(pattern, query, re.I):
                # 'target gross margin' is an ordinary metric target. A capitalized
                # company name or possessive form is needed for this ambiguous name.
                if alias.casefold() == "target" and not (match.group() == "Target" or query[match.end():].startswith("'s")):
                    continue
                mentions.append({"ticker": ticker, "start": match.start(), "end": match.end(), "kind": "company"})
        if ticker not in _AMBIGUOUS_TICKERS:
            for match in re.finditer(r"(?<![\w$])" + re.escape(ticker) + r"(?!\w)", query):
                mentions.append({"ticker": ticker, "start": match.start(), "end": match.end(), "kind": "ticker"})
    # Explicit symbol forms also identify companies absent from this inventory.
    explicit = r"(?:\$|\b(?i:NASDAQ|NYSE)\s*:\s*|\b(?i:ticker|symbol)\s*[:=]?\s*)([A-Z][A-Z0-9.\-]{0,9})\b"
    for match in re.finditer(explicit, query):
        if match.group(1) in {"USD", "EUR", "GBP", "JPY", "CAD", "AUD"}:
            continue
        mentions.append({"ticker": match.group(1), "start": match.start(1), "end": match.end(1), "kind": "explicit_ticker"})
    unique = {(row["ticker"], row["start"], row["end"]): row for row in mentions}
    return sorted(unique.values(), key=lambda row: (row["start"], -row["end"]))


def _requested_mentions(query: str, mentions: list[dict]) -> list[dict]:
    requested = []
    distinct = {row["ticker"] for row in mentions}
    comparison = bool(re.search(r"\b(?:compar(?:e|ed|ing|ative)|versus|vs\.?|higher|lower|between|both)\b", query, re.I))
    for mention in mentions:
        before, after = query[:mention["start"]], query[mention["end"]:]
        if re.search(r"\b(?:excluding|except|not|rather than)\s+$", before, re.I):
            continue
        incidental = re.search(r"\b(?:about|from|against|including|such as|mention(?:s|ed|ing)?|customers?\s+(?:like|such as))\s+$", before, re.I)
        if len(distinct) > 1 and incidental and not comparison and not after.startswith("'s"):
            continue
        requested.append(mention)
    return requested


def _report_year_mentions(query: str) -> list[tuple[int, int, int]]:
    found = []
    for pattern in (_REPORT_BEFORE, _REPORT_AFTER):
        for match in pattern.finditer(query):
            found.append((int(match.group(1)), match.start(), match.end()))
    lists = []
    for pattern in (_REPORT_YEAR_LIST, _REPORT_YEAR_LIST_AFTER):
        for match in pattern.finditer(query):
            years = [int(value) for value in re.findall(_YEAR, match.group(1))]
            if len(years) == 2 and re.search(r"-|\b(?:to|through)\b", match.group(1), re.I) and 0 < years[1] - years[0] <= 10:
                years = list(range(years[0], years[1] + 1))
            lists.extend((year, match.start(), match.end()) for year in years)
    # A shared year list is one source clause. Drop its smaller single-year
    # matches so company-to-clause association does not accidentally split it.
    found = [row for row in found if not any(start <= row[1] and row[2] <= end for _, start, end in lists)]
    return sorted(set(found + lists), key=lambda row: row[1])


def _requested_reports(mentions: list[dict], report_years: list[tuple[int, int, int]]) -> list[dict]:
    tickers = list(dict.fromkeys(row["ticker"] for row in mentions))
    years = sorted({row[0] for row in report_years})
    spans = {(start, end) for _, start, end in report_years}
    # One shared clause, or a group of companies entirely before/after a list
    # of reports, applies all stated source years to all stated companies.
    grouped = bool(report_years) and (
        len(spans) == 1
        or all(row["end"] <= min(start for start, _ in spans) for row in mentions)
        or all(row["start"] >= max(end for _, end in spans) for row in mentions)
    )
    if len(tickers) <= 1 or len(years) <= 1 or grouped:
        pairs = {(ticker, year) for ticker in tickers for year in years}
    else:
        # Separate report clauses normally place the company next to its report
        # year. This prevents an erroneous Cartesian-product availability check.
        pairs = set()
        for year, start, end in report_years:
            nearest = min(mentions, key=lambda row: min(abs(start - row["end"]), abs(row["start"] - end)))
            pairs.add((nearest["ticker"], year))
    return [{"ticker": ticker, "fiscal_year": year} for ticker, year in sorted(pairs)]


def plan_query(query: str, inventory: Iterable[Mapping]) -> dict:
    """Return a JSON-compatible retrieval plan without network or model calls.

    ``inventory`` accepts manifest rows (company, ticker, fiscal_year); when a
    status is present, only validated rows count as available reports. The
    caller must derive this inventory from the corpus actually being searched.
    A covered status means explicit scope is available, not that the answer is.
    Unknown, unrecognized company names remain unspecified rather than guessed.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    normalized = _normalized(query)
    catalog = _inventory(inventory)
    mentions = _company_mentions(normalized, catalog)
    requested = _requested_mentions(normalized, mentions)
    recognized_tickers = list(dict.fromkeys(row["ticker"] for row in mentions))
    requested_tickers = list(dict.fromkeys(row["ticker"] for row in requested))
    report_mentions = _report_year_mentions(normalized)
    report_years = sorted({row[0] for row in report_mentions})
    reports = _requested_reports(requested, report_mentions)
    missing_tickers = sorted(set(requested_tickers) - catalog.keys())
    missing_reports = [row for row in reports if row["ticker"] in catalog and row["fiscal_year"] not in catalog[row["ticker"]]["years"]]
    # A source year can be unavailable even when no company was specified.
    if not requested_tickers:
        all_years = {year for company in catalog.values() for year in company["years"]}
        missing_reports.extend({"ticker": None, "fiscal_year": year} for year in report_years if year not in all_years)
    reasons = []
    if missing_tickers:
        reasons.append("company_not_in_corpus")
    if missing_reports:
        reasons.append("requested_report_not_in_corpus")
    metric_concepts, hints, matched = [], [], []
    for concept, pattern, alternatives in _METRICS:
        matches = [match for match in re.finditer(pattern, normalized, re.I)
                   if not re.search(r"\b(?:not|excluding|except|rather than)\s+$", normalized[:match.start()], re.I)]
        # The 'gross profit' substring of a margin is not a requested amount.
        if concept == "gross_profit":
            matches = [match for match in matches if not any(start <= match.start() and match.end() <= end for start, end in matched)]
        if not matches:
            continue
        metric_concepts.append(concept)
        if concept == "gross_margin":
            matched.extend((match.start(), match.end()) for match in matches)
        hints.extend(hint for hint in alternatives if hint.casefold() not in normalized.casefold())
    hints = list(dict.fromkeys(hints))[:2]
    bases = list(dict.fromkeys(match.group().upper().replace(" ", "-") for match in re.finditer(r"\b(?:non[- ]?GAAP|GAAP)\b", normalized, re.I)))
    currencies = list(dict.fromkeys(re.findall(r"\b(?:USD|EUR|GBP|JPY|CAD|AUD|CNY)\b|[$€£¥]", query)))
    return {
        "policy_version": POLICY_VERSION,
        "original_query": query, "dense_query": query,
        "reranker_query": query, "generation_query": query,
        "lexical_queries": [query, *(f"{query} {hint}" for hint in hints)],
        "lexical_hints": hints,
        "constraints": {
            "recognized_tickers": recognized_tickers, "requested_tickers": requested_tickers,
            "explicit_fiscal_years": sorted({int(match.group(1)) for match in _FISCAL_YEAR.finditer(normalized)}),
            "requested_report_years": report_years, "requested_reports": reports,
            "forecast_intent": bool(_FORECAST.search(normalized)),
            "metric_concepts": metric_concepts, "accounting_bases": bases, "currencies": currencies,
        },
        "coverage": {
            "status": "unavailable" if reasons else "covered" if requested_tickers or report_years else "unspecified",
            "reason_codes": reasons, "missing_tickers": missing_tickers, "missing_reports": missing_reports,
            "available_years": {ticker: sorted(catalog[ticker]["years"]) for ticker in requested_tickers if ticker in catalog},
        },
        "expansion_notes": [
            "Financial terms are bounded lexical retrieval hints, not accounting equivalences.",
            "Original wording, accounting basis, currency, entity, and period remain unchanged.",
            "Coverage checks identify explicit source gaps; they do not establish answer support.",
        ],
    }


def analyze_query(query: str, inventory: Iterable[Mapping]) -> dict:
    """Alias for callers that use the query analyzer without a retrieval stage."""
    return plan_query(query, inventory)
