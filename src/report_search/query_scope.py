"""Deterministic query scope, resolved once before retrieval and passed to generation.

A live audit of real questions showed scope errors that ranking cannot repair:
aliases ('coke'), ambiguous tickers ('HD'), typos ('starbuks'), two-digit fiscal
years ('FY23'), relative periods ('last yr') answered from an old report, silent
answers when no period was given, and years outside the corpus answered as if
they existed. This module resolves those interpretations from the corpus
inventory and states each one as a short note the answer model can disclose.

It never guesses beyond the inventory: an unknown company or an uncovered year
is reported, never mapped to the nearest available match. Matching is
deliberately conservative because a false company match silently changes which
reports are searched. Malformed inventories raise; arbitrary user text never does.
"""

from __future__ import annotations

import bisect
import difflib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import NamedTuple


SCOPE_VERSION = "query-scope-v1"
PERIOD_MODES = ("explicit", "relative_latest", "unspecified")
_MAX_NOTES = 8
_MAX_NOTE_CHARS = 220
_FUZZY_RATIO = 0.86
_MAX_RANGE_SPAN = 10
# A 10-K presents up to three fiscal years, so the earliest report still
# carries two prior years as comparative columns.
_COMPARATIVE_YEARS = 2

# ASCII-only boundaries: Chinese text often embeds Latin names without spaces
# ("英伟达NVIDIA"), and CJK characters count as \w in Python regexes.
_B = r"(?<![A-Za-z0-9_])"
_E = r"(?![A-Za-z0-9_])"
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")

_ALIASES: dict[str, tuple[str, ...]] = {
    "KO": ("coke", "coca-cola", "coca cola", "可口可乐"),
    "PEP": ("pepsi", "pepsico", "百事"),
    "HD": ("home depot", "家得宝"),
    "LOW": ("lowe's", "lowes", "劳氏"),
    "PG": ("p&g", "procter & gamble", "procter and gamble", "宝洁"),
    "CL": ("colgate", "colgate-palmolive", "高露洁"),
    "KMB": ("kimberly-clark", "kimberly clark"),
    "KHC": ("kraft heinz", "kraft-heinz", "卡夫亨氏"),
    "MDLZ": ("mondelez", "mondelēz", "亿滋"),
    "GIS": ("general mills",),
    "NVDA": ("nvidia", "英伟达"),
    "MSFT": ("microsoft", "微软"),
    "AMZN": ("amazon", "亚马逊"),
    "INTC": ("intel", "英特尔"),
    "AMD": ("amd", "advanced micro devices", "超威"),
    "QCOM": ("qualcomm", "高通"),
    "AVGO": ("broadcom", "博通"),
    "TXN": ("texas instruments", "德州仪器"),
    "ADI": ("analog devices", "亚德诺"),
    "MU": ("micron", "美光"),
    "ORCL": ("oracle", "甲骨文"),
    "CRM": ("salesforce", "赛富时"),
    "NOW": ("servicenow",),
    "ADBE": ("adobe", "奥多比"),
    "ADSK": ("autodesk", "欧特克"),
    "CDNS": ("cadence", "楷登"),
    "INTU": ("intuit",),
    "CAT": ("caterpillar", "卡特彼勒"),
    "DE": ("deere", "john deere", "迪尔"),
    "CMI": ("cummins", "康明斯"),
    "EMR": ("emerson", "艾默生"),
    "ITW": ("illinois tool works",),
    "DOV": ("dover",),
    "PH": ("parker hannifin", "parker-hannifin", "派克汉尼汾"),
    "ROK": ("rockwell automation", "rockwell"),
    "COST": ("costco", "好市多", "开市客"),
    "WMT": ("walmart", "wal-mart", "沃尔玛"),
    # 'Target' itself comes from the inventory name and has its own rule.
    "TGT": ("塔吉特",),
    "NKE": ("nike", "耐克"),
    "SBUX": ("starbucks", "星巴克"),
}

# Tickers that are also words or common abbreviations count only when written
# in uppercase inside mixed-case text ("HD or Lowe's", not "HD OR LOWE'S").
_AMBIGUOUS_TICKERS = frozenset({"CAT", "COST", "LOW", "NOW", "HD", "KO", "PG", "PH", "CL", "DE", "GIS", "ON", "IT", "TI"})
_WORD_TICKERS = frozenset({
    "PEP", "ALL", "ARE", "FOR", "KEY", "SEE", "BIG", "FAST", "WELL", "OPEN", "MAIN", "LIFE",
    "WORK", "PLAY", "FUN", "EAT", "RUN", "CAR", "NET", "BEST", "TRUE", "HAS", "LOVE", "MAN",
    "AIR", "GOOD", "SAVE", "HOLD", "CASH", "BOX", "TEAM", "SHOP", "SNOW", "ZONE", "DOCS",
})
_TICKER_ALIASES = {"TI": "TXN"}
_CAPS_CONNECTORS = frozenset({"VS", "AND", "OR", "VERSUS"})
# Only a ticker that is itself an English word can be part of an uppercase
# heading ("COST OF SALES", "NOW TRENDING"); "HD FCF" is still Home Depot.
_HEADING_WORD_TICKERS = frozenset({"CAT", "COST", "LOW", "NOW", "ON", "IT"}) | _WORD_TICKERS
# Uppercase finance abbreviations beside a ticker ("KO vs PEP EPS", "HD vs LOW
# ROE") are how people write tickers, not evidence of a heading.
_FINANCE_ACRONYMS = frozenset({
    "FY", "EPS", "FCF", "ROE", "ROA", "ROI", "ROIC", "EBIT", "EBITDA", "GAAP", "YOY", "YTD", "TTM", "LTM", "CAPEX",
    "OPEX", "SGA", "SG", "COGS", "PE", "DPS", "OCF", "DCF", "NI", "CEO", "CFO", "ESG", "AI", "USD", "US", "IPO",
})
_NON_TICKER_SYMBOLS = frozenset({
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CNY", "RMB", "HKD", "CHF", "MM", "BN", "MN", "K", "M", "B", "T",
    "US", "HK", "NT", "CA", "AU", "SG", "NZ", "TN", "MLN", "BLN", "BIL", "MIL",
})
# CJK aliases match as substrings, so a few need guards against common words
# that contain them: 高通胀 / 高通货膨胀 is "high inflation", 三星期 is "three weeks".
_CJK_GUARDS = {"高通": "(?!胀|货)", "三星": "(?!期)"}

# Well-known companies outside the corpus. Recognizing them lets the answer
# abstain explicitly instead of answering from a peer company's report.
# Each pattern is (regex source, flags, rule); rule "upper" = uppercase-only symbol.
_UNKNOWN_COMPANIES: tuple[tuple[str, tuple[str, ...], tuple[tuple[str, int, str], ...]], ...] = (
    ("Apple", ("AAPL",), (
        (_B + r"Apple(?![A-Za-z0-9_])(?!-to-)", 0, "plain"),
        (_B + r"apple(?=(?:'s|\s+(?:inc|revenues?|sales|stock|shares|iphones?|earnings|net|income|profits?"
         r"|annual|10-k|reports?|fiscal|fy))(?![A-Za-z0-9_]))", re.I, "plain"),
        (_B + r"aapl" + _E, re.I, "plain"),
        (r"苹果", 0, "plain"),
    )),
    ("Tesla", ("TSLA",), ((_B + r"(?:tesla|tsla)" + _E, re.I, "plain"), (r"特斯拉", 0, "plain"))),
    ("Alphabet", ("GOOGL", "GOOG"), (
        (_B + r"(?:alphabet|google|googl|goog)" + _E, re.I, "plain"), (r"谷歌", 0, "plain"),
    )),
    ("Meta", ("META",), (
        (_B + r"(?:meta\s+platforms|facebook)" + _E, re.I, "plain"),
        (_B + r"Meta(?![A-Za-z0-9_\-])", 0, "plain"),
        (_B + r"META" + _E, 0, "upper"),
    )),
    ("Netflix", ("NFLX",), ((_B + r"(?:netflix|nflx)" + _E, re.I, "plain"), (r"网飞|奈飞", 0, "plain"))),
    ("Berkshire Hathaway", ("BRK.A", "BRK.B"), (
        (_B + r"berkshire(?:\s+hathaway)?" + _E, re.I, "plain"),
        (_B + r"BRK\.?[AB]" + _E, 0, "plain"),
        (r"伯克希尔", 0, "plain"),
    )),
    ("Samsung", (), ((_B + r"samsung" + _E, re.I, "plain"), (r"三星" + _CJK_GUARDS["三星"], 0, "plain"))),
    ("TSMC", ("TSM",), (
        (_B + r"(?:tsmc|taiwan\s+semiconductor)" + _E, re.I, "plain"),
        (_B + r"TSM" + _E, 0, "upper"),
        (r"台积电", 0, "plain"),
    )),
)

# Typo matching compares single words, so these must never be treated as a
# misspelled company even when they are a near-subsequence of one ("lowest"
# vs "lowes", "micro" vs "micron", "decadence" vs "cadence").
_FUZZY_STOPWORDS = frozenset("""
about above access across actual after again against along among amount annual another answer apple apples
asset assets audit based basis before being below billion board bonds brand build business capex capital
cards carry cases change charge cheap check claim class close closed comes company compare compared costs
could count cover covers credit current daily dealer debts decade decadence decline deliver dividend dividends
dollar dollars doing during early earned earnings either emerging entire equity every exact excluding expense
expenses facts fiscal floor flows focus force forces funds given gonna gross group growth guess guide happen
heavy higher holds hours house income index inside intellect items labor large later latest leader lease least
level lines liquid loans local lower lowered lowers lowest slower slowest major makes margin margins market
maybe means media merger metric micro omicron might million model money month months nearly needs never notes
number offer often order other otherwise outlook owner parent parts party people period place plans point power
press price prices prior profit profits quarter quarters ratio rates really recent report reports return revenue
revenues right rough round rules sales scale score segment segments share shares should since small sound spend
split staff stake start state still stock stocks store stores story study table taken target targets targeted
targeting taxes terms their there these thing think those three today total trade trend under units until value
watch where which while whose world worth would years yield proctor collate cummings parked darker barker
""".split())
# First words that do not identify a company on their own ("General Mills").
_GENERIC_NAME_WORDS = frozenset({
    "general", "texas", "analog", "advanced", "illinois", "home", "international", "the", "united", "american",
    "national", "first", "global", "digital", "micro", "data", "applied", "electric", "systems", "design",
})
_CORPORATE_SUFFIX = re.compile(
    r"(?:[\s,&]+(?:inc|incorporated|corporation|corp|company|co|international|holdings|group|plc|ltd|limited"
    r"|technology|technologies)\.?)+$", re.I)

# Two-digit years are accepted only in unmistakable forms ("FY23", "fiscal '23");
# "FY 10-K" or "fiscal 52 weeks" are not years, but "FY23 Q4" and "FY23-Q4" are.
# ASCII digits only: int() would also accept other scripts' digits.
_FY_YEAR = re.compile(
    r"(?<![A-Za-z0-9_])FY\s*[-']?\s*([0-9]{4}|[0-9]{2}"
    r"(?!\s*-\s*[A-Za-z](?![A-Za-z0-9])|[KQF](?![A-Za-z0-9])|\s*weeks?(?![A-Za-z])))(?![0-9])", re.I)
_FISCAL_YEAR = re.compile(r"(?<![A-Za-z])fiscal[\s\-]+(?:years?[\s\-]+)?([0-9]{4}|'\s*[0-9]{2})(?![0-9])", re.I)
# Standalone years are limited to 1990-2099; "2026E" is an analyst estimate year.
_PLAIN_YEAR = re.compile(r"(?<![A-Za-z0-9_])(?:199[0-9]|20[0-9]{2})(?=E?(?![A-Za-z0-9_]))")
_ESTIMATE_SUFFIX = re.compile(r"E(?![A-Za-z0-9_])")
_AMOUNT_BEFORE = re.compile(r"(?:[$€£¥]\s?|\d\.)$")
_AMOUNT_AFTER = re.compile(
    r"\.\d|,\d{3}(?!\d)|\s*(?:%|percent|pct|bps|basis\s+points|(?:million|billion|thousand|trillion)s?|mn|bn|mm)"
    r"(?![A-Za-z])|\s*(?:亿|万|千|百万|元|美元)", re.I)
_FISCAL_SUFFIX = re.compile(r"\s*财年")
_RANGE_JOIN = re.compile(r"\s*(?:年度?|财年度?)?\s*(?:-|~|to|through|thru|until|至|到)\s*", re.I)

_RELATIVE = re.compile(
    r"(?<![A-Za-z])(?:last\s+(?:fiscal\s+)?(?:year|yr|fy)|past\s+(?:fiscal\s+)?year|this\s+(?:fiscal\s+)?year"
    r"|current\s+(?:fiscal\s+)?year|most\s+recent(?:ly)?|latest|currently"
    # Balance-sheet terms such as "current assets" are not a period.
    r"|current(?!\s+(?:assets?|liabilit|ratio|portion|maturit|debt|income\s+tax|tax|deferred|investments?"
    r"|receivable|payable|accounts?|operating\s+lease|borrowings?|notes?))"
    r"|recently|recent|right\s+now|now|today)(?![A-Za-z])"
    r"|最新|去年|今年|最近|目前|现在|当前",
    re.I,
)
_FORECAST = re.compile(
    r"(?<![A-Za-z])(?:forecast(?:s|ed|ing)?|projections?|projected(?!\s+benefit)|guidance|outlook"
    r"|predict(?:s|ed|ing|ion|ions)?|(?<!accounting\s)(?<!critical\s)(?<!use\sof\s)estimat(?:e|es|ing)"
    r"|guess(?:es|ed|ing)?|gonna|going\s+to|will\s+(?:[\w'.&$-]+\s+){0,3}?(?:make|earn|be)"
    r"|next\s+(?:fiscal\s+)?(?:year|yr|quarter))"
    r"(?![A-Za-z])|明年|预测|预计",
    re.I,
)
_COMPARISON = re.compile(
    r"(?<![A-Za-z])(?:vs\.?|versus|compar(?:e|es|ed|ing|ison)|higher|lower|better|worse|bigger|smaller"
    r"|which|whose|both|between|or)(?![A-Za-z])|对比|比较|相比|还是|哪个|哪家",
    re.I,
)
_EXPLICIT_SYMBOL = re.compile(
    r"(?<![A-Za-z0-9_$])\$([A-Za-z]{1,5}(?:\.[A-Za-z])?)(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z])(?:NASDAQ|NYSE|AMEX)\s*:\s*([A-Za-z]{1,5}(?:\.[A-Za-z])?)(?![A-Za-z0-9_])",
    re.I,
)

# Same-length substitutions only, so match offsets stay valid for the original.
_TRANSLATION = {ord(char): "'" for char in "\u2019\u2018\u02bc\u2032"}
_TRANSLATION.update({ord(char): '"' for char in "\u201c\u201d\u2033"})
_TRANSLATION.update({ord(char): "-" for char in "\u2013\u2010\u2011\u2212"})
_TRANSLATION.update({ord(char): " " for char in "\u00a0\u2007\u202f\u3000\u200b"})
_TRANSLATION.update({0xFF01 + offset: chr(0x21 + offset) for offset in range(94)})  # full-width ASCII


class _ReadOnlyDict(dict):
    """A dict that refuses mutation but still pickles, deep-copies and JSON-encodes.

    A scope is shared by retrieval and generation and must not drift between
    the two; MappingProxyType would also forbid writes, but it breaks
    ``copy.deepcopy``, ``pickle`` and ``dataclasses.asdict`` on the scope.
    """

    def _readonly(self, *args, **kwargs):
        raise TypeError("QueryScope mappings are read-only")

    __setitem__ = __delitem__ = __ior__ = _readonly
    clear = pop = popitem = setdefault = update = _readonly

    def __reduce__(self):
        return type(self), (dict(self),)


@dataclass(frozen=True)
class QueryScope:
    """What a question asks about, in corpus terms, plus notes explaining each interpretation."""

    tickers: tuple[str, ...] = ()
    unknown_companies: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    period_mode: str = "unspecified"
    report_years: Mapping[str, tuple[int, ...]] = field(default_factory=dict, hash=False)
    latest_year: Mapping[str, int] = field(default_factory=dict, hash=False)
    out_of_coverage_years: tuple[int, ...] = ()
    comparison: bool = False
    forecast_intent: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.period_mode not in PERIOD_MODES:
            raise ValueError(f"period_mode must be one of {PERIOD_MODES}")
        # Read-only copies: a scope is shared by retrieval and generation and
        # must not drift between the two.
        for name in ("tickers", "unknown_companies", "years", "out_of_coverage_years", "notes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "report_years", _ReadOnlyDict(
            {str(ticker): tuple(years) for ticker, years in dict(self.report_years).items()}))
        object.__setattr__(self, "latest_year", _ReadOnlyDict(dict(self.latest_year)))

    def to_dict(self) -> dict:
        return {
            "version": SCOPE_VERSION,
            "tickers": list(self.tickers),
            "unknown_companies": list(self.unknown_companies),
            "years": list(self.years),
            "period_mode": self.period_mode,
            "report_years": {ticker: list(years) for ticker, years in self.report_years.items()},
            "latest_year": dict(self.latest_year),
            "out_of_coverage_years": list(self.out_of_coverage_years),
            "comparison": self.comparison,
            "forecast_intent": self.forecast_intent,
            "notes": list(self.notes),
        }


class _Pattern(NamedTuple):
    regex: re.Pattern
    key: str
    known: bool
    rule: str  # "plain" | "upper" | "target"


class _Mention(NamedTuple):
    start: int
    end: int
    key: str      # corpus ticker, or a display name for unknown companies
    known: bool
    surface: str
    fuzzy: bool = False


class _YearMention(NamedTuple):
    start: int
    end: int
    year: int
    fiscal_label: bool
    estimate: bool = False  # "2026E" / "FY26E": an analyst estimate, not a reported year


class ScopeResolver:
    """Resolve companies and report years against a fixed, validated inventory."""

    def __init__(self, inventory: Iterable[Mapping]) -> None:
        catalog = _catalog(inventory)
        self._names = {ticker: company for ticker, (company, _) in catalog.items()}
        self._years = {ticker: years for ticker, (_, years) in catalog.items()}
        self._corpus_years = tuple(sorted({year for years in self._years.values() for year in years}))
        self._unknown = [entry for entry in _UNKNOWN_COMPANIES if not self._in_corpus(entry)]
        self._unknown_symbols = {symbol: name for name, symbols, _ in self._unknown for symbol in symbols}
        self._symbols = frozenset(self._names) | frozenset(_TICKER_ALIASES) | frozenset(self._unknown_symbols)
        self._patterns = self._build_patterns()
        self._fuzzy_candidates = self._build_fuzzy_candidates()

    def resolve(self, query: str) -> QueryScope:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        text = query.translate(_TRANSLATION)
        mentions = self._mentions(text)
        tickers = tuple(dict.fromkeys(row.key for row in mentions if row.known))
        unknown = tuple(dict.fromkeys(row.key for row in mentions if not row.known))
        year_mentions = _year_mentions(text)
        years = _expand_years(text, year_mentions)
        relative = None if years else _relative_phrase(text, mentions)
        period_mode = "explicit" if years else "relative_latest" if relative else "unspecified"
        latest = {ticker: self._years[ticker][-1] for ticker in tickers}

        if tickers:
            earliest_all = min(self._years[ticker][0] for ticker in tickers)
            latest_all = max(latest.values())
        else:
            earliest_all, latest_all = self._corpus_years[0], self._corpus_years[-1]
        out_of_coverage = tuple(year for year in years
                                if year > latest_all or year < earliest_all - _COMPARATIVE_YEARS)

        report_years, coverage = self._report_years(tickers, years, period_mode, latest)
        forecast = (bool(_FORECAST.search(text)) or any(row.estimate for row in year_mentions)
                    or any(year > latest_all for year in years)
                    or any(year > latest[ticker] for ticker in tickers for year in years))
        comparison = len(tickers) >= 2 or (
            len(tickers) + len(unknown) >= 2 and bool(_COMPARISON.search(text)))

        notes = self._notes(
            mentions=mentions, tickers=tickers, unknown=unknown, period_mode=period_mode,
            relative=relative, latest=latest, out_of_coverage=out_of_coverage, coverage=coverage,
            forecast=forecast, calendar_years=sorted(
                {row.year for row in year_mentions if not row.fiscal_label}
                - {row.year for row in year_mentions if row.fiscal_label}),
        )
        return QueryScope(
            tickers=tickers, unknown_companies=unknown, years=years, period_mode=period_mode,
            report_years=report_years, latest_year=latest, out_of_coverage_years=out_of_coverage,
            comparison=comparison, forecast_intent=forecast, notes=notes,
        )

    def companies_in_text(self, text: str) -> frozenset[str]:
        """Corpus tickers named in arbitrary text, e.g. a generated claim."""
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return frozenset(row.key for row in self._mentions(text.translate(_TRANSLATION)) if row.known)

    @property
    def company_names(self) -> dict[str, str]:
        """Ticker -> inventory company name."""
        return dict(self._names)

    def mention_spans(self, text: str) -> list[tuple[int, int, str]]:
        """(start, end, ticker) for each corpus company mention, in text order.

        Used to rewrite a comparison into per-company sub-questions by removing
        the other companies' names.
        """
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        translated = text.translate(_TRANSLATION)  # one-to-one, so offsets are unchanged
        return [(row.start, row.end, row.key) for row in self._mentions(translated) if row.known]

    def entities_in_text(self, text: str) -> frozenset[str]:
        """Corpus tickers plus 'unknown:<name>' for companies outside the corpus.

        Claim checks use this so that "Apple's revenue was ..." cited to a
        Microsoft table fails: no cited passage can belong to Apple unless the
        cited text itself names Apple.
        """
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        return frozenset(row.key if row.known else f"unknown:{row.key}"
                         for row in self._mentions(text.translate(_TRANSLATION)))

    # -- company matching -------------------------------------------------

    def _in_corpus(self, entry: tuple) -> bool:
        name, symbols, _ = entry
        names = {company.casefold() for company in self._names.values()}
        return name.casefold() in names or any(symbol in self._names for symbol in symbols)

    def _build_patterns(self) -> list[_Pattern]:
        patterns = []
        for ticker, company in self._names.items():
            surfaces = dict.fromkeys((company, *_name_variants(company), *_ALIASES.get(ticker, ())))
            for surface in surfaces:
                rule = "target" if surface.casefold() == "target" else "plain"
                patterns.append(_Pattern(_surface_regex(surface), ticker, True, rule))
            if len(ticker) >= 3 and ticker not in _AMBIGUOUS_TICKERS and ticker not in _WORD_TICKERS:
                patterns.append(_Pattern(re.compile(_B + re.escape(ticker) + _E, re.I), ticker, True, "plain"))
            else:
                patterns.append(_Pattern(_upper_token(ticker), ticker, True, "upper"))
        for symbol, ticker in _TICKER_ALIASES.items():
            if ticker in self._names:
                patterns.append(_Pattern(_upper_token(symbol), ticker, True, "upper"))
        for name, _, specs in self._unknown:
            for source, flags, rule in specs:
                patterns.append(_Pattern(re.compile(source, flags), name, False, rule))
        return patterns

    def _build_fuzzy_candidates(self) -> tuple[tuple[str, str], ...]:
        pairs = {}
        for ticker, company in self._names.items():
            forms = [re.split(r"[\s\-]+", company.strip())[0], company, *_ALIASES.get(ticker, ())]
            for form in forms:
                candidate = re.sub(r"[\s\-'.&]", "", form).casefold()
                if (len(candidate) >= 5 and candidate.isascii() and candidate.isalpha()
                        and candidate not in _GENERIC_NAME_WORDS and not candidate.startswith("target")):
                    pairs[(candidate, ticker)] = None
        return tuple(pairs)

    def _mentions(self, text: str) -> list[_Mention]:
        all_caps = _all_caps(text)
        found = []
        for pattern in self._patterns:
            for match in pattern.regex.finditer(text):
                if pattern.rule == "target" and not _is_target_company(text, match):
                    continue
                if pattern.rule == "upper" and (all_caps or self._in_caps_phrase(text, match)):
                    continue
                found.append(_Mention(match.start(), match.end(), pattern.key, pattern.known, match.group()))
        found.extend(self._explicit_symbols(text))
        kept = _non_overlapping(found)
        kept.extend(self._fuzzy_mentions(text, kept))
        kept = _drop_incidental_words(text, kept)
        return sorted(kept, key=lambda row: (row.start, row.end))

    def _in_caps_phrase(self, text: str, match: re.Match) -> bool:
        # "COST OF SALES" or "TOTAL COST" is a heading, not Costco: an uppercase
        # neighbouring word that is not a ticker, connector or finance acronym
        # marks an all-caps phrase. Non-word tickers ("HD", "KO") never form one.
        if match.group() not in _HEADING_WORD_TICKERS:
            return False
        start, end = match.start(), match.end()
        before = re.search(r"(?<![A-Za-z0-9_])([A-Za-z]+)[ \t]+$", text[max(0, start - 40):start])
        after = re.match(r"[ \t]+([A-Za-z]+)(?![A-Za-z0-9_])", text[end:end + 40])
        for neighbour in (before, after):
            if neighbour:
                word = neighbour.group(1)
                if (len(word) >= 2 and word.isupper() and word not in self._symbols
                        and word not in _CAPS_CONNECTORS and word not in _FINANCE_ACRONYMS):
                    return True
        return False

    def _explicit_symbols(self, text: str) -> list[_Mention]:
        found = []
        for match in _EXPLICIT_SYMBOL.finditer(text):
            group = 1 if match.group(1) is not None else 2
            written = match.group(group)
            symbol = written.upper()
            ticker = _TICKER_ALIASES.get(symbol, symbol)
            if ticker in self._names:
                found.append(_Mention(match.start(), match.end(), ticker, True, match.group()))
            elif symbol in self._unknown_symbols:
                found.append(_Mention(match.start(), match.end(), self._unknown_symbols[symbol], False, match.group()))
            elif written.isupper() and len(symbol) >= 2 and symbol not in _NON_TICKER_SYMBOLS:
                # An explicit symbol outside the corpus is still a requested company.
                found.append(_Mention(match.start(), match.end(), symbol, False, match.group()))
        return found

    def _fuzzy_mentions(self, text: str, exact: list[_Mention]) -> list[_Mention]:
        found, cache, spans = [], {}, _Spans(exact)
        for match in re.finditer(r"[A-Za-z]+", text):
            word = match.group().casefold()
            if (len(word) < 5 or word in _FUZZY_STOPWORDS or word.startswith("target")
                    or spans.overlaps(match.start(), match.end())):
                continue
            if word not in cache:
                cache[word] = self._fuzzy_ticker(word)
            if cache[word]:
                found.append(_Mention(match.start(), match.end(), cache[word], True, match.group(), fuzzy=True))
        return found

    def _fuzzy_ticker(self, word: str) -> str | None:
        winners = set()
        for candidate, ticker in self._fuzzy_candidates:
            if 2 * min(len(word), len(candidate)) / (len(word) + len(candidate)) < _FUZZY_RATIO:
                continue
            matcher = difflib.SequenceMatcher(None, word, candidate)
            if matcher.quick_ratio() >= _FUZZY_RATIO and matcher.ratio() >= _FUZZY_RATIO:
                winners.add(ticker)
        # Only an unambiguous typo is accepted; a tie means the word is unclear.
        return next(iter(winners)) if len(winners) == 1 else None

    # -- periods ------------------------------------------------------------

    def _report_years(self, tickers, years, period_mode, latest):
        report_years, coverage = {}, {"beyond": {}, "too_early": {}, "comparative": {}, "missing": {}}
        for ticker in tickers:
            if period_mode == "relative_latest":
                report_years[ticker] = (latest[ticker],)
                continue
            if period_mode == "unspecified":
                report_years[ticker] = ()
                continue
            available = self._years[ticker]
            chosen = set()
            for year in years:
                if year in available:
                    chosen.add(year)
                elif year > available[-1]:
                    coverage["beyond"].setdefault(ticker, []).append(year)
                elif year < available[0] - _COMPARATIVE_YEARS:
                    coverage["too_early"].setdefault(ticker, []).append(year)
                else:
                    sources = tuple(later for later in range(year + 1, year + 1 + _COMPARATIVE_YEARS)
                                    if later in available)
                    if sources:
                        chosen.update(sources)
                        key = (year, sources, year < available[0])
                        coverage["comparative"].setdefault(key, []).append(ticker)
                    else:
                        coverage["missing"].setdefault(ticker, []).append(year)
            # No usable report year means no filter, never an empty search.
            report_years[ticker] = tuple(sorted(chosen))
        return report_years, coverage

    # -- notes ------------------------------------------------------------

    def _notes(self, *, mentions, tickers, unknown, period_mode, relative, latest,
               out_of_coverage, coverage, forecast, calendar_years) -> tuple[str, ...]:
        name = self._names.__getitem__
        notes = []
        if unknown:
            count = len(self._names)
            if len(unknown) == 1:
                notes.append(f"{unknown[0]} is not in the corpus of {count} companies; "
                             f"no {unknown[0]} annual report is available.")
            else:
                notes.append(f"{_join(list(unknown))} are not in the corpus of {count} companies; "
                             "no annual reports are available for them.")
        elif not tickers:
            notes.append("No company from the corpus was identified in the question.")

        exact = {row.key for row in mentions if not row.fuzzy}
        fuzzy = {row.surface: row.key for row in mentions if row.fuzzy and row.key not in exact}
        if fuzzy:
            notes.append("; ".join(f"'{surface}' was interpreted as {name(ticker)}"
                                   for surface, ticker in fuzzy.items()) + ".")
        if forecast:
            notes.append("The question asks about future or projected results; annual reports contain historical "
                         "results and at most management's stated outlook, so do not estimate figures.")

        if not tickers:
            beyond = [year for year in out_of_coverage if year > self._corpus_years[-1]]
            early = [year for year in out_of_coverage if year <= self._corpus_years[-1]]
            if beyond:
                notes.append(f"Requested {_years_phrase(beyond)} not covered: the corpus contains reports "
                             f"up to FY{self._corpus_years[-1]} only.")
            if early:
                notes.append(f"Requested {_years_phrase(early)} not covered: the earliest corpus reports are "
                             f"FY{self._corpus_years[0]}, with comparatives back to "
                             f"{self._corpus_years[0] - _COMPARATIVE_YEARS} at most.")
        for group_years, group in _group(coverage["beyond"]).items():
            if len(group) == 1:
                tail = f"the latest available {name(group[0])} report is FY{latest[group[0]]}."
            else:
                tail = f"the latest available reports are {_join([f'{name(t)} FY{latest[t]}' for t in group])}."
            notes.append(f"Requested {_years_phrase(group_years)} not covered: {tail}")
        for group_years, group in _group(coverage["too_early"]).items():
            first = min(self._years[ticker][0] for ticker in group)
            notes.append(f"Requested {_years_phrase(group_years)} not covered: the earliest available "
                         f"{_reports(group, name)} {'is' if len(group) == 1 else 'are'} FY{first}, with "
                         f"comparatives back to {first - _COMPARATIVE_YEARS} at most.")
        for (year, sources, before_corpus), group in coverage["comparative"].items():
            source_text = " or ".join(f"FY{source}" for source in sources)
            if before_corpus:
                notes.append(f"Year {year} predates the earliest {_reports(group, name)} (FY{sources[0]}); it may "
                             f"appear only as a prior-year comparative in {source_text} reports.")
            else:
                notes.append(f"No FY{year} {_reports(group, name)} {'is' if len(group) == 1 else 'are'} "
                             f"available; {year} figures may appear only as prior-year comparatives in "
                             f"{source_text} reports.")
        for group_years, group in _group(coverage["missing"]).items():
            notes.append(f"No {_reports(group, name)} {'covers' if len(group) == 1 else 'cover'} "
                         f"{_years_list(group_years)}.")

        if period_mode == "relative_latest":
            if tickers:
                resolved = _join([f"{name(ticker)} FY{latest[ticker]}" for ticker in tickers])
                notes.append(f"'{relative}' was interpreted as the latest available fiscal year: {resolved}.")
            else:
                notes.append(f"'{relative}' was interpreted as the latest available fiscal year; the corpus "
                             f"contains reports up to FY{self._corpus_years[-1]}.")
        elif period_mode == "unspecified" and tickers:
            notes.append(_unspecified_note(tickers, latest, name))

        matched = [year for year in calendar_years if any(year in self._years[ticker] for ticker in tickers)]
        if period_mode == "explicit" and matched:
            plural = len(matched) > 1
            notes.append(f"Calendar year{'s' if plural else ''} {_join([str(y) for y in matched])} "
                         f"{'were' if plural else 'was'} read as the fiscal year label"
                         f"{'s' if plural else ''} {_join([f'FY{y}' for y in matched])}. A fiscal year can end "
                         "in a different calendar year; state the period end only if the sources show it.")
        return tuple(_clip(note) for note in notes[:_MAX_NOTES])


def _catalog(inventory: Iterable[Mapping]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Validated reports only; a malformed row is a corpus bug, so it raises."""
    if isinstance(inventory, (str, bytes, Mapping)) or not isinstance(inventory, Iterable):
        raise ValueError("inventory must be an iterable of company/ticker/fiscal_year records")
    companies, years = {}, {}
    for row in inventory:
        if not isinstance(row, Mapping):
            raise ValueError("inventory rows must be mappings")
        status = row.get("status", "validated")
        if not isinstance(status, str):
            raise ValueError("inventory status must be a string when present")
        if status != "validated":
            continue
        ticker, company, year = row.get("ticker"), row.get("company"), row.get("fiscal_year")
        if not isinstance(ticker, str) or not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", ticker):
            raise ValueError(f"invalid inventory ticker: {ticker!r}")
        if not isinstance(company, str) or not company.strip():
            raise ValueError(f"invalid inventory company for {ticker}")
        if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2099:
            raise ValueError(f"invalid inventory fiscal_year for {ticker}: {year!r}")
        company = company.strip()
        if companies.setdefault(ticker, company) != company:
            raise ValueError(f"conflicting company names for ticker {ticker}")
        years.setdefault(ticker, set()).add(year)
    names = [company.casefold() for company in companies.values()]
    if len(set(names)) != len(names):
        raise ValueError("one company name is assigned to several tickers")
    if not companies:
        raise ValueError("inventory has no validated reports")
    return {ticker: (companies[ticker], tuple(sorted(years[ticker]))) for ticker in companies}


def _name_variants(company: str) -> tuple[str, ...]:
    stripped = _CORPORATE_SUFFIX.sub("", company).strip(" ,&")
    if stripped and stripped != company and len(stripped) >= 4 and stripped.casefold() not in _GENERIC_NAME_WORDS:
        return (stripped,)
    return ()


def _surface_regex(surface: str) -> re.Pattern:
    if _CJK.search(surface):
        return re.compile(re.escape(surface) + _CJK_GUARDS.get(surface, ""))
    parts = []
    for char in re.sub(r"\s*&\s*", "&", surface.strip()):
        if char in " -":
            if parts and parts[-1] == r"[\s\-]+":
                continue
            parts.append(r"[\s\-]+")
        elif char == "&":
            parts.append(r"\s*&\s*")
        elif char == "'":
            parts.append("'?")
        elif char == ".":
            parts.append(r"\.?")
        else:
            parts.append(re.escape(char))
    return re.compile(_B + "".join(parts) + _E, re.I)


def _upper_token(symbol: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z0-9_&])" + re.escape(symbol) + r"(?![A-Za-z0-9_&])")


def _is_target_company(text: str, match: re.Match) -> bool:
    # "target gross margin" is a metric target; the retailer needs "Target" or "target's".
    return match.group() == "Target" or text[match.end():match.end() + 2].casefold() == "'s"


def _all_caps(text: str) -> bool:
    return (any(char.isupper() for char in text) and not any(char.islower() for char in text)
            and not _CJK.search(text))


class _Spans:
    """Overlap lookups against sorted, non-overlapping spans in O(log n).

    A linear scan per word made long inputs (a pasted answer, a repeated
    name) quadratic; resolve() and companies_in_text() must stay fast on any str.
    """

    def __init__(self, rows) -> None:
        self._starts = [row.start for row in rows]
        self._ends = [row.end for row in rows]

    def overlaps(self, start: int, end: int) -> bool:
        index = bisect.bisect_left(self._starts, end) - 1
        return index >= 0 and self._ends[index] > start


# Company names that are also ordinary words ("product cadence", "competitive
# intel", "Amazon rainforest" aside). Lowercase, they count only when they are
# the sole company in the text or written as a possessive ("intuit's").
_WORD_LIKE_NAMES = frozenset({"cadence", "intel", "micron", "oracle", "dover", "intuit", "amazon",
                              "parker", "rockwell", "emerson"})


def _drop_incidental_words(text: str, mentions: list[_Mention]) -> list[_Mention]:
    def incidental(row: _Mention) -> bool:
        return (row.known and not row.fuzzy and row.surface.islower() and row.surface in _WORD_LIKE_NAMES
                and text[row.end:row.end + 2] != "'s")  # text is already translated: curly -> ASCII

    others = {row.key for row in mentions if not incidental(row)}
    return [row for row in mentions if not incidental(row) or not (others - {row.key})]


def _non_overlapping(mentions: list[_Mention]) -> list[_Mention]:
    kept, end = [], -1
    for row in sorted(mentions, key=lambda row: (row.start, row.start - row.end)):
        if row.start >= end:
            kept.append(row)
            end = row.end
    return kept


def _year_value(written: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", written)  # "' 23" or "'\t23" -> "23"
    value = int(digits)
    if len(digits) == 2:
        value += 1900 if value >= 90 else 2000
    return value if 1990 <= value <= 2099 else None


def _is_amount(text: str, start: int, end: int) -> bool:
    # "$2,023", "2023.5" and "2023 million" are amounts, not periods.
    return bool(_AMOUNT_BEFORE.search(text[max(0, start - 2):start]) or _AMOUNT_AFTER.match(text, end))


def _year_mentions(text: str) -> list[_YearMention]:
    found = []
    for regex in (_FY_YEAR, _FISCAL_YEAR):
        for match in regex.finditer(text):
            year = _year_value(match.group(1))
            if year is not None and not _is_amount(text, match.start(1), match.end(1)):
                estimate = bool(_ESTIMATE_SUFFIX.match(text, match.end(1)))
                found.append(_YearMention(match.start(), match.end(), year, True, estimate))
    for match in _PLAIN_YEAR.finditer(text):
        if not _is_amount(text, match.start(), match.end()):
            fiscal = bool(_FISCAL_SUFFIX.match(text, match.end()))
            estimate = bool(_ESTIMATE_SUFFIX.match(text, match.end()))
            found.append(_YearMention(match.start(), match.end(), int(match.group()), fiscal, estimate))
    # "fiscal 2023" also contains a plain "2023"; the earlier, longer FY form wins.
    return _non_overlapping(found)


def _expand_years(text: str, mentions: list[_YearMention]) -> tuple[int, ...]:
    years = {row.year for row in mentions}
    for left, right in zip(mentions, mentions[1:]):
        if (_RANGE_JOIN.fullmatch(text[left.end:right.start])
                and 0 < right.year - left.year <= _MAX_RANGE_SPAN):
            years.update(range(left.year, right.year + 1))
    return tuple(sorted(years))


def _relative_phrase(text: str, mentions: list[_Mention]) -> str | None:
    spans = _Spans(mentions)
    for match in _RELATIVE.finditer(text):
        # An uppercase "NOW" already identified ServiceNow; it is not a period.
        if not spans.overlaps(match.start(), match.end()):
            return " ".join(match.group().split())
    return None


def _unspecified_note(tickers, latest, name) -> str:
    years = {latest[ticker] for ticker in tickers}
    if len(years) == 1:
        year = next(iter(years))
        if len(tickers) == 1:
            head = f"the latest available report for {name(tickers[0])} is FY{year}."
        else:
            head = f"the latest available reports for {_join([name(t) for t in tickers])} are FY{year}."
        return (f"No period was specified; {head} Answer for FY{year} unless the question clearly asks "
                "otherwise, and state the fiscal year.")
    resolved = _join([f"{name(ticker)} FY{latest[ticker]}" for ticker in tickers])
    return (f"No period was specified; the latest available reports are {resolved}. Answer for those fiscal "
            "years unless the question clearly asks otherwise, and state the fiscal year.")


def _group(by_ticker: dict[str, list[int]]) -> dict[tuple[int, ...], list[str]]:
    grouped = {}
    for ticker, years in by_ticker.items():
        grouped.setdefault(tuple(sorted(set(years))), []).append(ticker)
    return grouped


def _reports(tickers, name) -> str:
    return f"{_join([name(ticker) for ticker in tickers])} report{'' if len(tickers) == 1 else 's'}"


def _years_list(years) -> str:
    years = sorted(set(years))
    label = "year" if len(years) == 1 else "years"
    return f"{label} {_join([str(year) for year in years])}"


def _years_phrase(years) -> str:
    return _years_list(years) + (" is" if len(set(years)) == 1 else " are")


def _join(items: list[str], limit: int = 4) -> str:
    if len(items) > limit:
        items = [*items[:limit], f"{len(items) - limit} more"]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _clip(note: str) -> str:
    return note if len(note) <= _MAX_NOTE_CHARS else note[:_MAX_NOTE_CHARS - 3].rstrip() + "..."
