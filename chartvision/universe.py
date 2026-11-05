"""A universe of a few hundred liquid, large-cap US stocks grouped by sector.

The list is embedded so the pipeline runs offline and reproducibly. Pick one
or two sectors (``Config.sectors``) to get a set of relatively similar
stocks, or take everything for a broad sample.
"""

from __future__ import annotations

UNIVERSE: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "ACN",
        "INTU", "IBM", "TXN", "QCOM", "NOW", "AMAT", "PANW", "ANET", "MU", "LRCX",
        "ADI", "KLAC", "CRWD", "SNPS", "CDNS", "APH", "MSI", "ROP", "FTNT", "ADSK",
        "NXPI", "MCHP", "TEL", "IT", "FICO", "HPQ", "CTSH", "GLW", "ON", "CDW",
        "KEYS", "HPE", "ANSS", "TYL", "NTAP", "WDC", "STX", "TER", "PTC", "ZBRA",
        "TDY", "AKAM", "SWKS", "JBL", "FSLR", "ENPH", "GEN", "EPAM", "FFIV", "QRVO",
        "JNPR", "TRMB", "VRSN", "GDDY", "SMCI", "DELL", "PLTR", "DDOG", "SNOW", "WDAY",
        "TEAM", "ZS", "NET", "MDB", "HUBS", "OKTA", "DOCU", "TWLO", "ARM", "MRVL",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR", "EA",
        "TTWO", "WBD", "OMC", "IPG", "LYV", "MTCH", "FOXA", "NWSA", "PARA", "RBLX",
        "SPOT", "PINS", "SNAP", "ROKU", "TTD",
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG",
        "ORLY", "MAR", "GM", "F", "HLT", "AZO", "ROST", "YUM", "DHI", "LEN",
        "LULU", "EBAY", "TSCO", "GRMN", "DRI", "ULTA", "PHM", "NVR", "BBY", "EXPE",
        "DPZ", "POOL", "KMX", "RCL", "CCL", "NCLH", "LVS", "MGM", "WYNN", "TPR",
        "RL", "HAS", "APTV", "BWA", "GPC", "LKQ", "DECK", "ABNB", "DASH", "UBER",
    ],
    "Consumer Staples": [
        "WMT", "PG", "COST", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "TGT",
        "KMB", "GIS", "STZ", "SYY", "KHC", "KDP", "HSY", "KR", "MNST", "ADM",
        "DG", "DLTR", "EL", "CHD", "CLX", "MKC", "TSN", "HRL", "CAG", "SJM",
        "CPB", "K", "TAP", "BG", "WBA", "KVUE",
    ],
    "Health Care": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN",
        "ISRG", "SYK", "BSX", "MDT", "GILD", "VRTX", "ELV", "REGN", "CI", "BDX",
        "ZTS", "HCA", "CVS", "MCK", "BMY", "EW", "A", "IDXX", "IQV", "HUM",
        "DXCM", "CNC", "RMD", "MTD", "BIIB", "WST", "ZBH", "COR", "CAH", "STE",
        "BAX", "LH", "DGX", "HOLX", "ALGN", "MRNA", "PODD", "COO", "WAT", "TECH",
        "INCY", "MOH", "RVTY", "CRL", "VTRS", "HSIC", "DVA", "UHS",
    ],
    "Financials": [
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "SPGI",
        "BLK", "C", "SCHW", "MMC", "PGR", "CB", "ICE", "CME", "AON", "USB",
        "PNC", "MCO", "AJG", "TFC", "COF", "AIG", "MET", "TRV", "AFL", "BK",
        "ALL", "PRU", "AMP", "MSCI", "FIS", "ACGL", "TROW", "FITB", "STT", "WTW",
        "HIG", "MTB", "NDAQ", "DFS", "HBAN", "RF", "CINF", "NTRS", "CFG", "SYF",
        "KEY", "BRO", "RJF", "FDS", "PFG", "WRB", "L", "EG", "GL", "AIZ",
        "BEN", "IVZ", "MKTX", "CBOE", "PYPL", "FI", "GPN", "JKHY",
    ],
    "Industrials": [
        "GE", "CAT", "RTX", "UNP", "HON", "ETN", "BA", "DE", "LMT", "UPS",
        "ADP", "GD", "WM", "TT", "ITW", "CTAS", "PH", "NOC", "EMR", "CSX",
        "FDX", "PCAR", "NSC", "CARR", "JCI", "GWW", "CPRT", "URI", "PAYX", "CMI",
        "FAST", "AME", "LHX", "OTIS", "RSG", "ODFL", "PWR", "VRSK", "DAL", "IR",
        "XYL", "EFX", "DOV", "ROK", "HWM", "WAB", "FTV", "LUV", "UAL", "BR",
        "GNRC", "SNA", "EXPD", "J", "MAS", "CHRW", "TXT", "NDSN", "SWK", "PNR",
        "ALLE", "AOS", "HUBB", "LDOS", "AXON", "BLDR", "DAY", "PAYC", "HII", "IEX",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "WMB", "OXY",
        "KMI", "HES", "FANG", "HAL", "BKR", "DVN", "TRGP", "OKE", "CTRA", "EQT",
        "APA", "MRO",
    ],
    "Materials": [
        "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "CTVA",
        "PPG", "VMC", "MLM", "IFF", "LYB", "STLD", "CF", "ALB", "BALL", "AVY",
        "PKG", "CE", "EMN", "MOS", "AMCR", "IP", "FMC",
    ],
    "Utilities": [
        "NEE", "SO", "DUK", "CEG", "SRE", "AEP", "D", "PCG", "EXC", "XEL",
        "PEG", "ED", "EIX", "WEC", "ETR", "DTE", "AWK", "PPL", "ES", "FE",
        "AEE", "CMS", "CNP", "ATO", "NI", "LNT", "EVRG", "NRG", "PNW", "AES",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "WELL", "SPG", "PSA", "O", "DLR", "CCI", "CBRE",
        "VICI", "EXR", "AVB", "CSGP", "VTR", "IRM", "EQR", "SBAC", "WY", "INVH",
        "MAA", "ESS", "ARE", "KIM", "DOC", "UDR", "HST", "REG", "CPT", "FRT",
    ],
}


def get_universe(sectors: list[str] | None = None, max_tickers: int | None = None) -> list[str]:
    """Return the ticker list for the requested sectors (all sectors if None)."""
    if sectors:
        unknown = [s for s in sectors if s not in UNIVERSE]
        if unknown:
            raise ValueError(f"Unknown sectors {unknown}. Known: {list(UNIVERSE)}")
        groups = [UNIVERSE[s] for s in sectors]
    else:
        groups = list(UNIVERSE.values())

    seen: set[str] = set()
    tickers: list[str] = []
    for group in groups:
        for t in group:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    if max_tickers:
        tickers = tickers[:max_tickers]
    return tickers


def sector_of(ticker: str) -> str:
    for sector, names in UNIVERSE.items():
        if ticker in names:
            return sector
    return "Unknown"
