import os
import tempfile
import time
from typing import Sequence

import numpy as np
import pandas as pd

# Try to import yfinance, but don't fail if it's not available.
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False

USE_SAMPLE_DATA_ENV = "QUOP_PORTFOLIO_USE_SAMPLE_DATA"
YFINANCE_CACHE_DIR_NAME = "quop_mpi_yfinance_cache"
DEFAULT_CANDIDATES_FILE = os.path.join(os.path.dirname(__file__), "asx_candidates.txt")

_YFINANCE_CACHE_CONFIGURED = False


def _env_var_is_true(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalise_ticker(symbol: str | None) -> str | None:
    if symbol is None:
        return None

    cleaned = symbol.strip().strip('"').upper()
    if not cleaned:
        return None
    if "." not in cleaned:
        cleaned = f"{cleaned}.AX"
    return cleaned


def _load_candidate_stocks() -> list[str]:
    candidate_file = DEFAULT_CANDIDATES_FILE

    if not os.path.exists(candidate_file):
        raise FileNotFoundError(
            f"Candidate stock file not found: {candidate_file}\n"
            "Please ensure asx_candidates.txt exists in the portfolio_rebalancing directory."
        )

    candidates = []
    with open(candidate_file, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            raw = stripped.split(",")[0]
            if raw.strip().strip('"').lower() in {"ticker", "asx code", "asx_code", "code"}:
                continue
            ticker = _normalise_ticker(raw)
            if ticker is not None:
                candidates.append(ticker)

    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise ValueError(f"No valid tickers found in candidate stock file: {candidate_file}")

    return candidates


def _random_subset(items: Sequence[str], n_items: int, seed: int) -> list[str]:
    n_selected = min(n_items, len(items))
    if n_selected <= 0:
        return []

    rng = np.random.RandomState(seed)
    selected_indices = rng.choice(len(items), n_selected, replace=False)
    return [items[i] for i in selected_indices]


def _choose_sample_stocks(
    available_stocks: Sequence[str], n_stocks: int, stocks: Sequence[str] | None, seed: int
) -> list[str]:
    n_selected = min(n_stocks, len(available_stocks))
    if n_selected <= 0:
        return []

    if stocks is None:
        return _random_subset(available_stocks, n_selected, seed)

    requested = []
    for symbol in stocks:
        normalised = _normalise_ticker(symbol)
        if normalised in available_stocks and normalised not in requested:
            requested.append(normalised)
        if len(requested) == n_selected:
            return requested

    remaining = [symbol for symbol in available_stocks if symbol not in requested]
    requested.extend(_random_subset(remaining, n_selected - len(requested), seed))
    return requested


def _configure_yfinance_cache() -> None:
    global _YFINANCE_CACHE_CONFIGURED

    if _YFINANCE_CACHE_CONFIGURED or not YFINANCE_AVAILABLE:
        return

    try:
        cache_dir = os.path.join(tempfile.gettempdir(), YFINANCE_CACHE_DIR_NAME)
        os.makedirs(cache_dir, exist_ok=True)
        yf.set_tz_cache_location(cache_dir)
    except Exception as exc:
        print(f"Warning: unable to configure yfinance cache directory: {exc}", flush=True)
    finally:
        _YFINANCE_CACHE_CONFIGURED = True


def _extract_close_prices(stock_data: pd.DataFrame, selected_stocks: Sequence[str]) -> pd.DataFrame:
    if stock_data.empty:
        raise ValueError("Downloaded data is empty")

    close_data = None
    if isinstance(stock_data.columns, pd.MultiIndex):
        first_level = stock_data.columns.get_level_values(0)
        for price_field in ("Close", "Adj Close"):
            if price_field in first_level:
                close_data = stock_data[price_field]
                break
    else:
        for price_field in ("Close", "Adj Close"):
            if price_field in stock_data.columns:
                close_data = stock_data[[price_field]].copy()
                if len(selected_stocks) == 1:
                    close_data.columns = [selected_stocks[0]]
                break

    if close_data is None:
        raise ValueError(
            f"Downloaded data does not contain close prices. Columns: {list(stock_data.columns)}"
        )

    if isinstance(close_data, pd.Series):
        column_name = selected_stocks[0] if selected_stocks else "Close"
        close_data = close_data.to_frame(name=column_name)

    if close_data.isna().all().all():
        raise ValueError("Downloaded close-price data is all NaN")

    return close_data


def _build_candidate_pool(
    n_stocks: int, stocks: Sequence[str] | None, seed: int
) -> tuple[list[str], int]:
    candidates = _load_candidate_stocks()
    if stocks is None:
        requested = _random_subset(candidates, n_stocks, seed)
    else:
        requested = [_normalise_ticker(symbol) for symbol in stocks]
        requested = [symbol for symbol in requested if symbol is not None]
        requested = list(dict.fromkeys(requested))
    remaining = [symbol for symbol in candidates if symbol not in requested]

    # Keep selection deterministic while still exploring a broad candidate pool.
    rng = np.random.RandomState(seed)
    rng.shuffle(remaining)

    candidate_pool = requested + remaining
    candidate_pool = list(dict.fromkeys(candidate_pool))
    return candidate_pool, len(requested)


def _select_valid_symbols(
    close_data: pd.DataFrame, symbols_in_priority_order: Sequence[str], n_stocks: int
) -> list[str]:
    valid_symbols = []
    for symbol in symbols_in_priority_order:
        if symbol not in close_data.columns:
            continue
        if close_data[symbol].isna().all():
            continue
        valid_symbols.append(symbol)
        if len(valid_symbols) == n_stocks:
            break
    return valid_symbols


def get_sample_stock_data(n_stocks, stocks=None, seed=0):
    """
    Load sample stock data from bundled CSV file.

    This is used as a fallback when yfinance is unavailable or fails.
    The sample data contains 2020 historical prices for 5 ASX stocks.

    :param n_stocks: Number of stocks to retrieve
    :param stocks: Custom list of stocks (will be mapped to available sample stocks)
    :param seed: Random seed for reproducibility
    :return: DataFrame of adjusted close prices
    """
    sample_file = os.path.join(os.path.dirname(__file__), "sample_stock_data.csv")

    if not os.path.exists(sample_file):
        raise FileNotFoundError(
            f"Sample stock data file not found: {sample_file}\n"
            "Please ensure sample_stock_data.csv exists in the portfolio_rebalancing directory."
        )

    df = pd.read_csv(sample_file, index_col=0, parse_dates=True)

    available_stocks = list(df.columns)
    selected_stocks = _choose_sample_stocks(available_stocks, n_stocks, stocks, seed)

    print(f"Using sample data for: {', '.join(selected_stocks)}.", flush=True)

    return df[selected_stocks]


def get_stock_data(
    n_stocks,
    start_date,
    end_date,
    stocks=None,
    seed=0,
    retries=3,
):
    """
    Fetches adjusted close prices for a subset of ASX stocks.

    First attempts to download from Yahoo Finance. If that fails, falls back
    to bundled sample data.

    :param n_stocks: Number of stocks to retrieve
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :param stocks: Optional prioritized list of stock symbols to try first.
        If symbols are invalid/unavailable, replacements are drawn from the
        candidate universe.
    :param seed: Random seed for reproducibility
    :param retries: Number of retry attempts in case of failures
    :return: DataFrame of adjusted close prices
    """

    candidate_pool, requested_count = _build_candidate_pool(n_stocks, stocks, seed)
    if len(candidate_pool) < n_stocks:
        raise ValueError(
            f"Requested {n_stocks} stocks but only {len(candidate_pool)} candidates are available."
        )

    if _env_var_is_true(USE_SAMPLE_DATA_ENV):
        print("Using sample data (forced by runtime option).", flush=True)
        return get_sample_stock_data(n_stocks, stocks, seed)

    if not YFINANCE_AVAILABLE:
        print("yfinance not available, using sample data.", flush=True)
        return get_sample_stock_data(n_stocks, stocks, seed)

    _configure_yfinance_cache()
    batch_growth = max(n_stocks * 2, 1)
    batch_size = min(len(candidate_pool), max(n_stocks, requested_count, batch_growth))

    for attempt in range(retries):
        try:
            batch = candidate_pool[:batch_size]
            print(
                f"Attempt {attempt + 1}: retrieving prices for {len(batch)} candidate stocks.",
                flush=True,
            )
            stock_data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                progress=True,
                auto_adjust=True,
                threads=False,
            )

            close_data = _extract_close_prices(stock_data, batch)
            selected_stocks = _select_valid_symbols(close_data, batch, n_stocks)
            if len(selected_stocks) < n_stocks:
                if batch_size < len(candidate_pool):
                    batch_size = min(len(candidate_pool), batch_size + batch_growth)
                raise ValueError(
                    f"Only found {len(selected_stocks)} valid stocks from {len(batch)} candidates."
                )

            print(f"Selected stocks: {', '.join(selected_stocks)}.", flush=True)
            return close_data[selected_stocks]

        except Exception as exc:
            print(f"Attempt {attempt + 1} failed: {exc}")
            time.sleep(2)

    # All retries failed, fall back to sample data
    print("Failed to retrieve live data, falling back to sample data.", flush=True)
    return get_sample_stock_data(n_stocks, stocks, seed)


if __name__ == "__main__":
    df = get_stock_data(n_stocks=5, start_date="2024-01-01", end_date="2024-03-01")
    print(df.head())
