"""Read-only cryptocurrency price and wallet lookups."""

from __future__ import annotations

import json

import requests

from ..util import clip

# Top 15 coins by market cap.
MARKET_PARAMS: dict[str, str | int] = {
    "vs_currency": "usd",
    "order": "market_cap_desc",
    "per_page": 15,
    "page": 1,
}


def t_crypto(
    action: str,
    symbol: str | None = None,
    address: str | None = None,
    chain: str = "ethereum",
) -> str:
    """Read-only crypto lookups: ``price``, ``market``, ``balance`` or ``gas``.

    Uses public APIs (CoinGecko, blockchain.info, Ethplorer, Blocknative).
    Holds no keys and cannot move funds.
    """
    a = action.lower()
    try:
        if a == "price":
            ids = (symbol or "bitcoin").lower().replace(" ", "")
            r = requests.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": ids,
                    "vs_currencies": "usd,gbp",
                    "include_24hr_change": "true",
                },
                timeout=30,
            )
            return clip(json.dumps(r.json(), indent=1))
        if a == "market":
            r = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params=MARKET_PARAMS,
                timeout=30,
            )
            return clip(
                "\n".join(
                    f"{c['market_cap_rank']:>3} {c['symbol'].upper():<6} "
                    f"${c['current_price']:<12,.2f} "
                    f"{c.get('price_change_percentage_24h') or 0:+.1f}%"
                    for c in r.json()
                )
            )
        if a == "balance":
            if not address:
                return "give an address"
            if chain.lower() in ("bitcoin", "btc"):
                r = requests.get(
                    f"https://blockchain.info/rawaddr/{address}?limit=0", timeout=30
                ).json()
                return (
                    f"{address}\n"
                    f"balance: {r.get('final_balance', 0) / 1e8:.8f} BTC\n"
                    f"received: {r.get('total_received', 0) / 1e8:.8f}\n"
                    f"txs: {r.get('n_tx')}"
                )
            r = requests.get(
                f"https://api.ethplorer.io/getAddressInfo/{address}?apiKey=freekey",
                timeout=30,
            ).json()
            eth = (r.get("ETH") or {}).get("balance", 0)
            toks = r.get("tokens") or []
            lines = [f"{address}", f"ETH: {eth}"]
            for t in toks[:15]:
                ti = t.get("tokenInfo", {})
                dec = int(ti.get("decimals") or 18)
                lines.append(f"{ti.get('symbol', '?')}: {t.get('balance', 0) / (10**dec):,.4f}")
            return clip("\n".join(lines))
        if a == "gas":
            r = requests.get("https://api.blocknative.com/gasprices/blockprices", timeout=30).json()
            bp = (r.get("blockPrices") or [{}])[0]
            return clip(json.dumps(bp.get("estimatedPrices", [])[:3], indent=1))
    except Exception as e:  # noqa: BLE001 - a tool must report failures, not crash the agent
        return f"lookup failed: {type(e).__name__}: {e}"
    return (
        "actions: price, market, balance, gas. Read-only by design "
        "— this tool holds no keys and cannot move funds."
    )
