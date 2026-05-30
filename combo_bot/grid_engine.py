from __future__ import annotations

import math
from dataclasses import dataclass

from typing import TYPE_CHECKING

from combo_bot.types import (
    EMAState,
    ExchangeParams,
    Order,
    OrderSource,
    Position,
    Side,
    TradingMode,
    TrailingState,
    VolatilityState,
)

if TYPE_CHECKING:
    from combo_bot.types import AccountState, Candle, TrendSignal


@dataclass
class GridConfig:
    entry_initial_ema_dist: float = 0.008
    entry_initial_qty_pct: float = 0.012
    entry_grid_spacing_pct: float = 0.025
    entry_grid_spacing_volatility_weight: float = 15.0
    entry_grid_spacing_we_weight: float = 1.0
    entry_grid_double_down_factor: float = 1.3
    # In AGGRESSIVE mode (favorable regime), the grid stacks faster: larger
    # DDF and compressed spacing pull entries in tighter.
    aggressive_double_down_factor: float = 1.6
    aggressive_spacing_compression: float = 0.75
    close_grid_markup_start: float = 0.005
    close_grid_markup_end: float = 0.015
    close_grid_qty_pct: float = 0.5
    close_trailing_threshold_pct: float = 0.01
    close_trailing_retracement_pct: float = 0.004
    # Stage 8 trailing re-entry (passivbot ``calc_trailing_entry_long``).
    # Defaults of 0 disable the path entirely — opt-in safety.
    #
    # Stage 1: price has moved at least ``entry_trailing_threshold_pct``
    # against the avg entry (i.e. the position is at least that far
    # underwater) — we don't trail until we're committed enough that a
    # cheaper re-entry is meaningful.
    # Stage 2: price has recovered at least ``entry_trailing_retracement_pct``
    # from the local extreme — we want to DCA after a bottom is in, not
    # while still bleeding.
    # Both conditions must fire on the same tick.
    entry_trailing_threshold_pct: float = 0.0
    entry_trailing_retracement_pct: float = 0.0
    entry_trailing_double_down_factor: float = 1.3
    wallet_exposure_limit: float = 1.0
    n_positions: int = 7
    total_wallet_exposure_limit: float = 1.5
    max_grid_levels: int = 10
    ema_span_0: float = 385.0
    ema_span_1: float = 620.0
    entry_volatility_ema_span_hours: float = 1000.0


@dataclass
class ForagerWeights:
    volume: float = 0.20
    volatility: float = 0.55
    ema_readiness: float = 0.05
    # |trend.direction| × strength contribution to the score. Adds a
    # "momentum" axis so the Forager prefers symbols with a clear
    # directional regime over choppy / mean-reverting ones. Zero keeps
    # the legacy volume/volatility/ema behaviour.
    trend_conviction: float = 0.20


def quantize_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


def quantize_price(price: float, step: float, *, round_up: bool = False) -> float:
    if step <= 0:
        return price
    if round_up:
        return math.ceil(price / step) * step
    return math.floor(price / step) * step


def calc_wallet_exposure(
    balance: float,
    position_size: float,
    entry_price: float,
    c_mult: float,
) -> float:
    if balance <= 0:
        return 0.0
    return abs(position_size) * entry_price * c_mult / balance


class GridEngine:
    def __init__(self, config: GridConfig) -> None:
        self._cfg = config

    def compute_orders(
        self,
        symbol: str,
        side: Side,
        position: Position,
        ema_state: EMAState,
        volatility: VolatilityState,
        balance: float,
        wallet_exposure: float,
        exchange_params: ExchangeParams,
        mode: TradingMode,
        mark_price: float = 0.0,
        close_markup_multiplier: float = 1.0,
    ) -> list[Order]:
        orders: list[Order] = []

        if mode == TradingMode.PANIC:
            return self._panic_close(symbol, side, position, mark_price)

        if position.is_open:
            close_orders = self._compute_close_orders(
                symbol,
                side,
                position,
                exchange_params,
                mode,
                close_markup_multiplier,
            )
            orders.extend(close_orders)

        # AGGRESSIVE and NORMAL both produce new entries; AGGRESSIVE stacks
        # faster via a larger DDF and compressed spacing.
        if mode in (TradingMode.NORMAL, TradingMode.AGGRESSIVE):
            entry_orders = self._compute_entry_orders(
                symbol,
                side,
                position,
                ema_state,
                volatility,
                balance,
                wallet_exposure,
                exchange_params,
                mode,
            )
            orders.extend(entry_orders)

        return orders

    def compute_trailing_entry(
        self,
        symbol: str,
        side: Side,
        position: Position,
        trailing: TrailingState,
        balance: float,
        wallet_exposure: float,
        exchange_params: ExchangeParams,
        mark_price: float,
        mode: TradingMode = TradingMode.NORMAL,
    ) -> Order | None:
        """passivbot-style two-stage trailing re-entry.

        Returns a single re-entry :class:`Order` when both stages fire
        on the same tick, or ``None`` otherwise. See
        :class:`TrailingState` for the bundle semantics.

        The method is a no-op (returns ``None``) when:

        * trailing is disabled (either threshold is 0);
        * the position isn't open (passivbot trails *re*-entries, not
          initial entries);
        * the trailing bundle hasn't been seeded yet;
        * the side is in a non-entering mode (TP_ONLY / GRACEFUL_STOP /
          PANIC) — trailing is a new-exposure path, so risk-off modes
          gate it;
        * the wallet-exposure ceiling would be breached.
        """
        cfg = self._cfg
        threshold = cfg.entry_trailing_threshold_pct
        retracement = cfg.entry_trailing_retracement_pct
        if threshold <= 0 or retracement <= 0:
            return None
        if not position.is_open or not trailing.initialized:
            return None
        if mode not in (TradingMode.NORMAL, TradingMode.AGGRESSIVE):
            return None
        if balance <= 0 or mark_price <= 0:
            return None
        if wallet_exposure >= cfg.wallet_exposure_limit * 0.999:
            return None

        if side == Side.LONG:
            below_threshold = trailing.extreme < position.entry_price * (
                1.0 - threshold
            )
            bounced = trailing.recovery > trailing.extreme * (1.0 + retracement)
            if not (below_threshold and bounced):
                return None
            # Limit price sits between the extreme and the entry — a level
            # we'll only fill if price retraces back into it.
            raw_price = position.entry_price * (1.0 - threshold + retracement)
            order_price = min(mark_price, raw_price)
        else:
            above_threshold = trailing.extreme > position.entry_price * (
                1.0 + threshold
            )
            retraced = trailing.recovery < trailing.extreme * (1.0 - retracement)
            if not (above_threshold and retraced):
                return None
            raw_price = position.entry_price * (1.0 + threshold - retracement)
            order_price = max(mark_price, raw_price)

        # Size the re-entry like a DDF-scaled DCA against the current
        # position. Use the trailing-specific DDF so users can tune it
        # independently from the grid ladder.
        ddf = cfg.entry_trailing_double_down_factor
        qty = abs(position.size) * ddf
        qty = max(qty, exchange_params.min_qty)
        qty = quantize_qty(qty, exchange_params.qty_step)
        if qty <= 0:
            return None

        # Crop qty so we don't blow past the wallet-exposure limit.
        cost = qty * order_price * exchange_params.c_mult
        room = (
            cfg.wallet_exposure_limit * balance
            - abs(position.size) * position.entry_price * exchange_params.c_mult
        )
        if room <= 0:
            return None
        if cost > room:
            qty = quantize_qty(
                room / max(order_price * exchange_params.c_mult, 1e-12),
                exchange_params.qty_step,
            )
            if qty < exchange_params.min_qty:
                return None
            cost = qty * order_price * exchange_params.c_mult

        if cost < exchange_params.min_cost:
            return None

        return Order(
            symbol=symbol,
            side=side,
            price=order_price,
            qty=qty,
            source=OrderSource.GRID,
        )

    def _panic_close(
        self,
        symbol: str,
        side: Side,
        position: Position,
        mark_price: float,
    ) -> list[Order]:
        if not position.is_open:
            return []
        # Use mark_price as the limit hint; the is_market flag makes the
        # fill simulator and live executor treat this as a taker market exit.
        # Fallback to entry_price only if caller forgot to pass mark_price.
        price = mark_price if mark_price > 0 else position.entry_price
        return [
            Order(
                symbol=symbol,
                side=side,
                price=price,
                qty=abs(position.size),
                source=OrderSource.RISK,
                reduce_only=True,
                is_market=True,
            ),
        ]

    def _compute_entry_orders(
        self,
        symbol: str,
        side: Side,
        position: Position,
        ema_state: EMAState,
        volatility: VolatilityState,
        balance: float,
        wallet_exposure: float,
        exchange_params: ExchangeParams,
        mode: TradingMode = TradingMode.NORMAL,
    ) -> list[Order]:
        cfg = self._cfg
        ep = exchange_params

        ema_band = ema_state.lower if side == Side.LONG else ema_state.upper
        if ema_band <= 0:
            return []

        first_price = self._first_entry_price(side, ema_band)
        first_qty = self._initial_entry_qty(balance, first_price, ep)

        if first_qty < ep.min_qty:
            return []

        # AGGRESSIVE mode: larger DDF (stack faster) + compressed spacing
        # (entries cluster closer together for quicker fills in顺势).
        if mode == TradingMode.AGGRESSIVE:
            ddf = cfg.aggressive_double_down_factor
            spacing_compression = cfg.aggressive_spacing_compression
        else:
            ddf = cfg.entry_grid_double_down_factor
            spacing_compression = 1.0

        orders: list[Order] = []
        cumulative_size = abs(position.size) if position.is_open else 0.0
        price = first_price
        qty = first_qty

        for _ in range(cfg.max_grid_levels):
            price = self._quantize_entry_price(price, ep.price_step, side)
            qty = quantize_qty(qty, ep.qty_step)

            if qty < ep.min_qty or price <= 0:
                break
            if qty * price * ep.c_mult < ep.min_cost:
                break

            projected_we = calc_wallet_exposure(
                balance,
                cumulative_size + qty,
                price,
                ep.c_mult,
            )
            if projected_we > cfg.wallet_exposure_limit:
                remaining_cost = (
                    cfg.wallet_exposure_limit * balance
                    - cumulative_size * price * ep.c_mult
                )
                capped_qty = quantize_qty(
                    remaining_cost / (price * ep.c_mult),
                    ep.qty_step,
                )
                if capped_qty >= ep.min_qty:
                    orders.append(self._make_entry(symbol, side, price, capped_qty))
                break

            orders.append(self._make_entry(symbol, side, price, qty))
            cumulative_size += qty

            # Spacing for the NEXT level should widen as cumulative
            # wallet exposure grows — passivbot's `we_multiplier` does the
            # same. Without this the loop emits a constant-spacing ladder
            # regardless of how much commitment has stacked.
            spacing = (
                self._grid_spacing(
                    volatility.value,
                    projected_we,
                )
                * spacing_compression
            )
            price = self._next_entry_price(price, spacing, side)
            qty *= ddf

        return orders

    def _compute_close_orders(
        self,
        symbol: str,
        side: Side,
        position: Position,
        exchange_params: ExchangeParams,
        mode: TradingMode,
        close_markup_multiplier: float = 1.0,
    ) -> list[Order]:
        cfg = self._cfg
        ep = exchange_params

        markup_start = cfg.close_grid_markup_start
        markup_end = cfg.close_grid_markup_end

        if mode == TradingMode.GRACEFUL_STOP:
            markup_start *= 0.5
            markup_end *= 0.5

        # Regime-driven compression on top of mode-driven adjustment.
        # close_markup_multiplier < 1.0 → close sooner (顺势 take-profit).
        if close_markup_multiplier != 1.0:
            markup_start *= close_markup_multiplier
            markup_end *= close_markup_multiplier

        remaining = abs(position.size)
        if remaining < ep.min_qty:
            return []

        n_levels = max(1, int(math.ceil(1.0 / cfg.close_grid_qty_pct)))
        markup_step = (markup_end - markup_start) / max(n_levels - 1, 1)

        orders: list[Order] = []
        for i in range(n_levels):
            markup = markup_start + markup_step * i
            close_price = self._close_price(position.entry_price, markup, side)
            close_price = self._quantize_close_price(close_price, ep.price_step, side)

            if i < n_levels - 1:
                qty = quantize_qty(
                    abs(position.size) * cfg.close_grid_qty_pct,
                    ep.qty_step,
                )
                qty = min(qty, remaining)
            else:
                qty = quantize_qty(remaining, ep.qty_step)

            if qty < ep.min_qty:
                continue

            remaining -= qty
            orders.append(
                Order(
                    symbol=symbol,
                    side=side,
                    price=close_price,
                    qty=qty,
                    source=OrderSource.GRID,
                    reduce_only=True,
                ),
            )

            if remaining < ep.min_qty:
                break

        return orders

    def _first_entry_price(self, side: Side, ema_band: float) -> float:
        dist = self._cfg.entry_initial_ema_dist
        if side == Side.LONG:
            return ema_band * (1.0 - dist)
        return ema_band * (1.0 + dist)

    def _initial_entry_qty(
        self,
        balance: float,
        entry_price: float,
        ep: ExchangeParams,
    ) -> float:
        cfg = self._cfg
        notional = balance * cfg.wallet_exposure_limit * cfg.entry_initial_qty_pct
        raw_qty = notional / (entry_price * ep.c_mult)
        return max(quantize_qty(raw_qty, ep.qty_step), ep.min_qty)

    def _grid_spacing(self, volatility: float, wallet_exposure: float) -> float:
        cfg = self._cfg
        vol_component = volatility * cfg.entry_grid_spacing_volatility_weight
        we_component = wallet_exposure * cfg.entry_grid_spacing_we_weight
        return cfg.entry_grid_spacing_pct * (1.0 + vol_component + we_component)

    def _next_entry_price(
        self,
        price: float,
        spacing: float,
        side: Side,
    ) -> float:
        if side == Side.LONG:
            return price * (1.0 - spacing)
        return price * (1.0 + spacing)

    def _close_price(
        self,
        entry_price: float,
        markup: float,
        side: Side,
    ) -> float:
        if side == Side.LONG:
            return entry_price * (1.0 + markup)
        return entry_price * (1.0 - markup)

    @staticmethod
    def _quantize_entry_price(
        price: float,
        step: float,
        side: Side,
    ) -> float:
        # Long entries: round down to get a better fill price.
        # Short entries: round up.
        return quantize_price(price, step, round_up=(side == Side.SHORT))

    @staticmethod
    def _quantize_close_price(
        price: float,
        step: float,
        side: Side,
    ) -> float:
        # Long closes (sells): round up for better fill.
        # Short closes (buys): round down.
        return quantize_price(price, step, round_up=(side == Side.LONG))

    @staticmethod
    def _make_entry(
        symbol: str,
        side: Side,
        price: float,
        qty: float,
    ) -> Order:
        return Order(
            symbol=symbol,
            side=side,
            price=price,
            qty=qty,
            source=OrderSource.GRID,
            reduce_only=False,
        )


class ForagerScorer:
    @staticmethod
    def score_symbol(
        volume_score: float,
        volatility_score: float,
        ema_readiness_score: float,
        weights: ForagerWeights,
        trend_conviction: float = 0.0,
    ) -> float:
        """Linear weighted score over [0, 1] inputs.

        ``trend_conviction`` should be ``|direction| * strength`` from
        a TrendSignal — high in clear regimes, low in choppy / NEUTRAL
        ones. The high-risk profile uses this to bias selection toward
        symbols with a real directional opinion.
        """
        return (
            weights.volume * volume_score
            + weights.volatility * volatility_score
            + weights.ema_readiness * ema_readiness_score
            + weights.trend_conviction * trend_conviction
        )

    @staticmethod
    def select_symbols(
        candidates: dict[
            str, tuple[float, float, float] | tuple[float, float, float, float]
        ],
        n_positions: int,
        weights: ForagerWeights,
    ) -> list[str]:
        """Pick the top-N symbols by weighted Forager score.

        Round-22: the Python backtest and live paths now call this
        directly to mirror Passivbot's ``multi_symbol.rs`` selection
        before any per-symbol order generation; symbols outside the
        active set only emit reduce-only orders so existing positions
        wind down rather than getting stranded.

        Each candidate tuple is ``(volume, volatility, ema_readiness)``
        or ``(volume, volatility, ema_readiness, trend_conviction)`` —
        the 4-tuple form lets callers feed a per-symbol ``|direction|
        * strength`` value derived from ``TrendSignal``. The 3-tuple
        form is supported for callers that don't compute a trend
        signal yet (defaults to 0).
        """
        scored: list[tuple[str, float]] = []
        for symbol, scores in candidates.items():
            if len(scores) == 4:
                vol_score, volatility_score, ema_score, trend_conv = scores
            else:
                vol_score, volatility_score, ema_score = scores  # type: ignore[misc]
                trend_conv = 0.0
            total = ForagerScorer.score_symbol(
                vol_score,
                volatility_score,
                ema_score,
                weights,
                trend_conviction=trend_conv,
            )
            scored.append((symbol, total))

        # Sort by descending score, then by symbol name as a deterministic
        # tiebreaker so two symbols with identical scores produce a
        # stable active set across ticks.
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [symbol for symbol, _ in scored[:n_positions]]


def build_forager_candidates(
    symbols: list[str],
    candles: dict[str, "Candle"],
    account: "AccountState",
    signals: dict[str, "TrendSignal"],
) -> dict[str, tuple[float, float, float, float]]:
    """Build normalized Forager input tuples for the given symbols.

    Each input axis is normalized into ``[0, 1]`` so the
    ``ForagerWeights`` linear combination is comparable across runs:

    * ``volume`` — current bar volume divided by the max volume in
      ``candles``.
    * ``volatility`` — ``ss.volatility.value`` divided by the max in
      the universe.
    * ``ema_readiness`` — how tight current price is to the EMA band
      midpoint (1.0 means dead-on the band, 0.0 means far away). Uses
      the *narrower* of the two EMA distances since either side can
      trigger a grid entry.
    * ``trend_conviction`` — ``|direction| × strength`` from the
      ``TrendSignal``, already in ``[0, 1]``.

    A symbol with no candle / no signal contributes zeros for those
    axes — it just won't rank, rather than crashing the selector.
    """
    max_volume = max((candles[s].volume for s in symbols if s in candles), default=0.0)
    max_volatility = max(
        (account.symbols[s].volatility.value for s in symbols if s in account.symbols),
        default=0.0,
    )
    out: dict[str, tuple[float, float, float, float]] = {}
    for s in symbols:
        c = candles.get(s)
        ss = account.symbols.get(s)
        sig = signals.get(s)
        if c is None or ss is None:
            continue
        volume_score = (c.volume / max_volume) if max_volume > 0 else 0.0
        volatility_score = (
            ss.volatility.value / max_volatility if max_volatility > 0 else 0.0
        )
        ema_ready = 0.0
        if ss.ema.initialized and c.close > 0:
            d_lower = abs(c.close - ss.ema.lower) / c.close
            d_upper = abs(c.close - ss.ema.upper) / c.close
            ema_ready = max(0.0, 1.0 - min(d_lower, d_upper))
        conviction = 0.0
        if sig is not None:
            conviction = min(1.0, abs(sig.direction) * sig.strength)
        out[s] = (
            min(1.0, volume_score),
            min(1.0, volatility_score),
            ema_ready,
            conviction,
        )
    return out
