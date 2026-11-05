"""90-day cash-flow forecast over reconciled challenge financial events.

The timeline combines (a) CSV pending/scheduled rows and confirmed reconciled
facts and (b) recurring income/expenses projected from settled history by
finance.recurrence. Settled events are assumed already reflected in
`current_available_balance` and are not replayed forward. Pending credits,
cancelled/failed/unrealized events and non-cash events never count (Sec.
"90-Day Safety Check" in problem_statement.md). A blank amount that no
confirmed fact resolves is excluded from the timeline rather than guessed.
"""

from dataclasses import dataclass
from datetime import date as date_cls
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from extraction.event_reconciliation import find_superseded_events, reconcile_event

from finance.adjustments import apply_adjustments
from finance.recurrence import detect_recurring_series

_EXCLUDED_STATUSES = ("cancelled", "failed", "unrealized", "settled")
STOP_FLEXIBILITY = ("stoppable", "reducible_or_stoppable")
REDUCE_FLEXIBILITY = ("reducible", "reducible_or_stoppable")


@dataclass(frozen=True)
class FlexibleCandidate:
    """One flexible expense (a CSV row or a recurring series) that may be stopped or reduced.

    `occurrences` lists every (date, amount) the change affects; for a recurring
    series that is every projected instance, referenced by its latest event_id.
    """

    event_id: str
    date: date_cls
    amount: Decimal
    category: Optional[str]
    action: str  # "stop" or "reduce"
    reduced_amount: Optional[Decimal] = None
    occurrences: Tuple[Tuple[date_cls, Decimal], ...] = ()

    @property
    def affected(self) -> Tuple[Tuple[date_cls, Decimal], ...]:
        return self.occurrences or ((self.date, self.amount),)

    @property
    def freed(self) -> Decimal:
        per_unit = Decimal(0) if self.action == "stop" else self.reduced_amount
        return sum((amount - per_unit for _, amount in self.affected), Decimal(0))


def _flexible_candidate(event_id, category, flexibility, minimum_allowed, profile, occurrences):  # noqa: PLR0913, PLR0917
    """Build a stop/reduce candidate when the profile permits changing this category."""
    if category in profile.expense_categories_to_protect:
        return None
    first_date, amount = occurrences[0]
    if flexibility in STOP_FLEXIBILITY and category in profile.expense_categories_user_is_willing_to_stop:
        return FlexibleCandidate(event_id, first_date, amount, category, "stop", occurrences=occurrences)
    if (flexibility in REDUCE_FLEXIBILITY and category in profile.expense_categories_user_is_willing_to_reduce
            and minimum_allowed is not None and minimum_allowed < amount):
        return FlexibleCandidate(event_id, first_date, amount, category, "reduce", minimum_allowed,
                                 occurrences=occurrences)
    return None


def _home_amount(converted_event, reconciled) -> Optional[Decimal]:
    if reconciled.amount is None:
        return None
    if converted_event.exchange_rate is None:
        return reconciled.amount
    return reconciled.amount * converted_event.exchange_rate.rate


def build_timeline(  # noqa: PLR0913, PLR0917 -- one forecast-construction boundary
    events, profile, start: date_cls, end: date_cls, facts_by_event: Optional[Dict] = None,
    project_recurring: bool = True, adjustments=(),
) -> Tuple[List[Tuple[date_cls, Decimal]], List[FlexibleCandidate]]:
    """Reconcile every event and return (timeline, flexible_candidates).

    `timeline` is a list of (date, signed home-currency delta) for events
    that must be reserved or counted within [start, end]. `facts_by_event`
    optionally maps event_id -> iterable of EventFact for AI-assisted
    reconciliation; omit it to reconcile against the CSV facts alone.
    `project_recurring` adds recurring income/expenses detected in history;
    `adjustments` (typed CashFlowAdjustment items interpreted from messages)
    then amend those projections, e.g. a salary change or an ended contract.
    """
    facts_by_event = facts_by_event or {}
    superseded = find_superseded_events(events)
    timeline = []
    flexible = []
    for converted in events:
        record = converted.event
        if record.event_id in superseded or record.direction == "non_cash":
            continue
        reconciled = reconcile_event(converted, facts_by_event.get(record.event_id, ()))
        if reconciled.status in _EXCLUDED_STATUSES:
            continue
        if reconciled.status == "pending" and record.direction == "credit":
            continue
        when = reconciled.effective_date
        amount = _home_amount(converted, reconciled)
        if when is None or amount is None or when < start or when > end:
            continue
        sign = Decimal(-1) if record.direction == "debit" else Decimal(1)
        timeline.append((when, sign * amount))
        if record.direction != "debit":
            continue
        candidate = _flexible_candidate(
            record.event_id, record.category, reconciled.flexibility,
            converted.home_minimum_allowed_amount, profile, ((when, amount),),
        )
        if candidate is not None:
            flexible.append(candidate)
    if project_recurring:
        series_list = detect_recurring_series(events, start, end, superseded)
        credit_dates = [when for when, delta in timeline if delta > 0]
        for flow in apply_adjustments(series_list, adjustments, start, end, credit_dates):
            timeline.extend(flow.signed())
            if flow.direction != "debit" or flow.series is None or not flow.items:
                continue
            series = flow.series
            candidate = _flexible_candidate(
                series.last_event_id, series.category, series.flexibility,
                series.minimum_allowed_amount, profile, tuple(flow.items),
            )
            if candidate is not None:
                flexible.append(candidate)
    timeline.sort(key=lambda item: item[0])
    flexible.sort(key=lambda item: item.freed, reverse=True)
    return timeline, flexible


def _aggregate_by_date(timeline) -> List[Tuple[date_cls, Decimal]]:
    """Net same-date deltas first so intraday ordering never creates a false dip."""
    totals: Dict[date_cls, Decimal] = {}
    for when, delta in timeline:
        totals[when] = totals.get(when, Decimal(0)) + delta
    return sorted(totals.items())


def lowest_balance(balance0: Decimal, timeline) -> Decimal:
    running = balance0
    lowest = running
    for _, delta in _aggregate_by_date(timeline):
        running += delta
        lowest = min(lowest, running)
    return lowest


def amount_safe_to_pay(balance0: Decimal, timeline, minimum_balance: Decimal, requested_amount: Decimal) -> Decimal:
    """Largest amount payable today without breaching minimum_balance, capped at the request."""
    headroom = lowest_balance(balance0, timeline) - minimum_balance
    return max(Decimal(0), min(requested_amount, headroom))


def _safe_paying_on(balance0, timeline, amount, minimum_balance, payment_date) -> bool:
    totals = dict(_aggregate_by_date(timeline))
    totals[payment_date] = totals.get(payment_date, Decimal(0)) - amount
    running = balance0
    for _when, delta in sorted(totals.items()):
        running += delta
        if running < minimum_balance:
            return False
    return True


def earliest_full_payment_date(  # noqa: PLR0913, PLR0917 -- one forecast-query boundary
    balance0: Decimal, timeline, requested_amount: Decimal, minimum_balance: Decimal,
    start: date_cls, end: date_cls,
) -> Optional[date_cls]:
    """First date a single full payment stays safe through the forecast end."""
    candidates = sorted({start, *(when for when, _ in timeline if start <= when <= end)})
    for candidate in candidates:
        if _safe_paying_on(balance0, timeline, requested_amount, minimum_balance, candidate):
            return candidate
    return None


def apply_candidate(timeline, candidate: FlexibleCandidate):
    """Return a new timeline with one flexible event (or recurring series) stopped or reduced."""
    pending = list(candidate.affected)
    adjusted = []
    for when, delta in timeline:
        match = next((item for item in pending if item[0] == when and delta == -item[1]), None)
        if match is None:
            adjusted.append((when, delta))
            continue
        pending.remove(match)
        if candidate.action == "reduce":
            adjusted.append((when, -candidate.reduced_amount))
    return adjusted
