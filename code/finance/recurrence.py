"""Detect recurring income/expenses in settled history and project them forward.

The challenge asks for a 90-day forecast built from "recurring income and
expenses", so a bill that has been paid every month is reserved again next
month even though no future CSV row exists for it. Recurrence is detected only
when the history supports it:

* Debits (expenses, subscriptions, debt payments) are grouped by category, so
  a weekly grocery habit made of differently described shop visits still
  forms one series.
* Salary credits are grouped by pay day-of-month, so two household incomes
  on different days stay separate. A scheduled ("confirmed") salary row joins
  its series as the latest occurrence and sets the projected amount.
* A series needs at least two occurrences, a regular spacing (every N days
  with N between 3 and 10, or monthly on the same day of month), and must not have missed its most recent expected
  occurrence (a stopped income or cancelled subscription is not projected).
* Salary must repeat a stable amount (gig payouts that swing are not
  projected), and a payroll described as "final" ends its series.
* Projected expense amounts are the mean of recent occurrences, rounded up to
  the cent (conservative); projected salary uses the latest confirmed amount.
* A projection is skipped when a CSV pending/scheduled row of the same
  category already covers that occurrence, so nothing is counted twice.

Refunds, investment flows, bonuses and other non-salary credits are never
projected: the rules forbid counting income that has not been confirmed.
"""

import calendar
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import timedelta
from decimal import ROUND_CEILING, Decimal
from statistics import median
from typing import Dict, List, Optional, Tuple

PROJECTED_DEBIT_TYPES = ("expense", "subscription", "debt_payment")
PROJECTED_CREDIT_CATEGORIES = ("salary",)
FUTURE_STATUSES = ("pending", "scheduled")
STALE_TOLERANCE_DAYS = 3
MONTHLY = 30  # marker: project on the same day of each following month
MIN_STEP_DAYS = 3
MONTHLY_GAP_DAYS = (27, 32)
MIN_OCCURRENCES = 2
GAP_TOLERANCE_DAYS = 2
MAX_STEP_DAYS = 10  # sparser 2-3 weekly habits proved too noisy to reserve
RECENT_OCCURRENCES = 3
STABLE_INCOME_TOLERANCE = Decimal("0.05")
ENDED_INCOME_WORDS = ("final", "last pay", "severance")
_CENT = Decimal("0.01")


@dataclass(frozen=True)
class RecurringSeries:
    """One detected recurring cash flow and its projected future occurrences."""

    key: Tuple
    direction: str
    category: Optional[str]
    step_days: int
    last_event_id: str
    flexibility: str
    minimum_allowed_amount: Optional[Decimal]
    amount: Decimal
    occurrences: Tuple[date_cls, ...]

    @property
    def signed_amount(self) -> Decimal:
        return -self.amount if self.direction == "debit" else self.amount


def _event_date(record) -> Optional[date_cls]:
    return record.settlement_date or record.event_date


def _add_months(anchor: date_cls, months: int, day: int) -> date_cls:
    index = anchor.month - 1 + months
    year, month = anchor.year + index // 12, index % 12 + 1
    return date_cls(year, month, min(day, calendar.monthrange(year, month)[1]))


def _cadence(dates: List[date_cls]) -> Optional[int]:
    gaps = [(later - earlier).days for earlier, later in zip(dates, dates[1:])]
    if not gaps:
        return None
    middle = median(gaps)
    if MONTHLY_GAP_DAYS[0] <= middle <= MONTHLY_GAP_DAYS[1]:
        return MONTHLY
    if MIN_STEP_DAYS <= middle <= MAX_STEP_DAYS and all(abs(gap - middle) <= GAP_TOLERANCE_DAYS for gap in gaps[-4:]):
        return round(middle)
    return None


def _series_key(record, when: date_cls):
    if record.direction == "credit":
        # Nominal pay day (event_date), so a late-settling salary stays in its series.
        return ("credit", record.category, (record.event_date or when).day)
    return ("debit", record.category)


def _is_stable_income(amounts: List[Decimal]) -> bool:
    """Salary repeats an amount; gig payouts swing from payout to payout."""
    pairs = list(zip(amounts, amounts[1:]))
    steady = sum(1 for prev, cur in pairs if prev and abs(cur - prev) <= abs(prev) * STABLE_INCOME_TOLERANCE)
    return steady * 2 >= len(pairs)


def _income_has_ended(record) -> bool:
    description = (record.description or "").lower()
    return any(word in description for word in ENDED_INCOME_WORDS)


def _is_projectable(record) -> bool:
    if record.direction == "debit":
        return getattr(record, "event_type", None) in PROJECTED_DEBIT_TYPES
    if record.direction == "credit":
        return getattr(record, "event_type", None) == "income" and record.category in PROJECTED_CREDIT_CATEGORIES
    return False


def detect_recurring_series(  # noqa: PLR0912 -- one detection pass, rules documented above
    events, start: date_cls, end: date_cls, excluded_ids=frozenset(),
) -> List[RecurringSeries]:
    """Return every recurring series with its projected dates in (start, end]."""
    history: Dict[Tuple, list] = defaultdict(list)
    future_rows: Dict[Tuple, List[date_cls]] = defaultdict(list)
    for converted in events:
        record = converted.event
        when = _event_date(record)
        if (record.event_id in excluded_ids or when is None
                or getattr(converted, "home_amount", None) is None or not _is_projectable(record)):
            continue
        if record.status in FUTURE_STATUSES and when >= start:
            future_rows[(record.direction, record.category)].append(when)
            if record.direction == "credit" and record.status == "scheduled":
                history[_series_key(record, when)].append((when, converted))
        elif record.status == "settled" and when <= start:
            history[_series_key(record, when)].append((when, converted))

    result = []
    for key, rows in history.items():
        rows.sort(key=lambda row: (row[0], row[1].event.event_id))
        dates = [when for when, _ in rows]
        step = _cadence(dates)
        if step is None or len(rows) < MIN_OCCURRENCES:
            continue
        last_date, last = rows[-1]
        if (start - last_date).days > step + STALE_TOLERANCE_DAYS:  # missed its latest occurrence
            continue
        record = last.event
        if record.direction == "credit":
            confirmed = record.status == "scheduled"
            if _income_has_ended(record) or not (
                    confirmed or _is_stable_income([c.home_amount for _, c in rows])):
                continue
            amount = last.home_amount
        else:
            recent = [converted.home_amount for _, converted in rows[-RECENT_OCCURRENCES:]]
            amount = (sum(recent) / len(recent)).quantize(_CENT, rounding=ROUND_CEILING)
        covered = future_rows.get((record.direction, record.category), ())
        occurrences = []
        index = 1
        while True:
            if step == MONTHLY:
                upcoming = _add_months(last_date, index, last_date.day)
            else:
                upcoming = last_date + timedelta(days=step * index)
            index += 1
            if upcoming > end:
                break
            if upcoming <= start:
                continue
            if record.direction == "debit" and any(abs((row - upcoming).days) <= step // 2 for row in covered):
                continue
            occurrences.append(upcoming)
        if occurrences:
            result.append(RecurringSeries(
                key=key, direction=record.direction, category=record.category, step_days=step,
                last_event_id=record.event_id, flexibility=getattr(record, "flexibility", None) or "fixed",
                minimum_allowed_amount=getattr(last, "home_minimum_allowed_amount", None), amount=amount,
                occurrences=tuple(occurrences),
            ))
    result.sort(key=lambda series: series.key)
    return result
