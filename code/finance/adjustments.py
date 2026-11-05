"""Apply validated message adjustments to projected recurring cash flows.

Input is the list of RecurringSeries from finance.recurrence plus typed
CashFlowAdjustment items from extraction.message_interpreter. Everything here
is deterministic arithmetic over dates and amounts; the LLM only supplied the
typed facts. Adjustments are applied oldest message first, so a newer message
from the same sender wins, as the conflict rules require.
"""

import calendar
from dataclasses import dataclass, field
from datetime import date as date_cls
from decimal import ROUND_CEILING, Decimal
from typing import List, Optional, Sequence, Tuple

from finance.recurrence import RecurringSeries

_CENT = Decimal("0.01")
DUPLICATE_WINDOW_DAYS = 5


@dataclass
class ProjectedFlow:
    """Mutable projected occurrences for one series (or a message-created flow)."""

    direction: str
    category: Optional[str]
    items: List[Tuple[date_cls, Decimal]]
    series: Optional[RecurringSeries] = None
    sources: List[str] = field(default_factory=list)

    @property
    def is_salary(self) -> bool:
        return self.direction == "credit" and self.category == "salary"

    def signed(self) -> List[Tuple[date_cls, Decimal]]:
        sign = Decimal(-1) if self.direction == "debit" else Decimal(1)
        return [(when, sign * amount) for when, amount in self.items]


def _monthly(first: date_cls, end: date_cls) -> List[date_cls]:
    dates, index = [], 0
    while True:
        month = first.month - 1 + index
        year, month = first.year + month // 12, month % 12 + 1
        upcoming = date_cls(year, month, min(first.day, calendar.monthrange(year, month)[1]))
        if upcoming > end:
            return dates
        dates.append(upcoming)
        index += 1


def _primary_salary(flows) -> Optional[ProjectedFlow]:
    salaries = [flow for flow in flows if flow.is_salary and flow.items]
    return max(salaries, key=lambda flow: flow.items[0][1]) if salaries else None


def _from(adjustment, start: date_cls) -> date_cls:
    return adjustment.effective_date or start


def _salary_amount(flows, adjustment, start, end):
    primary = _primary_salary(flows)
    effective = _from(adjustment, start)
    if primary is None:
        if adjustment.effective_date and start <= adjustment.effective_date <= end:
            flows.append(ProjectedFlow("credit", "salary",
                                       [(d, adjustment.amount) for d in _monthly(adjustment.effective_date, end)],
                                       sources=[adjustment.source_id]))
        return
    primary.items = [(when, adjustment.amount if when >= effective else amount) for when, amount in primary.items]
    primary.sources.append(adjustment.source_id)


def _salary_date(flows, adjustment, start, end):
    primary = _primary_salary(flows)
    target = adjustment.effective_date
    if primary is None or target is None or target < start:
        return
    upcoming = [item for item in primary.items if item[0] >= start]
    if not upcoming:
        return
    delta = target - upcoming[0][0]
    shifted = [(when + delta, amount) for when, amount in primary.items if when >= start]
    primary.items = [item for item in shifted if item[0] <= end]
    primary.sources.append(adjustment.source_id)


def _income_ended(flows, adjustment, start, _end):
    effective = _from(adjustment, start)
    salaries = [flow for flow in flows if flow.is_salary and flow.items]
    if adjustment.scope == "one" and len(salaries) > 1:
        salaries = [min(salaries, key=lambda flow: flow.items[0][1])]
    for flow in salaries:
        flow.items = [item for item in flow.items if item[0] < effective]
        flow.sources.append(adjustment.source_id)


def _confirmed_income(flows, adjustment, start, end, existing_credit_dates):
    when = adjustment.effective_date
    if when is None or when < start or when > end:
        return
    for flow in flows:
        if not flow.is_salary:
            continue
        for index, (existing, _amount) in enumerate(flow.items):
            if abs((existing - when).days) <= DUPLICATE_WINDOW_DAYS:
                flow.items[index] = (when, adjustment.amount)
                flow.sources.append(adjustment.source_id)
                return
    if any(abs((existing - when).days) <= DUPLICATE_WINDOW_DAYS for existing in existing_credit_dates):
        return  # already a CSV scheduled row for this credit
    dates = _monthly(when, end) if adjustment.recurring else [when]
    flows.append(ProjectedFlow("credit", "salary" if adjustment.recurring else "confirmed_income",
                               [(d, adjustment.amount) for d in dates], sources=[adjustment.source_id]))


def _expense_amount(flows, adjustment, start, _end):
    effective = _from(adjustment, start)
    for flow in flows:
        if flow.direction != "debit" or flow.category != adjustment.category:
            continue
        updated = []
        for when, amount in flow.items:
            new_amount = amount
            if when >= effective:
                new_amount = adjustment.amount if adjustment.amount is not None else (
                    amount * (1 + adjustment.percent_change / 100)).quantize(_CENT, rounding=ROUND_CEILING)
            updated.append((when, new_amount))
        flow.items = updated
        flow.sources.append(adjustment.source_id)


def _new_recurring_expense(flows, adjustment, start, end):
    first = adjustment.effective_date
    if first is None or first > end:
        return
    dates = [d for d in _monthly(first, end) if d >= start]
    flows.append(ProjectedFlow("debit", adjustment.category, [(d, adjustment.amount) for d in dates],
                               sources=[adjustment.source_id]))


def apply_adjustments(
    series_list: Sequence[RecurringSeries], adjustments, start: date_cls, end: date_cls,
    existing_credit_dates: Sequence[date_cls] = (),
) -> List[ProjectedFlow]:
    """Return projected flows after applying every adjustment, oldest message first."""
    flows = [ProjectedFlow(series.direction, series.category,
                           [(when, series.amount) for when in series.occurrences], series=series)
             for series in series_list]
    for adjustment in sorted(adjustments, key=lambda item: (item.sent_on, item.source_id)):
        if adjustment.kind == "salary_amount":
            _salary_amount(flows, adjustment, start, end)
        elif adjustment.kind == "salary_date":
            _salary_date(flows, adjustment, start, end)
        elif adjustment.kind == "income_ended":
            _income_ended(flows, adjustment, start, end)
        elif adjustment.kind == "confirmed_income":
            _confirmed_income(flows, adjustment, start, end, existing_credit_dates)
        elif adjustment.kind == "expense_amount":
            _expense_amount(flows, adjustment, start, end)
        elif adjustment.kind == "new_recurring_expense":
            _new_recurring_expense(flows, adjustment, start, end)
    for flow in flows:
        flow.items = [(when, amount) for when, amount in flow.items if start < when <= end]
    return flows


def describe_sources(flows) -> Tuple[str, ...]:
    return tuple(sorted({source for flow in flows for source in flow.sources}))

