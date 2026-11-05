"""Deterministic recommendation selection for the challenge output contract.

No LLM call happens here: every number comes from the 90-day forecast in
finance.challenge_forecast and the supplied payment options/preferences.
Ranking follows problem_statement.md "Choosing Between Safe Plans" exactly.
"""

from dataclasses import dataclass, replace
from datetime import timedelta
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import List, Optional, Tuple

from finance.challenge_forecast import (
    amount_safe_to_pay,
    apply_candidate,
    build_timeline,
    earliest_full_payment_date,
    lowest_balance,
)

_CENT = Decimal("0.01")
MAX_INSTALLMENT_DAY_SLACK = 31  # calendar days per month, upper bound so real months never reject


@dataclass(frozen=True)
class ChallengeDecision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: Tuple[Tuple, ...]
    earliest_date_for_full_payment: Optional[object]
    spending_changes: Tuple[str, ...]
    decision_explanation: str


@dataclass
class _Candidate:
    status: str
    method: str
    plan: List[Tuple]
    total: Decimal
    start: object
    n_payments: int
    completes_by_deadline: bool
    option_id: str = ""
    changes: Tuple[str, ...] = ()
    applied: Tuple = ()
    timeline: Optional[List[Tuple]] = None

    def rank_key(self):
        return (0 if self.completes_by_deadline else 1, 0 if not self.changes else 1,
                self.total, self.start, self.n_payments, self.option_id)


def _installment_candidate(  # noqa: PLR0913, PLR0917 -- one candidate-construction boundary
    option, request, timeline, balance0, min_balance, end, accepted,
):
    if "installments" not in accepted:
        return None
    dates = [option.first_payment_date + timedelta(days=(option.payment_frequency_days or 0) * i)
             for i in range(option.number_of_payments)]
    if dates[-1] > end:
        return None
    merged = sorted([*timeline, *((d, -option.payment_amount) for d in dates)])
    if lowest_balance(balance0, merged) < min_balance:
        return None
    plan = [(d, option.payment_amount) for d in dates]
    return _Candidate(
        "affordable_with_plan", "installments", plan, option.total_payable_amount,
        dates[0], option.number_of_payments, dates[-1] <= request.desired_completion_date,
        option_id=option.payment_option_id,
    )


def _eligible_installment_options(profile, options):
    if profile.max_installment_months is None:
        return ()
    limit_days = profile.max_installment_months * MAX_INSTALLMENT_DAY_SLACK
    result = []
    for option in options:
        if option.payment_method != "installments":
            continue
        span = (option.number_of_payments - 1) * (option.payment_frequency_days or 0)
        if span <= limit_days:
            result.append(option)
    return tuple(result)


def _spending_changes_tokens(applied):
    tokens = []
    for candidate in applied:
        if candidate.action == "stop":
            tokens.append(f"stop:{candidate.event_id}")
        else:
            tokens.append(f"reduce_to:{candidate.event_id}:{candidate.reduced_amount}")
    return tuple(tokens)


def _try_full_payment_with_changes(balance0, timeline, flexible, min_balance, requested):
    working = timeline
    applied = []
    for candidate in flexible[:3]:
        working = apply_candidate(working, candidate)
        applied.append(candidate)
        if lowest_balance(balance0, working) - min_balance >= requested:
            return applied, working
    return [], timeline


def _money(profile, amount) -> str:
    currency = getattr(profile, "home_currency", None) or ""
    cents = Decimal(amount).quantize(_CENT, rounding=ROUND_HALF_UP)
    text = f"{cents:,.2f}"
    if text.endswith(".00"):
        text = text[:-3]
    return f"{currency} {text}".strip()


def _day(value) -> str:
    return f"{value.day} {value.strftime('%B %Y')}"


def _change_phrase(candidate, profile) -> str:
    label = (candidate.category or candidate.event_id).replace("_", " ")
    if candidate.action == "stop":
        return f"stop the {label} payment ({candidate.event_id})"
    return f"reduce the {label} payment ({candidate.event_id}) to {_money(profile, candidate.reduced_amount)}"


def _explain(request, profile, safe_now, decision, lowest_after, applied=()):  # noqa: PLR0911, PLR0913, PLR0917
    """One or two plain-language sentences grounded in the forecast numbers."""
    amount = _money(profile, request.requested_amount)
    minimum = _money(profile, profile.minimum_balance_to_keep)
    method = decision.recommended_payment_method
    plan = decision.payment_plan
    keeps = (f"This leaves at least {_money(profile, lowest_after)} available over the next 90 days."
             if lowest_after is not None else f"This keeps the {minimum} minimum protected.")
    if method == "full_payment" and applied:
        changes = " and ".join(_change_phrase(item, profile) for item in applied)
        return f"{changes[0].upper()}{changes[1:]}, then pay {amount} today. {keeps}"
    if method == "full_payment":
        return f"Pay {amount} today. {keeps}"
    if method == "installments":
        return (f"Use {len(plan)} installments of {_money(profile, plan[0][1])}, starting "
                f"{_day(plan[0][0])}. {keeps}")
    if method == "partial_payment":
        return (f"Pay {_money(profile, plan[0][1])} today and the remaining {_money(profile, plan[1][1])} "
                f"on {_day(plan[1][0])}. This completes the full request and keeps the {minimum} minimum protected.")
    if method == "wait":
        return (f"Pay {amount} in full on {_day(plan[0][0])}. Paying earlier would take the balance "
                f"below the {minimum} minimum.")
    if safe_now > 0:
        return (f"Do not proceed with the {amount} request. Although {_money(profile, safe_now)} is "
                f"available today, the full amount cannot be completed safely within 90 days.")
    return (f"Do not make this payment by {_day(request.desired_completion_date)}. None of the available "
            f"options keeps the {minimum} minimum protected.")


def decide_request(context, facts_by_event=None, horizon_days=90, adjustments=()) -> ChallengeDecision:
    """Compute the full challenge decision for one loaders.dataset.RequestContext."""
    request, profile = context.request, context.profile
    start = request.request_date
    end = start + timedelta(days=horizon_days)
    timeline, flexible = build_timeline(context.events, profile, start, end, facts_by_event,
                                        adjustments=adjustments)
    balance0, min_balance = profile.current_available_balance, profile.minimum_balance_to_keep
    # Round down to the cent: a safe amount must never be overstated.
    safe_now = amount_safe_to_pay(balance0, timeline, min_balance, request.requested_amount).quantize(
        _CENT, rounding=ROUND_DOWN)
    earliest_full = earliest_full_payment_date(
        balance0, timeline, request.requested_amount, min_balance, start, end
    )
    accepted = set(profile.payment_methods_user_will_consider)
    candidates = []

    if "full_payment" in accepted and earliest_full == start:
        candidates.append(_Candidate(
            "affordable_now", "full_payment", [(start, request.requested_amount)],
            request.requested_amount, start, 1, True,
        ))
    if (request.allows_partial_payment and "partial_payment" in accepted
            and Decimal(0) < safe_now < request.requested_amount
            and earliest_full is not None):
        remaining = request.requested_amount - safe_now
        candidates.append(_Candidate(
            "affordable_with_plan", "partial_payment",
            [(start, safe_now), (earliest_full, remaining)], request.requested_amount, start, 2,
            earliest_full <= request.desired_completion_date,
        ))
    for option in _eligible_installment_options(profile, context.payment_options):
        candidate = _installment_candidate(option, request, timeline, balance0, min_balance, end, accepted)
        if candidate is not None:
            candidates.append(candidate)
    if earliest_full is not None and earliest_full > start and "full_payment" in accepted:
        candidates.append(_Candidate(
            "affordable_later", "wait", [(earliest_full, request.requested_amount)],
            request.requested_amount, earliest_full, 1,
            earliest_full <= request.desired_completion_date,
        ))

    # Spending changes are ranked against every other plan (rule 1, completing by
    # the deadline, outranks rule 2, avoiding changes), so try them whenever a
    # plain full payment today is not already safe.
    if earliest_full != start and "full_payment" in accepted:
        applied, adjusted = _try_full_payment_with_changes(
            balance0, timeline, flexible, min_balance, request.requested_amount
        )
        if applied:
            candidates.append(_Candidate(
                "affordable_with_plan", "full_payment", [(start, request.requested_amount)],
                request.requested_amount, start, 1, True, changes=_spending_changes_tokens(applied),
                applied=tuple(applied), timeline=adjusted,
            ))

    winner = min(candidates, key=_Candidate.rank_key) if candidates else None
    if winner is not None:
        status, method, plan, changes = winner.status, winner.method, winner.plan, winner.changes
        effective = winner.timeline if winner.timeline is not None else timeline
        lowest_after = lowest_balance(balance0, [*effective, *((when, -paid) for when, paid in plan)])
    else:
        status, method, plan, changes = "not_affordable", "not_recommended", [], ()
        lowest_after = None

    decision = ChallengeDecision(
        request_id=request.request_id, amount_safe_to_pay=safe_now, affordability_status=status,
        recommended_payment_method=method, payment_plan=tuple(plan),
        earliest_date_for_full_payment=earliest_full, spending_changes=changes,
        decision_explanation="",
    )
    explanation = _explain(request, profile, safe_now, decision, lowest_after,
                           winner.applied if winner is not None else ())
    return replace(decision, decision_explanation=explanation)
