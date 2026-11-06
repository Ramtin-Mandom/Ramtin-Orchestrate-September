"""Recurring income/expense detection and its effect on the 90-day forecast."""
# ruff: noqa: PLR2004, PLR0913

from datetime import date, timedelta
from decimal import Decimal
from types import SimpleNamespace

from finance.challenge_forecast import apply_candidate, build_timeline
from finance.recurrence import detect_recurring_series

START = date(2026, 3, 7)
END = START + timedelta(days=90)


def event(event_id, when, amount, *, category="rent", direction="debit", event_type="expense",
          status="settled", description="Rent", flexibility="fixed", minimum=None, settled=None):
    record = SimpleNamespace(
        event_id=event_id, event_type=event_type, description=description, category=category,
        direction=direction, event_date=when, settlement_date=settled or when, status=status,
        flexibility=flexibility, linked_event_id=None, amount=Decimal(amount), currency="EUR",
        minimum_allowed_amount=None,
    )
    return SimpleNamespace(event=record, home_amount=Decimal(amount), exchange_rate=None,
                           home_minimum_allowed_amount=None if minimum is None else Decimal(minimum),
                           messages=(), images=())


def monthly(prefix, day, amount, months=(12, 1, 2), **kwargs):
    years = {12: 2025, 1: 2026, 2: 2026, 3: 2026}
    return [event(f"{prefix}{i}", date(years[m], m, day), amount, **kwargs) for i, m in enumerate(months)]


def profile(**overrides):
    base = {
        "current_available_balance": Decimal(5000), "minimum_balance_to_keep": Decimal(1000),
        "expense_categories_to_protect": (), "expense_categories_user_is_willing_to_stop": (),
        "expense_categories_user_is_willing_to_reduce": (),
        "payment_methods_user_will_consider": ("full_payment",), "max_installment_months": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_monthly_bill_is_projected_on_the_same_day_each_month():
    events = monthly("r", 2, 600, months=(1, 2, 3))
    (series,) = detect_recurring_series(events, START, END)
    assert series.occurrences == (date(2026, 4, 2), date(2026, 5, 2), date(2026, 6, 2))
    assert series.amount == Decimal(600)
    assert series.last_event_id == "r2"


def test_weekly_variable_spending_uses_conservative_recent_mean():
    amounts = [90, 100, 110, 101]
    events = [event(f"g{i}", START - timedelta(days=7 * (4 - i)), amount, category="groceries")
              for i, amount in enumerate(amounts)]
    (series,) = detect_recurring_series(events, START, END)
    assert series.step_days == 7
    assert series.amount == Decimal("103.67")  # mean of last three, rounded up to the cent
    assert series.occurrences[0] == START + timedelta(days=7)


def test_one_off_purchase_is_not_projected():
    events = [event("tv", date(2026, 2, 20), 900, category="shopping")]
    assert detect_recurring_series(events, START, END) == []


def test_series_that_missed_its_latest_occurrence_is_treated_as_stopped():
    events = monthly("s", 20, 800, months=(12, 1), category="salary", direction="credit",
                     event_type="income", description="Second income")
    assert detect_recurring_series(events, START, END) == []


def test_salary_is_projected_from_a_confirmed_scheduled_row():
    events = [
        event("p0", date(2026, 2, 15), 500, category="salary", direction="credit", event_type="income"),
        event("p1", date(2026, 3, 15), 2300, category="salary", direction="credit", event_type="income",
              status="scheduled"),
    ]
    (series,) = detect_recurring_series(events, START, END)
    assert series.amount == Decimal(2300)
    assert series.occurrences == (date(2026, 4, 15), date(2026, 5, 15))


def test_irregular_gig_payouts_are_not_counted_as_income():
    amounts = [650, 410, 820, 480]
    events = [event(f"gig{i}", date(2025 + (m < 6), m, 4), amount, category="salary", direction="credit",
                    event_type="income")
              for i, (m, amount) in enumerate(zip((11, 12, 1, 2), amounts))]
    assert detect_recurring_series(events, START, END) == []


def test_final_payroll_ends_the_income_series():
    events = monthly("f", 15, 1500, months=(12, 1, 2), category="salary", direction="credit",
                     event_type="income", description="Final employer payroll")
    assert detect_recurring_series(events, START, END) == []


def test_late_salary_shifts_the_next_projected_payday():
    paydays = [date(2025, 11, 15), date(2025, 12, 15), date(2026, 1, 15), date(2026, 2, 15)]
    events = [event(f"w{i}", day, 1500, category="salary", direction="credit", event_type="income",
                    settled=date(2026, 2, 23) if i == 3 else None)
              for i, day in enumerate(paydays)]
    (series,) = detect_recurring_series(events, START, END)
    assert series.occurrences[0] == date(2026, 3, 23)


def test_existing_scheduled_row_is_not_double_counted():
    events = [*monthly("r", 2, 600, months=(1, 2, 3)),
              event("r-next", date(2026, 4, 2), 600, status="scheduled")]
    (series,) = detect_recurring_series(events, START, END)
    assert date(2026, 4, 2) not in series.occurrences


def test_refunds_are_never_projected():
    events = monthly("rf", 10, 50, months=(12, 1, 2), category="shopping", direction="credit",
                     event_type="refund")
    assert detect_recurring_series(events, START, END) == []


def test_flexible_series_can_be_stopped_across_every_projected_occurrence():
    events = monthly("st", 10, 20, months=(12, 1, 2), category="streaming",
                     event_type="subscription", flexibility="stoppable")
    timeline, flexible = build_timeline(
        events, profile(expense_categories_user_is_willing_to_stop=("streaming",)), START, END)
    (candidate,) = flexible
    assert candidate.event_id == "st2" and candidate.action == "stop"
    assert candidate.freed == Decimal(60)
    assert apply_candidate(timeline, candidate) == []


def test_reducible_series_is_reduced_to_its_minimum_allowed_amount():
    events = monthly("d", 10, 80, months=(1, 2, 3), category="dining", flexibility="reducible", minimum=30)
    timeline, flexible = build_timeline(
        events, profile(expense_categories_user_is_willing_to_reduce=("dining",)), START, END)
    (candidate,) = flexible
    assert candidate.reduced_amount == Decimal(30)
    assert [delta for _, delta in apply_candidate(timeline, candidate)] == [Decimal(-30)] * 3


def test_protected_category_is_never_a_spending_change():
    events = monthly("st", 10, 20, months=(1, 2, 3), category="streaming", flexibility="stoppable")
    _timeline, flexible = build_timeline(events, profile(
        expense_categories_user_is_willing_to_stop=("streaming",),
        expense_categories_to_protect=("streaming",)), START, END)
    assert flexible == []


def test_projection_can_be_disabled():
    events = monthly("r", 2, 600, months=(1, 2, 3))
    timeline, _ = build_timeline(events, profile(), START, END, project_recurring=False)
    assert timeline == []
