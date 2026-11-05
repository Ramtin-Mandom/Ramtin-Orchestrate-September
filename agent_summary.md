# Agent summary: how the Buy or Wait? agent was built and what it does

## What it does

The agent is a financial decision assistant. It receives a purchase or payment
request ("Can I complete this family transfer and still keep my minimum balance?"). Using only the
data it is given, it decides whether the payment is safe. If it is, the agent
also decides *how* to pay.

For every request in `dataset/requests.csv` it writes one row to `output.csv`:

| Output | Meaning |
|---|---|
| `amount_safe_to_pay` | The most the user can pay **today** without their balance ever dropping below the minimum they want to keep over the next 90 days |
| `affordability_status` | `affordable_now`, `affordable_with_plan`, `affordable_later` or `not_affordable` |
| `recommended_payment_method` | `full_payment`, `partial_payment`, `installments`, `wait` or `not_recommended` |
| `payment_plan` | The exact dated payments, e.g. `2024-09-12:68432\|2024-10-10:68432\|2024-11-07:68432` |
| `earliest_date_for_full_payment` | The first date one full payment becomes safe, or blank if never within 90 days |
| `spending_changes_needed` | Flexible expenses to stop or reduce, e.g. `stop:event_476`, or `none` |
| `decision_explanation` | One or two plain sentences, e.g. "Pay IDR 18,164,000 in full on 15 November 2024. Paying earlier would take the balance below the IDR 16,588,900 minimum." |

In short, it answers **"buy now, buy with a plan, buy later, or don't buy"**
for each request, and gives a reason built from the numbers.

## Inputs it reads

- **Financial profile.** The current balance, the minimum balance to keep,
  which spending categories are protected or flexible, which payment methods
  the user will consider, and how long an installment plan they accept.
- **Financial events.** Settled history, pending and scheduled payments,
  failed or cancelled transactions, refunds and investments. There are about
  25,000 rows across 275 users, in five currencies.
- **Exchange rates.** Fixed, dated rates for each foreign-currency event.
- **Payment options.** The seller's full-payment and installment offers for
  each request.
- **Messages and receipt images.** Notes from employers, banks and merchants,
  in English and Indonesian. They are treated as **untrusted** evidence.

## How it works, step by step

```text
CSVs ──► load & validate ──► reconcile events ──► detect recurring cash flow ─┐
                                                                               ├─► 90-day forecast ──► choose plan ──► output.csv
messages/images ──► LLM (typed facts only) ──► validate ──► adjustments ──────┘
```

1. **Load and validate** ([`code/loaders/`](code/loaders/)).
   - All nine CSVs are schema-checked and every ID reference is verified.
   - Foreign-currency amounts are converted with the exact rate for their
     settlement date. There is no guessing and no inverse rates.
   - Each request gets a self-contained "context": the user's profile, events,
     payment options and evidence.
2. **Reconcile each transaction**
   ([`code/extraction/event_reconciliation.py`](code/extraction/event_reconciliation.py)).
   - Linked events are resolved into their final state, such as an
     authorization that was later settled, a charge that was reversed, an
     amended bill or a duplicate.
   - Conflicts follow the challenge's priority order: an explicit
     cancellation, settlement or amendment first, then the newer record, then
     the settled record, then the financially safer reading.
3. **Detect recurring income and expenses**
   ([`code/finance/recurrence.py`](code/finance/recurrence.py)).
   - Settled history is grouped into series: expenses by category, salary by
     payday.
   - A series is projected only if its rhythm is regular (every N days, or
     monthly on the same day) and it hasn't missed its most recent occurrence.
   - Bills are reserved at the mean of their last three amounts, rounded up,
     which is deliberately conservative.
   - Salary uses its latest confirmed amount. A late-paid salary moves the
     following paydays with it.
   - Income that swings from payment to payment (gig payouts) is not counted
     as dependable. A payroll labelled "final" ends the income.
4. **Read messages with an LLM**
   ([`code/extraction/message_interpreter.py`](code/extraction/message_interpreter.py)).
   This is the AI part of the agent.
   - Each message that isn't tied to a specific transaction goes to
     `gpt-4o-mini`. The model restates it as zero or more typed adjustments:
     `salary_amount`, `salary_date`, `income_ended`, `confirmed_income`,
     `expense_amount` or `new_recurring_expense`.
   - Messages and images about one specific transaction go through a separate
     per-event extractor
     ([`code/extraction/event_extractor.py`](code/extraction/event_extractor.py)).
     It confirms, cancels, delays or fills in the amount of that one event.
5. **Apply the adjustments** ([`code/finance/adjustments.py`](code/finance/adjustments.py)).
   Plain code applies each adjustment to the projections, oldest message
   first so the newest wins. Examples:
   - a salary change from its effective date
   - a moved payday
   - an ended contract
   - a percentage rent increase
6. **Forecast 90 days**
   ([`code/finance/challenge_forecast.py`](code/finance/challenge_forecast.py)).
   - It builds a day-by-day balance from the current balance, every reserved
     debit and every confirmed credit.
   - Pending credits, bonuses, refunds and unrealized investment gains are
     never counted until they settle.
   - The safe amount is the cushion at the forecast's lowest point, rounded
     down to the cent.
7. **Choose a plan** ([`code/decision/challenge_decision.py`](code/decision/challenge_decision.py)).
   Each plan the user accepts is simulated on the forecast:
   - full payment today
   - partial payment (the safe amount now, the rest on the earliest safe date)
   - every seller installment option
   - waiting
   - stopping or reducing up to three flexible, non-protected expenses

   Unsafe plans are discarded. The rest are ranked: finish by the deadline,
   need no spending changes, lowest total cost, earliest start, fewest
   payments, lowest option ID. The winner becomes the recommendation, and a
   template turns its numbers into the explanation.

## Where the AI is used, and where it isn't

The design rule is that **the LLM reads and code decides**.

- The model only turns free text and images into small typed facts. It never
  sees the balance and never computes an amount.
- It never chooses a recommendation. Every number in `output.csv` comes from
  deterministic Python, so the same inputs always give the same output.
- Messages are untrusted, and some contain instructions aimed at the agent.
  The defences are:
  - trusted context (request date, home currency, known categories) travels
    in a separate system message
  - output must match a strict JSON schema and is re-validated locally;
    unknown categories and foreign-currency amounts are rejected
  - messages dated after the request are ignored
  - pending or unapproved money is excluded
- LLM output is cached by message text, model and prompt version, so re-runs
  are free and reproducible.
- The AI step is opt-in (`--use-ai-extraction`). Without it the agent still
  runs end to end on rules alone, offline and at no cost.

## How it was built

The project grew over several iterations, each one checked against the 25
public sample answers (`python code/main.py --evaluate-samples`).

1. **Foundation.** Typed, `Decimal`-based data models, strict CSV loaders,
   currency conversion, an exact-schema writer and a large offline test suite.
2. **Lifecycle reconciliation and first decision engine.** A 90-day forecast
   over CSV pending and scheduled rows only, plus the ranking rules. Scored on
   the samples, it got only **48%** of affordability verdicts right. It
   ignored the rent, groceries and salary that obviously recur.
3. **Recurrence detection.** The specification says to forecast from
   "recurring income and expenses". A scratch simulator was used to test rule
   variants (grouping, rhythm limits, amount rules, income stability) against
   the samples. The adopted rules can each be justified from the
   specification. Variants that fit one or two more samples were rejected as
   overfitting. Result: **80%**.
4. **LLM message reading.** Most messages were general updates, such as pay
   raises, moved paydays, ended contracts and rent increases, which the
   per-event extractor never saw. The message interpreter and adjustment layer
   were added. Result: **88% status and 96% payment method**.
5. **Presentation.** Human-readable explanations, a sample-scoring command,
   CI, documentation and a token and cost report.

### Sample-set results (25 public answers)

| Field | Initial | + recurrence | + LLM messages |
|---|---|---|---|
| affordability_status | 48% | 80% | **88%** |
| recommended_payment_method | 56% | 88% | **96%** |
| earliest_date_for_full_payment | 22% | 67% | **72%** |
| amount_safe_to_pay (exact to the cent) | 16% | 16% | 16% |

The safe amount is usually close but rarely exact to the cent. The reference
generator's precise forecasting assumptions are not published.

### Tech stack

- Python 3.8+, standard library only for the core logic (`decimal`,
  `dataclasses`, `csv`)
- The official OpenAI SDK (`gpt-4o-mini`) with strict JSON-schema structured
  outputs
- pytest, with every model call mocked, and Ruff for linting
- GitHub Actions CI

### Model usage on the full dataset

The full 250-request run made **209 `gpt-4o-mini` calls** using **501,201
tokens**, for an estimated **$0.08**. That is about 2,005 tokens and $0.0003
per request. Re-runs reuse cached message interpretations. Details are in
[`code/evaluation/usage_report.md`](code/evaluation/usage_report.md).

### What it recommended on the 250 evaluation requests

| Recommendation | Requests |
|---|---|
| Pay in full now | 69 |
| Wait for a safe date | 57 |
| Installments | 55 |
| Partial payment now and the rest later | 8 |
| Don't proceed | 61 |

## Where to look in the code

| Question | File |
|---|---|
| How is "safe to pay" computed? | `code/finance/challenge_forecast.py` |
| How are rent, salary and other recurring items found? | `code/finance/recurrence.py` |
| What does the LLM see and return? | `code/extraction/message_interpreter.py` |
| How are LLM facts turned into forecast changes? | `code/finance/adjustments.py` |
| How is the final plan chosen and explained? | `code/decision/challenge_decision.py` |
| How is everything wired together? | `code/challenge_pipeline.py`, `code/main.py` |
