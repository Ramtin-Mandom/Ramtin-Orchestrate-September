"""Deterministic joins and dated currency conversion; no cash-flow inference."""

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Optional, Tuple

from models import ChallengeRequest

from .dataset_schema import DatasetValidationError, SourceRecord, read_csv, records
from .requests import _parse_challenge_row
from .sample_outputs import load_sample_outputs


@dataclass(frozen=True)
class ConvertedEvent:
    event: SourceRecord
    home_currency: str
    home_amount: Optional[Decimal]
    home_minimum_allowed_amount: Optional[Decimal]
    exchange_rate: Optional[SourceRecord]
    messages: Tuple[SourceRecord, ...]
    images: Tuple[SourceRecord, ...]


@dataclass(frozen=True)
class RequestContext:
    request: ChallengeRequest
    profile: SourceRecord
    events: Tuple[ConvertedEvent, ...]
    payment_options: Tuple[SourceRecord, ...]
    messages: Tuple[SourceRecord, ...]
    images: Tuple[SourceRecord, ...]


@dataclass(frozen=True)
class LoadedDataset:
    contexts: Tuple[RequestContext, ...]
    sample_requests: Tuple[ChallengeRequest, ...]
    sample_outputs: Tuple[SourceRecord, ...]
    tables: dict
    sample_contexts: Tuple[RequestContext, ...] = ()


def index_unique(items, key, label):
    result = {}
    for item in items:
        identifier = key(item)
        if identifier in result:
            raise DatasetValidationError(f"{label}: duplicate ID/key {identifier}")
        result[identifier] = item
    return result


def group(items, key):
    result = defaultdict(list)
    for item in items:
        result[getattr(item, key)].append(item)
    return result


def convert_event(event, profile, rates, messages, images):
    rate = None
    amount = event.amount
    minimum = event.minimum_allowed_amount
    if event.currency != profile.home_currency:
        rate_date = event.settlement_date or event.event_date
        key = (rate_date, event.currency, profile.home_currency)
        rate = rates.get(key)
        if rate is None:
            raise DatasetValidationError(
                f"{event.event_id}: no supplied exchange rate for {key}"
            )
        amount = _multiply(amount, rate.rate)
        minimum = _multiply(minimum, rate.rate)
    return ConvertedEvent(event, profile.home_currency, amount, minimum, rate, tuple(messages), tuple(images))


def _multiply(amount, rate):
    if amount is None:
        return None
    # Exact multiplication independent of the process decimal context.
    with localcontext() as context:
        context.prec = len(amount.as_tuple().digits) + len(rate.as_tuple().digits)
        return amount * rate


def _requests(rows):
    result = []
    for row, provenance in rows:
        try:
            result.append(_parse_challenge_row(row, provenance.path, provenance.row_number))
        except (ValueError, TypeError) as exc:
            raise DatasetValidationError(f"{Path(provenance.path).name}: row {provenance.row_number}: {exc}") from exc
    return tuple(result)


def _validate_references(tables, requests, profiles, events):
    for name in ("financial_events", "messages", "images"):
        for row in tables[name]:
            if row.user_id not in profiles:
                raise DatasetValidationError(f"{name}: unknown user_id {row.user_id}")
            request_id = row.values.get("request_id")
            if request_id and (request_id not in requests or requests[request_id].user_id != row.user_id):
                raise DatasetValidationError(f"{name}: missing or mismatched request_id {request_id}")
            event_id = row.values.get("related_event_id") or row.values.get("linked_event_id")
            if event_id and (event_id not in events or events[event_id].user_id != row.user_id):
                raise DatasetValidationError(f"{name}: missing or mismatched event reference {event_id}")
            if event_id and event_id == row.values.get("event_id"):
                raise DatasetValidationError(f"{name}: self-linked event {event_id}")
    for option in tables["request_payment_options"]:
        if option.request_id not in requests:
            raise DatasetValidationError(f"payment option: unknown request_id {option.request_id}")
        if option.number_of_payments > 1 and option.payment_frequency_days is None:
            raise DatasetValidationError(f"{option.payment_option_id}: payment interval required")


def load_dataset(root="dataset"):
    """Read every participant CSV and join evaluation contexts in input order.

    Evidence scoped to a request stays with that request. User evidence with a
    blank request_id applies to every request for that user. Event associations
    additionally remain attached to the exact event. Samples validate references
    but never become evaluation requests. All CSV order and source IDs survive.
    Rates must match the exact settlement date (event date for unsettled rows)
    and stated direction; no inverses, carry-forward, triangulation or rounding.
    Unknown image amounts remain None for later interpretation.
    """
    root = Path(root).resolve()
    request_rows = read_csv(root, "requests")
    sample_rows = read_csv(root, "sample_requests")
    evaluation, samples = _requests(request_rows), _requests(sample_rows)
    tables = {name: records(read_csv(root, name)) for name in (
        "financial_profiles", "financial_events", "exchange_rates",
        "request_payment_options", "messages", "images",
    )}
    requests = index_unique(evaluation + samples, lambda r: r.request_id, "requests")
    profiles = index_unique(tables["financial_profiles"], lambda r: r.user_id, "profiles")
    events = index_unique(tables["financial_events"], lambda r: r.event_id, "events")
    for name, key in (("request_payment_options", "payment_option_id"), ("messages", "message_id"), ("images", "image_id")):
        index_unique(tables[name], lambda r, key=key: getattr(r, key), name)
    rates = index_unique(tables["exchange_rates"], lambda r: (r.rate_date, r.from_currency, r.to_currency), "rates")
    for request in requests.values():
        if request.user_id not in profiles:
            raise DatasetValidationError(f"{request.request_id}: unknown user_id {request.user_id}")
    _validate_references(tables, requests, profiles, events)
    for image in tables["images"]:
        path = root / "media" / "images" / f"{image.image_id}.png"
        if not path.is_file() or root not in path.resolve().parents:
            raise DatasetValidationError(f"{image.image_id}: image file missing or outside dataset")
        image.values["path"] = path
    template = read_csv(root, "output")
    template_ids = index_unique(template, lambda r: r[0]["request_id"], "output")
    if tuple(template_ids) != tuple(r.request_id for r in evaluation):
        raise DatasetValidationError("output template request IDs must match evaluation order")
    if any(value for row, _ in template for key, value in row.items() if key != "request_id"):
        raise DatasetValidationError("output template must be blank")
    tables["output"] = tuple(row for row, _ in template)
    by_user = group(tables["financial_events"], "user_id")
    options = group(tables["request_payment_options"], "request_id")
    messages = group(tables["messages"], "user_id")
    images = group(tables["images"], "user_id")

    def context_for(request):
        profile = profiles[request.user_id]
        relevant_messages = tuple(r for r in messages[request.user_id] if r.request_id in (None, request.request_id))
        relevant_images = tuple(r for r in images[request.user_id] if r.request_id in (None, request.request_id))
        event_messages = group(relevant_messages, "related_event_id")
        event_images = group(relevant_images, "related_event_id")
        converted = tuple(convert_event(e, profile, rates, event_messages[e.event_id], event_images[e.event_id]) for e in by_user[request.user_id])
        return RequestContext(request, profile, converted, tuple(options[request.request_id]), relevant_messages, relevant_images)

    contexts = tuple(context_for(request) for request in evaluation)
    sample_contexts = tuple(context_for(request) for request in samples)
    return LoadedDataset(contexts, samples, load_sample_outputs(sample_rows), tables, sample_contexts)
