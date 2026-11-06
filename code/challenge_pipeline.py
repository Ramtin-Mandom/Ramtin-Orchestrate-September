"""Orchestrate the challenge dataset into a completed output.csv.

AI extraction is opt-in (`use_ai_extraction=True`): it costs real API calls
and time, so a default run stays fully deterministic and free, reconciling
each event against its CSV facts alone (facts_by_event is then empty, which
`reconcile_event` treats as a no-op identity pass). Enable it once an
OPENAI_API_KEY is configured to also resolve blank amounts and status
changes described only in messages/images.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from decision.challenge_decision import ChallengeDecision, decide_request
from evaluation.metrics import EvaluationReport, evaluate_predictions
from extraction.event_extractor import UsageRecord, extract_event_facts
from extraction.message_interpreter import MessageInterpreter
from loaders import load_dataset
from loaders.dataset_schema import HEADERS
from output.challenge_writer import challenge_row, write_challenge_csv

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ChallengePipelineResult:
    decisions: Tuple[ChallengeDecision, ...]
    usage: Tuple[UsageRecord, ...]


def _cache_key(event):
    """Same event_id plus the exact evidence visible in this request's scope.

    A message/image can be scoped to one request_id, so the same event_id may
    expose different evidence across a user's requests; keying on the visible
    IDs (not just event_id) still skips a repeat call whenever the evidence is
    identical, without ever reusing facts extracted from different media.
    """
    return (
        event.event.event_id,
        tuple(sorted(item.message_id for item in event.messages)),
        tuple(sorted(item.image_id for item in event.images)),
    )


def _collect_facts(context, extractor: Optional[Callable], cache: dict, usage: list):
    """Extract once per (event, visible evidence); cache across a user's requests."""
    facts_by_event = defaultdict(list)
    for event in context.events:
        key = _cache_key(event)
        if key not in cache:
            confirmed = []
            for source in extract_event_facts(event, context, request=extractor):
                if source.usage is not None:
                    usage.append(source.usage)
                confirmed.extend(fact for fact in source.facts if fact.certainty == "confirmed")
            cache[key] = confirmed
        facts_by_event[event.event.event_id].extend(cache[key])
    return facts_by_event


def run_challenge_pipeline(  # noqa: PLR0913 -- public pipeline entry point
    dataset_root: PathLike, output_path: PathLike, *,
    use_ai_extraction: bool = False, extractor: Optional[Callable] = None,
    horizon_days: int = 90, interpreter: Optional[MessageInterpreter] = None,
) -> ChallengePipelineResult:
    """Load the challenge dataset, decide every request, and write output_path.

    Requests are processed in `requests.csv` order; every request_id gets
    exactly one row. With use_ai_extraction, each event's messages/images are
    sent through extraction.event_extractor before reconciliation, cached by
    event_id since the same event recurs across every request for its user;
    otherwise reconciliation runs against the CSV facts only (still fully
    correct for events with no lifecycle changes described in media).
    """
    dataset = load_dataset(dataset_root)
    decisions, usage = _decide_all(dataset.contexts, use_ai_extraction, extractor, interpreter, horizon_days)
    write_challenge_csv(decisions, output_path)
    return ChallengePipelineResult(tuple(decisions), tuple(usage))


def _decide_all(contexts, use_ai_extraction, extractor, interpreter, horizon_days):
    """Decide each context; with AI on, add event facts and message adjustments."""
    usage: list = []
    cache: dict = {}
    if use_ai_extraction and interpreter is None:
        interpreter = MessageInterpreter()
    decisions = []
    for context in contexts:
        facts_by_event, adjustments = {}, ()
        if use_ai_extraction:
            facts_by_event = _collect_facts(context, extractor, cache, usage)
            adjustments = interpreter.adjustments_for(context)
        decisions.append(decide_request(context, facts_by_event, horizon_days, adjustments))
    if use_ai_extraction:
        interpreter.save()
        usage.extend(interpreter.usage)
    return decisions, usage


def evaluate_samples(  # noqa: PLR0913 -- public evaluation entry point
    dataset_root: PathLike, output_path: Optional[PathLike] = None, *, horizon_days: int = 90,
    use_ai_extraction: bool = False, extractor: Optional[Callable] = None,
    interpreter: Optional[MessageInterpreter] = None,
) -> Tuple[EvaluationReport, Tuple[UsageRecord, ...]]:
    """Decide every public sample request and score it against its published answer.

    `sample_requests.csv` ships with completed output fields, so this is the
    project's offline accuracy check. It never touches the evaluation requests,
    and optionally writes the sample predictions for side-by-side inspection.
    """
    dataset = load_dataset(dataset_root)
    decisions, usage = _decide_all(dataset.sample_contexts, use_ai_extraction, extractor, interpreter,
                                   horizon_days)
    if output_path is not None:
        write_challenge_csv(decisions, output_path)
    columns = HEADERS["output"]
    predicted = [dict(zip(columns, challenge_row(decision))) for decision in decisions]
    expected = [{column: record.values.get(column, "") or "" for column in columns}
                for record in dataset.sample_outputs]
    return evaluate_predictions(predicted, expected), tuple(usage)
