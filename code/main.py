"""Load challenge contexts by default; opt into the legacy decision pipeline."""

import argparse
import sys
from datetime import date
from pathlib import Path

from challenge_pipeline import evaluate_samples, run_challenge_pipeline
from evaluation.metrics import format_summary
from loaders import DatasetValidationError
from logging_config import LOG_LEVELS, configure_logging
from pipeline import DEFAULT_HORIZON_DAYS, PipelineError, run_pipeline


def _date(value):
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("use YYYY-MM-DD") from None


def _print_usage(usage):
    """Token totals per model for the usage report (cached results cost nothing)."""
    by_model = {}
    for record in usage:
        calls, tokens_in, tokens_out = by_model.get(record.model, (0, 0, 0))
        by_model[record.model] = (calls + 1, tokens_in + (record.input_tokens or 0),
                                  tokens_out + (record.output_tokens or 0))
    for model, (calls, tokens_in, tokens_out) in sorted(by_model.items()):
        print(f"Model {model}: calls={calls} input_tokens={tokens_in} output_tokens={tokens_out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Load and validate Buy or Wait challenge contexts")
    parser.add_argument("--mode", choices=("challenge", "legacy"), default="challenge",
                        help="challenge loads joined contexts; legacy runs the synthetic decision pipeline")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"),
                        help="challenge dataset directory (default: dataset)")
    parser.add_argument("--input", type=Path, default=Path("dataset/requests.csv"),
                        help="legacy-mode structured requests CSV")
    parser.add_argument("--media-root", "--media", type=Path, default=Path("dataset/media"),
                        help="associated local media root (default: dataset/media)")
    parser.add_argument("--output", type=Path, default=Path("output.csv"),
                        help="destination CSV (default: output.csv)")
    parser.add_argument("--as-of-date", type=_date,
                        help="override all row request_date values with YYYY-MM-DD")
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS,
                        help="forecast days after the request date (default: 90)")
    parser.add_argument("--use-ai-extraction", action="store_true",
                        help="resolve blank amounts/lifecycle changes from messages and images "
                             "via OPENAI_API_KEY (off by default: no billable calls)")
    parser.add_argument("--evaluate-samples", action="store_true",
                        help="score the agent on the public sample_requests.csv answers instead of "
                             "writing output.csv (predictions go to --output when it is given)")
    parser.add_argument("--log-level", type=str.upper, choices=LOG_LEVELS, default="INFO",
                        help="stderr application log level (default: INFO)")
    args = parser.parse_args(argv)
    logger = configure_logging(args.log_level)
    if args.mode == "challenge" and args.evaluate_samples:
        destination = args.output if "--output" in (argv or sys.argv) else None
        try:
            report, usage = evaluate_samples(args.dataset, destination, horizon_days=args.horizon_days,
                                             use_ai_extraction=args.use_ai_extraction)
        except DatasetValidationError as exc:
            parser.error(str(exc))
        print(format_summary(report))
        _print_usage(usage)
        return report
    if args.mode == "challenge":
        try:
            result = run_challenge_pipeline(
                args.dataset, args.output, use_ai_extraction=args.use_ai_extraction,
                horizon_days=args.horizon_days,
            )
        except DatasetValidationError as exc:
            parser.error(str(exc))
        print(f"Wrote {len(result.decisions)} challenge decisions to {args.output}.")
        _print_usage(result.usage)
        return result
    try:
        results = run_pipeline(args.input, args.media_root, args.output,
                               as_of_date=args.as_of_date, horizon_days=args.horizon_days)
    except PipelineError as exc:
        logger.error("Pipeline failed (%s)", type(exc).__name__)
        parser.error(str(exc))
    print(f"Wrote {len(results)} decisions to {args.output}.")
    if results.failures:
        logger.error("Batch completed with failed requests=%d", len(results.failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
