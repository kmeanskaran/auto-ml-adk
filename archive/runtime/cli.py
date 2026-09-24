"""Command line for the agent orchestrator.

Exit status is 0 when an experiment completes, 3 when it is waiting for
approval, and 1 when it fails, is interrupted, or is rejected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from team.config import data_root
from team.config import detect_model_name
from team.config import load_config
from runtime.orchestrator import ExperimentError
from runtime.orchestrator import ExperimentRequest
from runtime.orchestrator import MlRuntime


def main(argv: list[str] | None = None) -> int:
  parser = _parser()
  args = parser.parse_args(argv)
  try:
    if args.command == "serve":
      return _serve(args)
    runtime = MlRuntime(root=data_root(), model_name=detect_model_name())
    if args.command == "predict":
      rows = json.loads(args.rows)
      if not isinstance(rows, list):
        raise ExperimentError("`--rows` must be a JSON list of objects.")
      print(json.dumps(runtime.predict(args.experiment_id, rows), indent=2, default=str))
      return 0
    record = asyncio.run(_dispatch(runtime, args))
  except ExperimentError as exc:
    print(f"error: {exc}")
    return 1
  except FileNotFoundError as exc:
    print(f"error: {exc}")
    return 1
  except json.JSONDecodeError as exc:
    print(f"error: rows are not valid JSON ({exc})")
    return 1
  _print_record(record, as_json=getattr(args, "json", False))
  return _exit_code(record)


async def _dispatch(runtime: MlRuntime, args: argparse.Namespace) -> dict[str, Any]:
  if args.command == "run":
    approve = True
    if args.no_approve:
      approve = False
    if args.approve:
      approve = True
    return await runtime.start(
        ExperimentRequest(
            objective=args.objective,
            target_column=args.target_column,
            metric=args.metric,
            target_value=args.target,
            dataset_path=args.dataset,
            approve_training=approve,
            fault_tool=args.fault_tool,
            user_id=args.user,
        )
    )
  if args.command == "status":
    return runtime.get(args.experiment_id)
  if args.command == "approve":
    return await runtime.approve(args.experiment_id, confirmed=not args.reject)
  if args.command == "resume":
    return await runtime.resume(args.experiment_id)
  if args.command == "iterate":
    return await runtime.iterate(
        args.experiment_id,
        note=args.note,
        target_value=args.target,
    )
  raise ExperimentError(f"Unknown command {args.command}.")


def _serve(args: argparse.Namespace) -> int:
  import uvicorn

  from api.app import create_app

  uvicorn.run(
      create_app(),
      host=args.host,
      port=args.port,
      log_level=args.log_level.lower(),
  )
  return 0


def _parser() -> argparse.ArgumentParser:
  cfg = load_config()
  parser = argparse.ArgumentParser(
      prog="runtime.cli",
      description="Run the ML engineer agents and the FastAPI service.",
  )
  commands = parser.add_subparsers(dest="command", required=True)
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument(
      "--json",
      action="store_true",
      help="Print the full experiment record as JSON.",
  )

  run = commands.add_parser("run", parents=[common], help="Start an experiment.")
  run.add_argument("--objective", default=cfg.experiment.objective)
  run.add_argument("--target-column", default=cfg.experiment.target_column)
  run.add_argument("--metric", default=cfg.experiment.metric)
  run.add_argument("--target", type=float, default=cfg.experiment.target_value)
  run.add_argument("--dataset", help="CSV path. Omit to use the sample churn table.")
  run.add_argument("--approve", action="store_true", help="Pause before the first training fit.")
  run.add_argument("--no-approve", action="store_true", help="Train without a human gate.")
  run.add_argument("--fault-tool", help="Interrupt once inside this tool, for resume drills.")
  run.add_argument("--user", default=cfg.runtime.user)

  serve = commands.add_parser("serve", help="Start the FastAPI service.")
  serve.add_argument("--host", default=cfg.api.host)
  serve.add_argument("--port", type=int, default=cfg.api.port)
  serve.add_argument("--log-level", default=cfg.api.log_level.lower())

  status = commands.add_parser("status", parents=[common], help="Show one experiment.")
  status.add_argument("experiment_id")

  approve = commands.add_parser(
      "approve",
      parents=[common],
      help="Approve or reject a paused training step.",
  )
  approve.add_argument("experiment_id")
  approve.add_argument("--reject", action="store_true")

  resume = commands.add_parser(
      "resume",
      parents=[common],
      help="Continue an interrupted experiment.",
  )
  resume.add_argument("experiment_id")

  iterate = commands.add_parser("iterate", parents=[common], help="Retrain from a human note.")
  iterate.add_argument("experiment_id")
  iterate.add_argument("--note", required=True)
  iterate.add_argument("--target", type=float)

  predict = commands.add_parser("predict", help="Score rows with the saved model.")
  predict.add_argument("experiment_id")
  predict.add_argument("--rows", required=True, help="JSON list of row objects.")
  return parser


def _print_record(record: dict[str, Any], *, as_json: bool) -> None:
  if as_json:
    print(json.dumps(record, indent=2, default=str))
    return
  best = record.get("best") or {}
  metrics = best.get("metrics") or {}
  print(f"experiment: {record.get('id')}")
  print(f"status: {record.get('status')}")
  if record.get("error"):
    print(f"error: {record['error']}")
  if best.get("model"):
    print(f"model: {best.get('model')} f1={metrics.get('f1')}")
  if record.get("design_path"):
    print(f"design: {record['design_path']}")
  pending = record.get("pending_approval") or {}
  if pending:
    print(f"waiting_for: {pending.get('tool')}")
    print(f"approve_with: python -m runtime.cli approve {record.get('id')}")


def _exit_code(record: dict[str, Any]) -> int:
  status = record.get("status")
  if status == "completed":
    return 0
  if status == "awaiting_approval":
    return 3
  return 1


if __name__ == "__main__":
  raise SystemExit(main())
