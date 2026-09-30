"""Command line.

  python cli.py update-sanctions
  python cli.py screen --type company --name "ТОО Ромашка" --id 123456789012 \
      --purpose "Проверка поставщика перед заключением договора" --operator analyst1
  python cli.py screen --type person --name "Иванов Иван Иванович" --birth-date 1980-05-01 \
      --city Алматы --purpose "KYC при открытии счёта" --no-deep
  python cli.py demo [--out docs]    # recorded fictional run, no API key or network
"""
import argparse
import getpass
import json
import sys
from datetime import date
from pathlib import Path

from config import settings


def _screen(args: argparse.Namespace) -> int:
    from models import Subject
    from pipeline import screen

    subject = Subject(
        type=args.type, name=args.name, identifier=args.id,
        birth_date=date.fromisoformat(args.birth_date) if args.birth_date else None,
        city=args.city, aliases=args.alias or [], related=args.related or [], purpose=args.purpose,
    )
    report = screen(subject, operator=args.operator, deep=not args.no_deep, max_rounds=args.rounds)
    out_dir = Path(args.out or settings.REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{report.request_id}.md"
    md_path.write_text(report.markdown, encoding="utf-8")
    (out_dir / f"{report.request_id}.json").write_text(
        report.model_dump_json(indent=2, exclude={"markdown"}), encoding="utf-8")
    print(json.dumps({"request_id": report.request_id, "risk_level": report.risk_level,
                      "risk_score": report.risk_score, "findings": len(report.findings),
                      "possible_namesakes": len(report.possible_namesakes),
                      "report": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def _demo(args: argparse.Namespace) -> int:
    from demo import run_demo

    report, md_path = run_demo(args.out)
    print(json.dumps({"demo": "вымышленные данные", "risk_level": report.risk_level,
                      "risk_score": report.risk_score, "facts": len(report.facts),
                      "possible_namesakes": len(report.possible_namesakes),
                      "report": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def _label(args: argparse.Namespace) -> int:
    import store

    n = store.set_label(args.finding_id, 1 if args.decision == "same" else 0, reviewer=args.reviewer,
                        request_id=args.request_id)
    if not n:
        print(f"no stored finding with id {args.finding_id}", file=sys.stderr)
        return 1
    print(f"labelled {n} finding(s) as {args.decision}")
    return 0


def _update_sanctions(_: argparse.Namespace) -> int:
    from sources.sanctions import download

    path = download()
    print(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1251/cp866
    p = argparse.ArgumentParser(description="Adverse media / due diligence screening")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("screen", help="screen a company or a person")
    s.add_argument("--type", choices=["company", "person"], required=True)
    s.add_argument("--name", required=True, help="person: Фамилия Имя Отчество")
    s.add_argument("--id", help="БИН / ИИН (12 digits)")
    s.add_argument("--birth-date", help="YYYY-MM-DD (person)")
    s.add_argument("--city")
    s.add_argument("--alias", action="append", help="other names, repeatable")
    s.add_argument("--related", action="append", help="director / founder / parent company, repeatable")
    s.add_argument("--purpose", required=True, help="why this check is done (logged)")
    s.add_argument("--operator", default=getpass.getuser())
    s.add_argument("--no-deep", action="store_true", help="registries only, no web search")
    s.add_argument("--rounds", type=int, help=f"deep search rounds (default {settings.MAX_ROUNDS})")
    s.add_argument("--out", help=f"report directory (default {settings.REPORTS_DIR})")
    s.set_defaults(func=_screen)

    d = sub.add_parser("demo", help="replay a recorded fictional screening (no API key, no network)")
    d.add_argument("--out", help=f"directory for sample_report.md / .json (default {settings.REPORTS_DIR})")
    d.set_defaults(func=_demo)

    u = sub.add_parser("update-sanctions", help="download the OpenSanctions export")
    u.set_defaults(func=_update_sanctions)

    lb = sub.add_parser("label", help="record a reviewer's identity decision for a finding (training data)")
    lb.add_argument("finding_id", help="the `id` shown next to each source in the report")
    lb.add_argument("decision", choices=["same", "namesake"], help="same entity as the subject, or a namesake")
    lb.add_argument("--request-id", help="limit to one screening (a finding id can recur across screenings)")
    lb.add_argument("--reviewer", default=getpass.getuser())
    lb.set_defaults(func=_label)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
