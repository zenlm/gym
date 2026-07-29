"""Unified CLI: python -m gym.distributed {coordinator|worker|self-test}.

Pure dispatcher. Each subcommand defers to a module that owns the concern.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gym.distributed",
                                description="zen lab federation")
    sub = p.add_subparsers(dest="cmd", required=True)

    co = sub.add_parser("coordinator", help="run the federation coordinator")
    co.add_argument("--lab", required=True)
    co.add_argument("--bind", default="0.0.0.0:8443")
    co.add_argument("--log-level", default="INFO")

    wo = sub.add_parser("worker", help="run a worker in dev mode (random data)")
    wo.add_argument("--lab", required=True)
    wo.add_argument("--as", dest="name", required=True)
    wo.add_argument("--coordinator", required=True)
    wo.add_argument("--rounds", type=int, default=10)
    wo.add_argument("--log-level", default="INFO")

    st = sub.add_parser("self-test", help="probe this box + optional coordinator")
    st.add_argument("--coordinator")
    st.add_argument("--as", dest="name")
    st.add_argument("--log-level", default="WARNING")

    args = p.parse_args(argv)
    logging.basicConfig(level=getattr(args, "log_level", "INFO"),
                        format="%(asctime)s [%(levelname)s] %(name)s %(message)s")

    if args.cmd == "coordinator":
        from .coordinator import main as coord_main
        return coord_main(["--lab", args.lab, "--bind", args.bind,
                           "--log-level", args.log_level]) or 0

    if args.cmd == "worker":
        from .dev_worker import run as run_dev
        from .topology import Lab
        run_dev(lab=Lab.from_yaml(args.lab), name=args.name,
                coordinator_url=args.coordinator, total_rounds=args.rounds)
        return 0

    if args.cmd == "self-test":
        from .selftest import selftest
        return selftest(args.coordinator, args.name)

    return 1


if __name__ == "__main__":
    sys.exit(main())
