from __future__ import annotations

import argparse

from gesturedesk.app import GestureDeskApp
from gesturedesk.config import load_config
from gesturedesk.diagnostics import run_preflight_checks
from gesturedesk.logging_setup import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="GestureDesk local runner")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run preflight checks (model/camera/display) and exit",
    )
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    if args.check:
        results = run_preflight_checks(config)
        has_error = False
        for item in results:
            prefix = "OK" if item.ok else "ERR"
            print(f"[{prefix}] {item.message}")
            if not item.ok:
                has_error = True
        return 1 if has_error else 0

    app = GestureDeskApp(config=config, logger=logger, config_path=args.config)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
