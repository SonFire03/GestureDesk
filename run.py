from __future__ import annotations

import argparse

from gesturedesk.config import load_config
from gesturedesk.diagnostics import list_camera_candidates, run_preflight_checks
from gesturedesk.logging_setup import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="GestureDesk local runner")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run preflight checks (model/camera/display) and exit",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available /dev/video* candidates and exit",
    )
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    if args.list_cameras:
        cams = list_camera_candidates()
        if not cams:
            print("Aucune camera detectee sur /dev/video*")
            return 1
        print("Cameras detectees:")
        for cam_id, opened, read_ok in cams:
            print(f"- id={cam_id} opened={opened} read={read_ok}")
        return 0

    if args.check:
        results = run_preflight_checks(config)
        has_error = False
        for item in results:
            prefix = "OK" if item.ok else "ERR"
            print(f"[{prefix}] {item.message}")
            if not item.ok:
                has_error = True
        return 1 if has_error else 0

    from gesturedesk.app import GestureDeskApp

    app = GestureDeskApp(config=config, logger=logger, config_path=args.config)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
