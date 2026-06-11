"""CLI entrypoint for VESTA."""
import sys

from vesta.runtime import _pytensor_compiledir, _thread_caps  # noqa: F401


def main() -> None:
    from vesta.core.experiment_config import ExperimentConfig
    from vesta.core.experiments import run_all

    config = ExperimentConfig(_cli_parse_args=True)

    from vesta.core.logging_utils import format_log_block
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    run_all(config=config)


if __name__ == "__main__":
    main()
