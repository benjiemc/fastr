"""fastr - an application to rapidly predict TCR:pMHC structures using AlphaFold 3."""

import argparse
import logging

from fastr._log import add_logging_arguments, setup_logger

parser = argparse.ArgumentParser()

logger = logging.getLogger()

parser = argparse.ArgumentParser(
    prog='fastr',
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--output', '-o', help='path to the output csv file')

add_logging_arguments(parser)


def main() -> None:
    """Run the application."""
    args = parser.parse_args()
    setup_logger(logger, args.log_level, args.log_file)


if __name__ == '__main__':
    main()
