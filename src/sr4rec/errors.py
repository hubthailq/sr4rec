"""Exception types and process exit codes used across SR4Rec."""


class SR4RecError(Exception):
    """Base class for all errors raised on purpose by SR4Rec.

    The command-line interface prints the message without a traceback and exits
    with ``exit_code``.
    """

    exit_code = 1


class ConfigError(SR4RecError):
    """The configuration file is invalid."""


class DatasetError(SR4RecError):
    """The dataset does not follow the required format."""


class SRModelError(SR4RecError):
    """An SR model cannot be loaded or does not follow the interface."""


class ReproductionMismatch(SR4RecError):
    """A reproduced metric falls outside the published tolerance."""

    exit_code = 2


class MissingResource(SR4RecError):
    """Data, weights, checkpoints or expected results needed by a command are missing."""

    exit_code = 3
