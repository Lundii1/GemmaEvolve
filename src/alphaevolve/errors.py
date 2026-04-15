"""Package-specific errors."""


class AlphaEvolveError(Exception):
    """Base class for package-level errors."""


class ConfigError(AlphaEvolveError):
    """Raised when configuration loading or validation fails."""


class PromptTooLargeError(AlphaEvolveError):
    """Raised when a prompt exceeds the configured prompt budget."""


class DiffParseError(AlphaEvolveError):
    """Raised when a model response cannot be parsed into diff blocks."""


class DiffApplyError(AlphaEvolveError):
    """Raised when a diff cannot be applied safely."""


class CapabilityUnavailableError(AlphaEvolveError):
    """Raised when an optional runtime dependency is unavailable."""
