"""The Brain runtime's closed error types, shared by the runtime and its provider-neutral helpers."""


class RuntimeContractError(ValueError):
    """Trusted orchestration input or persisted output violated the closed contract."""


class ProviderRequestError(RuntimeError):
    """A provider call failed without exposing provider response or credential material."""


class ProviderResponseError(ProviderRequestError):
    """A provider response violated a closed runtime contract without exposing its content."""


class RuntimeStateError(RuntimeError):
    """A checkpoint operation failed without exposing persisted conversation data."""
