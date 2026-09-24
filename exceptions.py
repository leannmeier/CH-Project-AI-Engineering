from typing import Optional
from Config import ClientType

class LLMClientError(Exception):
    """Excepción base personalizada para errores generados por proveedores de LLM."""

    def __init__(
        self,
        message: str,
        provider: ClientType,
        status_code: Optional[int] = None,
        raw_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.status_code = status_code
        self.raw_error = raw_error

    def __str__(self) -> str:
        code_str = f" (Status Code: {self.status_code})" if self.status_code else ""
        return f"[{self.provider.value} ERROR]{code_str}: {self.message}"