"""Trazas con Arize Phoenix.

Phoenix corre local (no necesita cuenta ni clave): `phoenix serve` o el perfil `observability` del
docker-compose, y se ve en http://localhost:6006. Con `auto_instrument=True` y el paquete
`openinference-instrumentation-langchain` instalado, cada nodo y cada llamada al modelo de
LangGraph/LangChain aparece como una traza con su latencia.

Si Phoenix no está instalado o no responde, la API sigue funcionando sin trazas.
"""

import logging
import os

logger = logging.getLogger("app.observability")


def setup_observability() -> bool:
    """Debe llamarse al arrancar, antes de crear el grafo y de ejecutar cualquier modelo."""
    if os.getenv("ENABLE_TRACING", "1") != "1":
        logger.info("Tracing deshabilitado (ENABLE_TRACING != 1).")
        return False

    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

    try:
        from phoenix.otel import register
    except ImportError:
        logger.warning("arize-phoenix-otel no está instalado: se omite el tracing.")
        return False

    try:
        register(
            project_name=os.getenv("PHOENIX_PROJECT_NAME", "sistema-intelligence"),
            auto_instrument=True,
        )
    except Exception as exc:
        logger.warning("No se pudo iniciar el tracing con Phoenix: %s", exc)
        return False

    logger.info("Tracing con Phoenix activo en %s", os.environ["PHOENIX_COLLECTOR_ENDPOINT"])
    return True
