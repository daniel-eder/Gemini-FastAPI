import os
import sys

import uvicorn
from loguru import logger

from app.main import create_app
from app.utils import g_config, setup_logging

app = create_app()

if __name__ == "__main__":
    # Setup loguru logging (env overrides: GEMINI_LOG_LEVEL, GEMINI_LOG_JSON)
    json_logging = os.getenv("GEMINI_LOG_JSON", "false").lower() in {"1", "true", "yes"}
    setup_logging(level=g_config.logging.level, json=json_logging)

    # Determine reload precedence: env overrides config
    env_reload = os.getenv("GEMINI_RELOAD")
    if env_reload is not None:
        reload_enabled = env_reload.lower() in {"1", "true", "yes"}
    else:
        reload_enabled = bool(getattr(g_config.server, "reload", False))
    if reload_enabled:
        logger.info("Hot reload enabled (GEMINI_RELOAD=true). Uvicorn will watch for file changes.")

    # Check HTTPS configuration
    if g_config.server.https.enabled:
        key_path = g_config.server.https.key_file
        cert_path = g_config.server.https.cert_file

        # Check if the certificate files exist
        if not os.path.exists(key_path) or not os.path.exists(cert_path):
            logger.critical(f"HTTPS enabled but SSL certificate files not found: {key_path}, {cert_path}")
            sys.exit(1)

        logger.info(f"Starting server at https://{g_config.server.host}:{g_config.server.port} ...")
        uvicorn.run(
            app,
            host=g_config.server.host,
            port=g_config.server.port,
            log_config=None,
            ssl_keyfile=key_path,
            ssl_certfile=cert_path,
        )
    else:
        logger.info(f"Starting server at http://{g_config.server.host}:{g_config.server.port} ...")
        uvicorn.run(
            app,
            host=g_config.server.host,
            port=g_config.server.port,
            log_config=None,
            reload=reload_enabled,
        )
