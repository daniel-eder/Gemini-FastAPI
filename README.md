# Gemini-FastAPI

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[ English | [中文](README.zh.md) ]

Web-based Gemini models wrapped into an OpenAI-compatible API. Powered by  [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API).

Note: this repository is configured to use a fork of the original client that adds streaming support: https://github.com/hvjg2578/Gemini-API. The fork is installed directly from GitHub so the code continues to import the package as `gemini_webapi`.

**✅ Call Gemini's web-based models via API without an API Key, completely free!**

## Features

- **🔐 No Google API Key Required**: Use web cookies to freely access Gemini's models via API.
- **🔍 Google Search Included**: Get up-to-date answers using web-based Gemini's search capabilities.
- **💾 Conversation Persistence**: LMDB-based storage supporting multi-turn conversations.
- **🖼️ Multi-modal Support**: Support for handling text, images, and file uploads.
- **🔧 Flexible Configuration**: YAML-based configuration with environment variable overrides.

## Quick Start

**For Docker deployment, see the [Docker Deployment](#docker-deployment) section below.**

### Prerequisites

- Python 3.12
- Google account with Gemini access on web
- `secure_1psid` and `secure_1psidts` cookies from Gemini web interface

### Installation

#### Using uv (Recommended)

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

#### Using pip

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
pip install -e .
```

### Configuration

Edit `config/config.yaml` and provide at least one credential pair:
```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
```

> [!NOTE]
> For details, refer to the [Configuration](#configuration-1) section below.

### Running the Server

```bash
# Using uv
uv run python run.py

# Using Python directly
python run.py
```

The server will start on `http://localhost:8000` by default.

### Enhanced Logging & Debugging

You can now control logging more granularly:

| Environment Variable | Values | Description |
|----------------------|--------|-------------|
| `GEMINI_LOG_LEVEL`   | DEBUG, INFO, WARNING, ERROR, CRITICAL | Override configured log level without editing YAML |
| `GEMINI_LOG_JSON`    | true/false (1/0) | Emit structured JSON logs (for ingestion into log systems) |

Example (PowerShell):

```powershell
$env:GEMINI_LOG_LEVEL = 'DEBUG'
$env:GEMINI_LOG_JSON = 'true'
uv run python run.py
```

Streaming-specific instrumentation has been added. Each request is tagged with a unique `request_id` (e.g. `req-<uuid>`). Look for these fields to correlate logs:

- Request summary: input length, first user message sample
- Session reuse decision (reused vs new)
- Chunk splitting details (`Sending X chunk(s)` ...)
- Intermediate streaming progress samples (every chunk up to 10 then every 25th)
- Final streaming metrics (total chunks, chars, elapsed ms) and token usage
- Warning if the upstream produced **zero streaming chunks** (`Streaming ended with zero chunks ...`)

If you experience a 200 response with no tokens received from the client:

1. Check for the zero-chunk warning in logs
2. Ensure `GEMINI_LOG_LEVEL=DEBUG` to see chunk-splitting and upstream send events
3. Verify cookies are still valid (look for auto-refresh logs from `gemini_webapi` if `verbose` is enabled in config)
4. Confirm the first message actually contains user content (sample logged)
5. Consider lowering `max_chars_per_request` in `config/config.yaml` if very large prompts stall upstream

To narrow logs to a single request, filter on its `request_id`.

## Docker Deployment

### Run with Options

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/.venv/lib/python3.12/site-packages/gemini_webapi/utils/temp \
  -e CONFIG_SERVER__API_KEY="your-api-key-here" \
  -e CONFIG_GEMINI__CLIENTS__0__ID="client-a" \
  -e CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid" \
  -e CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts" \
  ghcr.io/nativu5/gemini-fastapi
```

### Run with Docker Compose

Create a `docker-compose.yml` file:

```yaml
services:
  gemini-fastapi:
    image: ghcr.io/nativu5/gemini-fastapi:latest
    ports:
      - "8000:8000"
    volumes:
      # - ./config:/app/config  # Uncomment to use a custom config file
      # - ./certs:/app/certs        # Uncomment to enable HTTPS with your certs
      - ./data:/app/data
      - ./cache:/app/.venv/lib/python3.12/site-packages/gemini_webapi/utils/temp
    environment:
      - CONFIG_SERVER__HOST=0.0.0.0
      - CONFIG_SERVER__PORT=8000
      - CONFIG_SERVER__API_KEY=${API_KEY}
      - CONFIG_GEMINI__CLIENTS__0__ID=client-a
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID=${SECURE_1PSID}
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS=${SECURE_1PSIDTS}
    restart: on-failure:3 # Avoid retrying too many times
```

Then run:

```bash
docker compose up -d
```

> [!IMPORTANT]
> Make sure to mount the `/app/data` volume to persist conversation data between container restarts.
> It's also recommended to mount the `gemini_webapi/utils/temp` directory to save refreshed cookies.

## Configuration

The server reads a YAML configuration file located at `config/config.yaml`. 

For details on each configuration option, refer to the comments in the [`config/config.yaml`](https://github.com/Nativu5/Gemini-FastAPI/blob/main/config/config.yaml) file.

### Environment Variable Overrides

> [!TIP]
> This feature is particularly useful for Docker deployments and production environments where you want to keep sensitive credentials separate from configuration files. 

You can override any configuration option using environment variables with the `CONFIG_` prefix. Use double underscores (`__`) to represent nested keys, for example:

```bash
# Override server settings
export CONFIG_SERVER__API_KEY="your-secure-api-key"

# Override Gemini credentials (first client)
export CONFIG_GEMINI__CLIENTS__0__ID="client-a"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts"

# Override conversation storage size limit
export CONFIG_STORAGE__MAX_SIZE=268435456  # 256 MB
```

### Client IDs and Conversation Reuse

Conversations are stored with the ID of the client that generated them.
Keep these identifiers stable in your configuration so that sessions remain valid
when you update the cookie list.

### Gemini Credentials

> [!WARNING]
> Keep these credentials secure and never commit them to version control. These cookies provide access to your Google account.

To use Gemini-FastAPI, you need to extract your Gemini session cookies:

1. Open [Gemini](https://gemini.google.com/) in a private/incognito browser window and sign in
2. Open Developer Tools (F12)
3. Navigate to **Application** → **Storage** → **Cookies**
4. Find and copy the values for:
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

> [!TIP]
> For detailed instructions, refer to the [HanaokaYuzu/Gemini-API authentication guide](https://github.com/HanaokaYuzu/Gemini-API?tab=readme-ov-file#authentication).

## Acknowledgments

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - The underlying Gemini web API client
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - This project originated from this repository. After extensive refactoring and engineering improvements, it has evolved into an independent project, featuring multi-turn conversation reuse among other enhancements. Special thanks for the inspiration and foundational work provided.

## Disclaimer

This project is not affiliated with Google or OpenAI and is intended solely for educational and research purposes. It uses reverse-engineered APIs and may not comply with Google's Terms of Service. Use at your own risk.
