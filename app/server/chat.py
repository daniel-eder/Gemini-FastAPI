import uuid
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import orjson
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from loguru import logger

from ..models import (
    ChatCompletionRequest,
    ConversationInStore,
    Message,
    ModelData,
    ModelListResponse,
)
from ..services import (
    GeminiClientPool,
    GeminiClientWrapper,
    LMDBConversationStore,
)
from ..utils import g_config
from ..utils.helper import estimate_tokens
from .middleware import get_temp_dir, verify_api_key

# Maximum characters Gemini Web can accept in a single request (configurable)
MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)

CONTINUATION_HINT = "\n(More messages to come, please reply with just 'ok.')"


router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    now = int(datetime.now(tz=timezone.utc).timestamp())

    models = []
    for model in Model:
        m_name = model.model_name
        if not m_name or m_name == "unspecified":
            continue

        models.append(
            ModelData(
                id=m_name,
                created=now,
                owned_by="gemini-web",
            )
        )

    return ModelListResponse(data=models)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    # Correlate all logs for this request with a unique id
    request_id = f"req-{uuid.uuid4()}"
    req_logger = logger.bind(request_id=request_id)
    start_time = time.perf_counter()

    pool = GeminiClientPool()
    db = LMDBConversationStore()
    model = Model.from_name(request.model)

    if len(request.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required in the conversation.",
        )

    # Log basic request summary (sanitized / truncated)
    try:
        first_user_msg = next((m for m in request.messages if m.role == "user"), None)
        sample_text = ""
        if first_user_msg:
            if isinstance(first_user_msg.content, str):
                sample_text = first_user_msg.content[:200]
            else:
                # collect first text fragment
                for part in first_user_msg.content:
                    if getattr(part, "type", "") == "text" and part.text:
                        sample_text = part.text[:200]
                        break
        req_logger.debug(
            "Incoming chat completion request",  # message
            extra={
                "model": request.model,
                "stream": request.stream,
                "messages_count": len(request.messages),
                "first_user_sample": sample_text,
            },
        )
    except Exception as e:
        req_logger.warning(f"Failed to log request summary: {e}")

    # Check if conversation is reusable
    session, client, remaining_messages = _find_reusable_session(db, pool, model, request.messages)

    if session:
        req_logger.debug(
            f"Reusable session found; remaining_messages={len(remaining_messages)} metadata={session.metadata}"
        )
    else:
        req_logger.debug("No reusable session found; starting new conversation")

    if session:
        # Prepare the model input depending on how many turns are missing.
        if len(remaining_messages) == 1:
            model_input, files = await GeminiClientWrapper.process_message(
                remaining_messages[0], tmp_dir, tagged=False
            )
        else:
            model_input, files = await GeminiClientWrapper.process_conversation(
                remaining_messages, tmp_dir
            )
        req_logger.debug(
            f"Reused session {session.metadata} - sending {len(remaining_messages)} new messages."
        )
    else:
        # Start a new session and concat messages into a single string
        try:
            client = pool.acquire()
            session = client.start_chat(model=model)
            model_input, files = await GeminiClientWrapper.process_conversation(
                request.messages, tmp_dir
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in preparing conversation: {e}")
            raise
    req_logger.debug("New session started.")

    # Generate response
    try:
        assert session and client, "Session and client not available"
        req_logger.debug(
            f"Prepared model input (length={len(model_input)} chars, files={len(files)})"
        )

        completion_id = f"chatcmpl-{uuid.uuid4()}"
        timestamp = int(datetime.now(tz=timezone.utc).timestamp())

        if request.stream:
            # For streaming, get the stream iterator
            stream_iterator = await _send_with_split(
                session, model_input, files=files, stream=True, request_id=request_id
            )
            return _create_streaming_response(
                stream_iterator,
                completion_id,
                timestamp,
                request.model,
                request.messages,
                db,
                session,
                client,
                request_id,
                model_input,
                files,
            )
        else:
            # For non-streaming, get the full response
            response = await _send_with_split(
                session, model_input, files=files, stream=False, request_id=request_id
            )

            # Format the response from API
            model_output = GeminiClientWrapper.extract_output(response, include_thoughts=False)
            thoughts = response.thoughts if hasattr(response, 'thoughts') else None

            # After formatting, persist the conversation to LMDB
            try:
                last_message = Message(role="assistant", content=model_output)
                cleaned_history = db.sanitize_assistant_messages(request.messages)

                # Defensive: ensure model is a valid string. Use model.model_name
                # if available, otherwise fall back to the original request.model.
                model_name_to_store = model.model_name if getattr(model, "model_name", None) else request.model
                conv = ConversationInStore(
                    model=str(model_name_to_store),
                    client_id=client.id,
                    metadata=session.metadata,
                    messages=[*cleaned_history, last_message],
                )
                key = db.store(conv)
                logger.debug(f"Conversation saved to LMDB with key: {key}")
            except Exception as e:
                # We can still return the response even if saving fails
                req_logger.warning(f"Failed to save conversation to LMDB: {e}")

            elapsed = (time.perf_counter() - start_time) * 1000
            req_logger.debug(
                f"Non-streaming response ready (elapsed={elapsed:.1f}ms, tokens_estimate={estimate_tokens(model_output)})"
            )
            return _create_standard_response(
                model_output, thoughts, completion_id, timestamp, request.model, request.messages
            )

    except Exception as e:
        req_logger.exception(f"Error generating content from Gemini API: {e}")
        raise


def _text_from_message(message: Message) -> str:
    """Return text content from a message for token estimation."""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        item.text or "" for item in message.content if getattr(item, "type", "") == "text"
    )


def _find_reusable_session(
    db: LMDBConversationStore,
    pool: GeminiClientPool,
    model: Model,
    messages: list[Message],
) -> tuple[ChatSession | None, GeminiClientWrapper | None, list[Message]]:
    """Find an existing chat session that matches the *longest* prefix of
    ``messages`` **whose last element is an assistant/system reply**.

    Rationale
    ---------
    When a reply was generated by *another* server instance, the local LMDB may
    only contain an older part of the conversation.  However, as long as we can
    line-up **any** earlier assistant/system response, we can restore the
    corresponding Gemini session and replay the *remaining* turns locally
    (including that missing assistant reply and the subsequent user prompts).

    The algorithm therefore walks backwards through the history **one message at
    a time**, each time requiring the current tail to be assistant/system before
    querying LMDB.  As soon as a match is found we recreate the session and
    return the untouched suffix as ``remaining_messages``.
    """

    if len(messages) < 2:
        return None, None, messages

    # Start with the full history and iteratively trim from the end.
    search_end = len(messages)
    while search_end >= 2:
        search_history = messages[:search_end]

        # Only try to match if the last stored message would be assistant/system.
        if search_history[-1].role in {"assistant", "system"}:
            try:
                if conv := db.find(model.model_name, search_history):
                    client = pool.acquire(conv.client_id)
                    session = client.start_chat(metadata=conv.metadata, model=model)
                    remain = messages[search_end:]
                    return session, client, remain
            except Exception as e:
                logger.warning(f"Error checking LMDB for reusable session: {e}")
                break

        # Trim one message and try again.
        search_end -= 1

    return None, None, messages


async def _send_with_split(
    session: ChatSession,
    text: str,
    files: list[Path | str] | None = None,
    stream: bool = False,
    request_id: str | None = None,
):
    """Send text to Gemini, automatically splitting into multiple batches if it is
    longer than ``MAX_CHARS_PER_REQUEST``.

    Every intermediate batch (that is **not** the last one) is suffixed with a hint
    telling Gemini that more content will come, and it should simply reply with
    "ok". The final batch carries any file uploads and the real user prompt so
    that Gemini can produce the actual answer.

    Args:
        session: The ChatSession to send messages to
        text: The text to send
        files: Optional list of files to attach
        stream: If True, returns a stream iterator; if False, returns ModelOutput

    Returns:
        Either a ModelOutput (stream=False) or an async iterator of StreamChunk (stream=True)
    """
    bound_logger = logger.bind(request_id=request_id) if request_id else logger

    if len(text) <= MAX_CHARS_PER_REQUEST:
        # No need to split - a single request is fine.
        bound_logger.debug(
            f"Sending single request to Gemini (length={len(text)} <= max={MAX_CHARS_PER_REQUEST}, stream={stream})"
        )
        if stream:
            return await session.send_message_stream(text, files=files)
        else:
            return await session.send_message(text, files=files)

    hint_len = len(CONTINUATION_HINT)
    chunk_size = MAX_CHARS_PER_REQUEST - hint_len

    chunks: list[str] = []
    pos = 0
    total = len(text)
    while pos < total:
        end = min(pos + chunk_size, total)
        chunk = text[pos:end]
        pos = end

        # If this is NOT the last chunk, add the continuation hint.
        if end < total:
            chunk += CONTINUATION_HINT
        chunks.append(chunk)

    # Fire off all but the last chunk, discarding the interim "ok" replies.
    bound_logger.debug(
        f"Sending {len(chunks)} chunk(s) to Gemini (max_chars_per_request={MAX_CHARS_PER_REQUEST}, stream={stream})"
    )
    for i, chk in enumerate(chunks[:-1]):
        try:
            bound_logger.debug(
                f"Sending intermediate chunk {i+1}/{len(chunks)-1} (len={len(chk)})"
            )
            await session.send_message(chk)
        except Exception as e:
            bound_logger.exception(f"Error sending chunk {i+1} to Gemini: {e}")
            raise

    # The last chunk carries the files (if any) and we return its response.
    bound_logger.debug(
        f"Sending final chunk (len={len(chunks[-1])}) with files={len(files) if files else 0}" + (" (stream)" if stream else "")
    )
    if stream:
        return await session.send_message_stream(chunks[-1], files=files)
    else:
        return await session.send_message(chunks[-1], files=files)


def _create_streaming_response(
    stream_iterator,
    completion_id: str,
    created_time: int,
    model: str,
    messages: list[Message],
    db: LMDBConversationStore,
    session: ChatSession,
    client: GeminiClientWrapper,
    request_id: str,
    original_model_input: str,
    original_files: list[Path | str] | None,
) -> StreamingResponse:
    """Create streaming response with `usage` calculation included in the final chunk.
    
    Args:
        stream_iterator: Async iterator of StreamChunk objects from gemini webapi
        completion_id: Unique ID for this completion
        created_time: Timestamp for the completion
        model: Model name
        messages: Original messages (for token estimation)
        db: LMDB store for saving conversation
        session: ChatSession for metadata
        client: Client wrapper for ID
    """

    async def generate_stream():
        stream_logger = logger.bind(request_id=request_id)
        stream_start = time.perf_counter()
        stream_logger.debug(
            f"Starting streaming response (completion_id={completion_id}, model={model}, messages={len(messages)})"
        )
        # Send start event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Accumulate the full response for storage and token counting
        full_text = ""
        full_thoughts = ""

        # Stream chunks from Gemini API
        chunk_count = 0
        total_text_chars = 0
        total_thought_chars = 0
        try:
            async for chunk in stream_iterator:
                # Raw chunk debug (safe representation)
                try:
                    stream_logger.debug(
                        "Upstream raw chunk",
                        extra={
                            "has_text": bool(getattr(chunk, 'delta_text', None)),
                            "has_thoughts": bool(getattr(chunk, 'delta_thoughts', None)),
                            "raw_keys": list(chunk.__dict__.keys()),
                        },
                    )
                except Exception:
                    stream_logger.debug("Failed to introspect upstream chunk structure")
                # Build the delta object following OpenAI's format
                delta = {}
                
                # Handle thoughts (reasoning) separately using reasoning_content
                if chunk.delta_thoughts:
                    full_thoughts += chunk.delta_thoughts
                    delta["reasoning_content"] = chunk.delta_thoughts
                    total_thought_chars += len(chunk.delta_thoughts)
                
                # Handle regular text content
                if chunk.delta_text:
                    # Format the text before sending
                    formatted_text = GeminiClientWrapper.format_stream_text(chunk.delta_text)
                    delta["content"] = formatted_text
                    full_text += chunk.delta_text  # Store unformatted for token counting
                    total_text_chars += len(chunk.delta_text)

                # Send the delta if we have any content (thoughts or text)
                if delta:
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
                    chunk_count += 1
                    if chunk_count % 25 == 0 or chunk_count < 10:
                        # Avoid logging every single chunk after many; sample early & every 25th
                        stream_logger.debug(
                            f"Stream chunk {chunk_count}: text_chars={total_text_chars} thought_chars={total_thought_chars}"
                        )

        except Exception as e:
            stream_logger.exception(f"Error during streaming: {e}")
            raise

        if chunk_count == 0:
            stream_logger.warning(
                "Streaming ended with zero chunks received from upstream Gemini API. Check upstream connectivity or request formatting."
            )
            # Determine fallback enablement
            env_fb = os.getenv("GEMINI_STREAM_FALLBACK")
            if env_fb is not None:
                fallback_enabled = env_fb.lower() in {"1", "true", "yes"}
            else:
                fallback_enabled = getattr(g_config.gemini, "streaming_fallback", True)

            if fallback_enabled:
                stream_logger.debug("Attempting non-stream fallback after zero chunks.")
                try:
                    # Perform a non-stream send using the original final input (already split if needed earlier)
                    fallback_response = await session.send_message(original_model_input, files=original_files)
                    # Extract output
                    fb_text = GeminiClientWrapper.extract_output(fallback_response, include_thoughts=False)
                    if fb_text:
                        full_text = fb_text  # replace empty content
                        # Emit as a single synthetic chunk
                        data_fb = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": fb_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {orjson.dumps(data_fb).decode('utf-8')}\n\n"
                        chunk_count = 1
                        total_text_chars = len(fb_text)
                        stream_logger.debug(
                            f"Fallback succeeded; emitted synthetic chunk (chars={total_text_chars})."
                        )
                    else:
                        stream_logger.warning("Fallback response contained no text.")
                except Exception as fe:
                    stream_logger.exception(f"Fallback attempt failed: {fe}")

        # Calculate token usage
        prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
        completion_tokens = estimate_tokens(full_text)
        total_tokens = prompt_tokens + completion_tokens

        # Save conversation to LMDB after streaming completes (only if we have any assistant content)
        try:
            if full_text:
                last_message = Message(role="assistant", content=full_text)
                cleaned_history = db.sanitize_assistant_messages(messages)
                # Ensure the model field is always a valid string. session.metadata may contain None values.
                if session.metadata and len(session.metadata) > 0 and session.metadata[0]:
                    saved_model = session.metadata[0]
                else:
                    saved_model = model
                conv = ConversationInStore(
                    model=str(saved_model),
                    client_id=client.id,
                    metadata=session.metadata,
                    messages=[*cleaned_history, last_message],
                )
                key = db.store(conv)
                stream_logger.debug(f"Conversation saved to LMDB with key: {key}")
            else:
                stream_logger.debug(
                    "Skipped saving conversation because assistant content is empty after streaming/fallback."
                )
        except Exception as e:
            stream_logger.warning(f"Failed to save conversation to LMDB: {e}")

        # Send end event with usage
        elapsed_ms = (time.perf_counter() - stream_start) * 1000
        stream_logger.debug(
            f"Streaming completed (chunks={chunk_count}, text_chars={total_text_chars}, thought_chars={total_thought_chars}, elapsed={elapsed_ms:.1f}ms)"
        )
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_standard_response(
    model_output: str,
    thoughts: str | None,
    completion_id: str,
    created_time: int,
    model: str,
    messages: list[Message],
) -> dict:
    """Create standard response with OpenAI-compatible reasoning_content field"""
    # Calculate token usage
    prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
    completion_tokens = estimate_tokens(model_output)
    total_tokens = prompt_tokens + completion_tokens

    # Build the message with content and optionally reasoning_content
    message = {"role": "assistant", "content": model_output}
    if thoughts:
        message["reasoning_content"] = thoughts

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    logger.debug(f"Response created with {total_tokens} total tokens")
    return result
