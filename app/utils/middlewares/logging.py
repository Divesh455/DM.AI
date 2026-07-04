"""
Request Logging Middleware.
Intercepts all incoming HTTP requests to trace execution flow, log client IPs,
record latency (process time), and inject unique trace IDs into response headers.
"""
import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that captures request information and tracks processing duration.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()
        
        client_ip = request.client.host if request.client else "unknown"
        
        # Log incoming request metadata
        logger.info(
            f"[{trace_id}] Request started: {request.method} {request.url.path} | Client IP: {client_ip}"
        )
        
        try:
            response = await call_next(request)
        except Exception as exc:
            process_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{trace_id}] Request crashed: {request.method} {request.url.path} | "
                f"Time: {process_time:.2f}ms | Error: {str(exc)}"
            )
            raise exc

        process_time = (time.perf_counter() - start_time) * 1000
        
        # Inject custom trace and processing headers
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
        
        # Log outgoing response details
        logger.info(
            f"[{trace_id}] Request finished: {request.method} {request.url.path} | "
            f"Status: {response.status_code} | Time: {process_time:.2f}ms"
        )
        
        return response
