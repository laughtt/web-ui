import logging
import json
import sys
from typing import Any, Callable, Optional
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect
from .models import AgentUpdate, MessageType


class WebSocketLogHandler(logging.Handler):
    def __init__(self, websocket: WebSocket, level: int = logging.NOTSET):
        super().__init__(level)
        self.websocket = websocket
        self.formatter = logging.Formatter('%(levelname)s - %(message)s')

    async def async_emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            update = AgentUpdate(
                type=MessageType.LOG,
                data={
                    "level": record.levelname,
                    "message": msg
                }
            )
            await self.websocket.send_json(update.model_dump())
        except Exception as e:
            # Log to standard error as fallback
            print(f"Error in WebSocketLogHandler: {e}", file=sys.stderr)


class AsyncWebSocketHandler:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def send_update(self, update_type: MessageType, data: Any) -> None:
        """Send an update through the WebSocket connection"""
        if self._closed:
            return
            
        try:
            update = AgentUpdate(type=update_type, data=data)
            await self.websocket.send_json(update.model_dump())
        except WebSocketDisconnect:
            self._closed = True
        except Exception as e:
            print(f"Error sending WebSocket update: {e}", file=sys.stderr)
            self._closed = True

    async def send_error(self, error: str, details: Optional[dict] = None) -> None:
        """Send an error message through the WebSocket connection"""
        if self._closed:
            return
            
        try:
            update = AgentUpdate(
                type=MessageType.ERROR,
                data={
                    "error": error,
                    "details": details or {}
                }
            )
            await self.websocket.send_json(update.model_dump())
        except WebSocketDisconnect:
            self._closed = True
        except Exception as e:
            print(f"Error sending WebSocket error: {e}", file=sys.stderr)
            self._closed = True

    async def close(self):
        """Close the WebSocket connection"""
        if not self._closed:
            try:
                await self.websocket.close()
            except:
                pass
            finally:
                self._closed = True 