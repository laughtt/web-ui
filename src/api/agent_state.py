from fastapi import FastAPI, WebSocket
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from src.utils.agent_state import AgentState
from typing import Optional, Dict
from dataclasses import dataclass
from contextlib import asynccontextmanager
from .logging import AsyncWebSocketHandler

@dataclass
class GlobalState:
    browser: Optional[Browser] = None
    browser_context: Optional[any] = None
    agent: Optional[any] = None
    default_llm: Optional[any] = None  # Add default_llm field
    agent_state: AgentState = AgentState()
    active_connections: Dict[str, AsyncWebSocketHandler] = None

    def __post_init__(self):
        self.active_connections = {}

class StateManager:
    def __init__(self):
        self._state = GlobalState()

    @property
    def state(self) -> GlobalState:
        return self._state

    async def register_connection(self, connection_id: str, websocket: WebSocket) -> AsyncWebSocketHandler:
        """Register a new WebSocket connection"""
        try:
            # Remove any existing connection with the same ID
            await self.remove_connection(connection_id)
            
            # Create new handler
            ws_handler = AsyncWebSocketHandler(websocket)
            self._state.active_connections[connection_id] = ws_handler
            return ws_handler
        except Exception as e:
            print(f"Error registering connection: {e}")
            raise

    async def remove_connection(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self._state.active_connections:
            try:
                handler = self._state.active_connections[connection_id]
                await handler.close()
            except:
                pass
            finally:
                del self._state.active_connections[connection_id]

    async def initialize_browser(self, config: BrowserConfig):
        """Initialize browser if not already initialized"""
        if self._state.browser is None:
            self._state.browser = Browser(config=config)
        return self._state.browser

    async def initialize_context(self, config: BrowserContextConfig):
        """Initialize browser context if not already initialized"""
        if self._state.browser_context is None:
            self._state.browser_context = await self._state.browser.new_context(config=config)
        return self._state.browser_context

    async def cleanup(self, keep_browser_open: bool = False):
        """Cleanup resources"""
        # Close all active connections
        for connection_id in list(self._state.active_connections.keys()):
            await self.remove_connection(connection_id)
            
        self._state.agent = None
        if not keep_browser_open:
            if self._state.browser_context:
                await self._state.browser_context.close()
                self._state.browser_context = None
            if self._state.browser:
                await self._state.browser.close()
                self._state.browser = None

    def clear_stop(self):
        """Clear stop request"""
        self._state.agent_state.clear_stop()

    def request_stop(self):
        """Request agent to stop"""
        self._state.agent_state.request_stop()

# Create global state manager
state_manager = StateManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize any resources
    yield
    # Shutdown: Cleanup
    await state_manager.cleanup() 