import asyncio
from typing import Any, Dict, Optional
from fastapi import WebSocket
from src.utils.default_config_settings import default_config
from src.utils import utils
from .logging import AsyncWebSocketHandler
from .models import MessageType

class AgentManager:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.ws_handler = AsyncWebSocketHandler(websocket)
        self.agent = None
        self.browser = None
        self.browser_context = None
        self._stop_requested = False

    async def initialize_agent(self, task: str, add_infos: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent with given configuration"""
        try:
            # Merge default config with provided config
            agent_config = default_config()
            if config:
                agent_config.update(config)

            await self.ws_handler.send_update(
                MessageType.STATUS,
                {"status": "initializing", "message": "Initializing agent..."}
            )

            # Get LLM model
            llm = utils.get_llm_model(
                provider=agent_config['llm_provider'],
                model_name=agent_config['llm_model_name'],
                num_ctx=agent_config['llm_num_ctx'],
                temperature=agent_config['llm_temperature'],
                base_url=agent_config['llm_base_url'],
                api_key=agent_config['llm_api_key'],
            )

            # Initialize browser and context
            from browser_use.browser.browser import Browser, BrowserConfig
            from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize
            from src.browser.custom_browser import CustomBrowser

            self.browser = Browser(
                config=BrowserConfig(
                    headless=agent_config['headless'],
                    disable_security=agent_config['disable_security'],
                    extra_chromium_args=[f"--window-size={agent_config['window_w']},{agent_config['window_h']}"]
                )
            )

            self.browser_context = await self.browser.new_context(
                config=BrowserContextConfig(
                    trace_path=agent_config['save_trace_path'],
                    save_recording_path=agent_config['save_recording_path'] if agent_config['enable_recording'] else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=agent_config['window_w'],
                        height=agent_config['window_h']
                    ),
                )
            )

            # Initialize agent
            from src.agent.custom_agent import CustomAgent
            from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt

            self.agent = CustomAgent(
                task=task,
                add_infos=add_infos or "",
                llm=llm,
                browser=self.browser,
                browser_context=self.browser_context,
                use_vision=agent_config['use_vision'],
                max_actions_per_step=agent_config['max_actions_per_step'],
                tool_calling_method=agent_config['tool_calling_method']
            )

            await self.ws_handler.send_update(
                MessageType.STATUS,
                {"status": "initialized", "message": "Agent initialized successfully"}
            )

        except Exception as e:
            await self.ws_handler.send_error(
                error="Failed to initialize agent",
                details={"error": str(e)}
            )
            raise

    async def run_task(self, max_steps: int = 100):
        """Run the agent task"""
        try:
            if not self.agent:
                raise ValueError("Agent not initialized")

            await self.ws_handler.send_update(
                MessageType.STATUS,
                {"status": "running", "message": "Starting task execution"}
            )

            history = await self.agent.run(max_steps=max_steps)

            # Send final results
            await self.ws_handler.send_update(
                MessageType.RESULT,
                {
                    "final_result": history.final_result(),
                    "errors": history.errors(),
                    "model_actions": history.model_actions(),
                    "model_thoughts": history.model_thoughts()
                }
            )

        except Exception as e:
            await self.ws_handler.send_error(
                error="Task execution failed",
                details={"error": str(e)}
            )
            raise
        finally:
            await self.cleanup()

    async def stop_task(self):
        """Request the agent to stop"""
        self._stop_requested = True
        if self.agent:
            self.agent.stop()
        await self.ws_handler.send_update(
            MessageType.STATUS,
            {"status": "stopping", "message": "Stop requested"}
        )

    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.browser_context:
                await self.browser_context.close()
            if self.browser:
                await self.browser.close()
        except Exception as e:
            await self.ws_handler.send_error(
                error="Cleanup failed",
                details={"error": str(e)}
            ) 