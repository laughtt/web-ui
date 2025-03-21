import pdb

import pyperclip
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from main_content_extractor import MainContentExtractor
from browser_use.controller.views import (
    ClickElementAction,
    DoneAction,
    ExtractPageContentAction,
    GoToUrlAction,
    InputTextAction,
    OpenTabAction,
    ScrollAction,
    SearchGoogleAction,
    SendKeysAction,
    SwitchTabAction,
)
import logging
from src.utils.tools import scan_url_with_jina

logger = logging.getLogger(__name__)


class CustomController(Controller):
    def __init__(self, exclude_actions: list[str] = [],
                 output_model: Optional[Type[BaseModel]] = None
                 ):
        super().__init__(exclude_actions=exclude_actions, output_model=output_model)
        self._register_custom_actions()

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard")
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            # send text to browser
            page = await browser.get_current_page()
            await page.keyboard.type(text)

            return ActionResult(extracted_content=text)

        # Register the Jina scan URL action
        @self.registry.register(
            name="scan_url_with_jina",
            description="Scans a URL using Jina AI to extract and summarize its content. Useful for quickly understanding webpage content without navigating to it.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "The URL to scan (e.g., 'https://example.com')"
                },
                "api_key": {
                    "type": "string",
                    "description": "Optional: Custom Jina API key to use",
                    "required": False
                }
            },
            returns={
                "type": "string",
                "description": "The extracted and summarized content from the URL in markdown format"
            },
            include_in_memory=True
        )
        async def scan_url_with_jina_action(
            url: str,
            api_key: Optional[str] = None,
            **kwargs
        ) -> Dict[str, Any]:
            """Scan a URL using Jina AI and return the extracted content."""
            result = scan_url_with_jina(url, api_key)
            return {
                "extracted_content": result,
                "include_in_memory": True,
                "error": None,
                "is_done": False
            }
