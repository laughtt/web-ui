import pdb

import pyperclip
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel
from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller, DoneAction
from main_content_extractor import MainContentExtractor
from src.utils.s3 import S3FileHandler
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
        self.s3_handler = S3FileHandler(
            bucket_name="test-brownser-use",
            prefix="llm-workspace",
            region="us-east-2"
        )

    def _register_custom_actions(self):
        """Register all custom browser actions"""

        @self.registry.action("Copy text to clipboard")
        def copy_to_clipboard(text: str):
            pyperclip.copy(text)
            return ActionResult(extracted_content=text)

        @self.registry.action("Paste text from clipboard")
        async def paste_from_clipboard(browser: BrowserContext):
            text = pyperclip.paste()
            # send text to browserP
            page = await browser.get_current_page()
            await page.keyboard.type(text)

            return ActionResult(extracted_content=text)

        @self.registry.action("Extract information from a webpage using the url, do not use google.com or any other search engine")
        async def extract_content(url: str):
            """Extract information from a webpage using the url, do not use google.com or any other search engine"""
            try:
                if "google.com" in url:
                    return ActionResult(extracted_content="Google.com is not allowed")
                result = scan_url_with_jina(url)
                return ActionResult(extracted_content=result,include_in_memory=True)
            except Exception as e:
                return ActionResult(extracted_content=f"Error: {e}")
            
        @self.registry.action('Read file')
        def file_read(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None):
            try:
                content = self.s3_handler.file_read(file_path, start_line, end_line)
                return ActionResult(extracted_content=content, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))
        
        @self.registry.action('Write to file')
        def file_write(file_path: str, content: str, append: bool = False):
            try:
                self.s3_handler.file_write(file_path, content, append)
                return ActionResult(extracted_content=f"Successfully wrote to {file_path}", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))
        
        @self.registry.action('Find file in by name')
        def find_file(file_name: str, subfolder: Optional[str] = None):
            try:
                files = self.s3_handler.find_file_by_name(file_name)
                return ActionResult(extracted_content=f"Found files: {', '.join(files)}" if files else "No matching files found", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))
        
        @self.registry.action('List files')
        def list_files(subfolder: Optional[str] = None):
            try:
                files = self.s3_handler.list_files()
                return ActionResult(extracted_content=f"Files: {', '.join(files)}" if files else "No files found", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))
            
        @self.registry.action('Upload local file to s3')
        def upload_local_file_to_s3(file_path: str):
            try:
                self.s3_handler.upload_local_file_to_s3(file_path)
                return ActionResult(extracted_content=f"Successfully made {file_path} public", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))