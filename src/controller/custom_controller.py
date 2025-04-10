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
from src.utils.mongodb import db
import logging
from src.utils.tools import scan_url_with_jina, screenshot_url_analysis
from src.utils.shell import ShellTools
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
        self.shell_tools = ShellTools()
        self.mongodb = db

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
                url = self.s3_handler.file_write(file_path, content, append)
                return ActionResult(extracted_content=f"Successfully wrote to file, final url: {url}", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))
        
        @self.registry.action('Find file in by name')
        def find_file(file_name: str, subfolder: Optional[str] = None):
            try:
                files = self.s3_handler.find_file_by_name(file_name)
                return ActionResult(extracted_content=f"Found files: {', '.join(files)}" , include_in_memory=True)
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
                url = self.s3_handler.upload_local_file_to_s3(file_path)
                return ActionResult(extracted_content=f"Successfully made {url} public", include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))
        
        @self.registry.action('Run shell command')
        def run_shell_command(command: str):
            try:
                result = self.shell_tools.shell_exec(command)
                return ActionResult(extracted_content=result, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))
        
        @self.registry.action('Start interactive shell process')
        def start_shell_process(command: str, process_id: Optional[str] = None, cwd: Optional[str] = None):
            try:
                result = self.shell_tools.shell_exec(command, process_id=process_id, cwd=cwd, interactive=True)
                return ActionResult(extracted_content=result, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))

        @self.registry.action('View shell process output')
        def view_shell_process(process_id: str, timeout: float = 0.1):
            try:
                result = self.shell_tools.shell_view(process_id, timeout)
                return ActionResult(extracted_content=result, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))

        @self.registry.action('Wait for shell process to complete')
        def wait_for_shell_process(process_id: str, timeout: Optional[float] = None, check_interval: float = 0.5):
            try:
                result = self.shell_tools.shell_wait(process_id, timeout, check_interval)
                return ActionResult(extracted_content=result, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))

        @self.registry.action('Send input to shell process')
        def write_to_shell_process(process_id: str, input_text: str):
            try:
                result = self.shell_tools.shell_write_to_process(process_id, input_text)
                return ActionResult(extracted_content=result, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))

        @self.registry.action('Terminate shell process')
        def kill_shell_process(process_id: str, force: bool = False):
            try:
                result = self.shell_tools.shell_kill_process(process_id, force)
                return ActionResult(extracted_content=result, include_in_memory=True)
            except Exception as e:
                return ActionResult(error=str(e))
        
        @self.registry.action('save chat user context')
        async def save_chat_user_context(browser: BrowserContext):
            """save user chat information context from the brownser"""
            screenshot = await browser.take_screenshot()
            # save screenshot to mongodb
            analysis = await screenshot_url_analysis(screenshot)
            if analysis is None:
                return ActionResult(error="No chat context found", include_in_memory=True)
            await self.mongodb.save_user_chat_context(person_name=analysis.name_user_chat, screenshot=str(screenshot), context=analysis.context_user_chat)
            logger.debug(f"Successfully saved chat user context: {analysis.context_user_chat}")
            return ActionResult(extracted_content=f"Successfully saved chat user context: {analysis.context_user_chat}", include_in_memory=True)
            
        
        @self.registry.action('Get user chat context')
        async def get_user_chat_context(browser: BrowserContext):
            """Get user chat context from the mongodb"""
            screenshot = await browser.take_screenshot()
            analysis = await screenshot_url_analysis(screenshot)
            if analysis is None:
                return ActionResult(error="No chat context found", include_in_memory=True)
            context = await self.mongodb.get_recent_chat_contexts(person_name=analysis.name_user_chat)
            msg = ""
            for c in context:
                msg = msg + f'{str(c["messages"])}\n' 
            return ActionResult(extracted_content=msg, include_in_memory=True)
