from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
import glob
import asyncio
from typing import Optional, List, Dict, Any, Union
import uvicorn
from dotenv import load_dotenv
import json
import traceback
from datetime import datetime

load_dotenv(override=True)

# Import existing components
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from src.utils.agent_state import AgentState
from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file
from src.utils.utils import get_latest_files, capture_screenshot

# Import the MongoDB module
from src.utils.mongodb import db

# Import the SSH Terminal module
from src.ssh.ssh_terminal import SSHTerminal, SSHConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for persistence (same as in the original code)
_global_browser = None
_global_browser_context = None
_global_agent = None
_global_agent_state = AgentState()

# Create FastAPI app
app = FastAPI(title="Browser Agent API", description="API for controlling browser agents")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request validation
class AgentConfig(BaseModel):
    agent_type: str = "custom"
    llm_provider: str = "openai"  # Default provider
    llm_model_name: str = "gpt-4o"  # Default model
    llm_num_ctx: int = 16384
    llm_temperature: float = 0.7
    llm_base_url: Optional[str] = os.getenv("OPENAI_ENDPOINT", None)  # Use environment value if exists
    llm_api_key: Optional[str]
    use_own_browser: bool = os.getenv("CHROME_PERSISTENT_SESSION", "false").lower() == "true"
    keep_browser_open: bool = os.getenv("CHROME_PERSISTENT_SESSION", "false").lower() == "true"
    headless: bool = True
    disable_security: bool = False
    window_w: int = int(os.getenv("RESOLUTION_WIDTH", "1280"))  # Use environment value if exists
    window_h: int = int(os.getenv("RESOLUTION_HEIGHT", "720"))  # Use environment value if exists
    save_recording_path: Optional[str] = "./tmp/record_videos"
    save_agent_history_path: Optional[str] = "./tmp/agent_history"
    save_trace_path: Optional[str] = "./tmp/traces"
    enable_recording: bool = False
    task: str = "Navigate to example.com and summarize the main content of the page."
    add_infos: Optional[str] = None
    max_steps: int = 100
    use_vision: bool = True
    max_actions_per_step: int = 5
    tool_calling_method: str = "function_calling"
    chrome_cdp: Optional[str] = os.getenv("CHROME_CDP", "http://localhost:9222")  # Use environment value if exists

class ResearchConfig(BaseModel):
    research_task: str = "Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects."
    max_search_iteration_input: int = 3
    max_query_per_iter_input: int = 3
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o"
    llm_num_ctx: int = 16384
    llm_temperature: float = 0.5
    llm_base_url: Optional[str] = os.getenv("OPENAI_ENDPOINT", None)  # Use environment value if exists
    llm_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)  # Use environment value if exists
    use_vision: bool = True
    use_own_browser: bool = os.getenv("CHROME_PERSISTENT_SESSION", "false").lower() == "true"
    headless: bool = True
    chrome_cdp: Optional[str] = os.getenv("CHROME_CDP", "http://localhost:9222")  # Use environment value if exists

# Helper functions (ported from original code)
def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text
        
    import re
    
    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)
    
    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)
        
    return result

# API endpoints
@app.get("/")
async def root():
    return {"status": "running", "message": "Browser Agent API is running"}

@app.post("/run-agent")
async def run_agent(config: AgentConfig, background_tasks: BackgroundTasks):
    """Start a browser agent with the given configuration"""
    task_id = f"task_{os.urandom(4).hex()}"
    config.llm_api_key = os.getenv("OPENAI_API_KEY", None)
    
    # Store task in MongoDB
    await db.store_task(task_id, "agent", config.dict())
    
    background_tasks.add_task(
        run_browser_agent_task,
        task_id=task_id,
        config=config
    )
    
    return {"task_id": task_id, "status": "started"}

@app.get("/agent-status/{task_id}")
async def get_agent_status(task_id: str):
    """Get the status of a running agent task"""
    task = await db.get_task(task_id)
    if task:
        return {
            "task_id": task_id,
            "status": task["status"],
            "created_at": task["created_at"],
            "updated_at": task["updated_at"]
        }
    else:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

@app.post("/stop-agent")
async def stop_agent_endpoint():
    """Request the agent to stop"""
    global _global_agent
    
    try:
        if _global_agent:
            _global_agent.stop()
            return {"status": "stopping", "message": "Stop requested - the agent will halt at the next safe point"}
        else:
            return {"status": "no_agent", "message": "No active agent found"}
    except Exception as e:
        logger.error(f"Error during stop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping agent: {str(e)}")

@app.post("/stop-research")
async def stop_research_endpoint():
    """Request the research agent to stop"""
    global _global_agent_state
    
    try:
        _global_agent_state.request_stop()
        return {"status": "stopping", "message": "Stop requested - the research agent will halt at the next safe point"}
    except Exception as e:
        logger.error(f"Error during stop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stopping research: {str(e)}")

@app.post("/run-research")
async def run_research(config: ResearchConfig, background_tasks: BackgroundTasks):
    """Start a deep research task with the given configuration"""
    task_id = f"research_{os.urandom(4).hex()}"
    
    # Store task in MongoDB
    await db.store_task(task_id, "research", config.dict())
    
    background_tasks.add_task(
        run_research_task,
        task_id=task_id,
        config=config
    )
    
    return {"task_id": task_id, "status": "started"}

@app.get("/screenshot")
async def get_screenshot():
    """Get the current browser screenshot"""
    global _global_browser_context
    
    try:
        if _global_browser_context:
            encoded_screenshot = await capture_screenshot(_global_browser_context)
            if encoded_screenshot:
                return {"status": "success", "screenshot": encoded_screenshot}
        
        return {"status": "no_browser", "message": "No active browser context"}
    except Exception as e:
        logger.error(f"Error capturing screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error capturing screenshot: {str(e)}")

@app.post("/close-browser")
async def close_browser():
    """Close the global browser instance"""
    global _global_browser, _global_browser_context
    
    try:
        if _global_browser_context:
            await _global_browser_context.close()
            _global_browser_context = None
            
        if _global_browser:
            await _global_browser.close()
            _global_browser = None
            
        return {"status": "success", "message": "Browser closed"}
    except Exception as e:
        logger.error(f"Error closing browser: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error closing browser: {str(e)}")

# Task implementations
async def run_browser_agent_task(task_id: str, config: AgentConfig):
    """Background task to run the browser agent"""
    global _global_agent_state, _global_browser, _global_browser_context, _global_agent
    
    # Update task status to running
    await db.update_task_status(task_id, "running")
    
    # Store task results (would be better in a proper task storage system)
    results = {
        "task_id": task_id,
        "status": "running",
        "final_result": "",
        "errors": "",
        "model_actions": "",
        "model_thoughts": "",
        "latest_video": None,
        "trace_file": None,
        "history_file": None
    }
    
    try:
        _global_agent_state.clear_stop()
        
        # Disable recording if not enabled
        save_recording_path = None
        if config.enable_recording and config.save_recording_path:
            save_recording_path = config.save_recording_path
            os.makedirs(save_recording_path, exist_ok=True)
            
            # Get existing videos
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
        
        # Resolve sensitive variables in task
        task = resolve_sensitive_env_variables(config.task)
        
        # Get LLM model
        llm = utils.get_llm_model(
            provider=config.llm_provider,
            model_name=config.llm_model_name,
            num_ctx=config.llm_num_ctx,
            temperature=config.llm_temperature,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        
        # Run the appropriate agent type
        if config.agent_type == "org":
            result = await run_org_agent(
                llm=llm,
                use_own_browser=config.use_own_browser,
                keep_browser_open=config.keep_browser_open,
                headless=config.headless,
                disable_security=config.disable_security,
                window_w=config.window_w,
                window_h=config.window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=config.save_agent_history_path,
                save_trace_path=config.save_trace_path,
                task=task,
                max_steps=config.max_steps,
                use_vision=config.use_vision,
                max_actions_per_step=config.max_actions_per_step,
                tool_calling_method=config.tool_calling_method,
                chrome_cdp=config.chrome_cdp
            )
        elif config.agent_type == "custom":
            result = await run_custom_agent(
                llm=llm,
                use_own_browser=config.use_own_browser,
                keep_browser_open=config.keep_browser_open,
                headless=config.headless,
                disable_security=config.disable_security,
                window_w=config.window_w,
                window_h=config.window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=config.save_agent_history_path,
                save_trace_path=config.save_trace_path,
                task=task,
                add_infos=config.add_infos,
                max_steps=config.max_steps,
                use_vision=config.use_vision,
                max_actions_per_step=config.max_actions_per_step,
                tool_calling_method=config.tool_calling_method,
                chrome_cdp=config.chrome_cdp,
                register_new_step_callback=None
            )
        else:
            raise ValueError(f"Invalid agent type: {config.agent_type}")
            
        # Update results
        final_result, errors, model_actions, model_thoughts, trace_file, history_file = result
        
        # Get latest video if recording was enabled
        latest_video = None
        if save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]
                
        results.update({
            "status": "completed",
            "final_result": final_result,
            "errors": errors,
            "model_actions": model_actions,
            "model_thoughts": model_thoughts,
            "latest_video": latest_video,
            "trace_file": trace_file,
            "history_file": history_file
        })
        
        # Update task in MongoDB
        await db.update_task_status(
            task_id, 
            "completed", 
            {
                "final_result": final_result,
                "model_actions": model_actions,
                "model_thoughts": model_thoughts,
                "latest_video": latest_video,
                "trace_file": trace_file,
                "history_file": history_file
            }
        )
        
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Error in agent task {task_id}: {error_details}")
        results.update({
            "status": "failed",
            "errors": error_details
        })
        
        # Update task in MongoDB with error
        await db.update_task_status(task_id, "failed", errors=error_details)
        
    finally:
        _global_agent = None
        if not config.keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None
                
            if _global_browser:
                await _global_browser.close()
                _global_browser = None
    
    logger.info(f"Task {task_id} completed with status: {results['status']}")
    return results

async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state, _global_agent
        
        # Clear any previous stop request
        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None
            
        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    cdp_url=cdp_url,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        register_new_step_callback=None
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state, _global_agent

        # Clear any previous stop request
        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp
        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)

            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()

        # Initialize global browser if needed
        if _global_browser is None:
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        # Create and run agent
        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                register_new_step_callback=register_new_step_callback
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)        

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_research_task(task_id: str, config: ResearchConfig):
    from src.utils.deep_research import deep_research
    global _global_agent_state
    
    # Update task status to running
    await db.update_task_status(task_id, "running")
    
    results = {
        "task_id": task_id,
        "status": "running",
        "markdown_content": "",
        "file_path": None
    }
    
    try:
        # Clear any previous stop request
        _global_agent_state.clear_stop()
        
        llm = utils.get_llm_model(
            provider=config.llm_provider,
            model_name=config.llm_model_name,
            num_ctx=config.llm_num_ctx,
            temperature=config.llm_temperature,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )
        
        markdown_content, file_path = await deep_research(
            config.research_task, 
            llm, 
            _global_agent_state,
            max_search_iterations=config.max_search_iteration_input,
            max_query_num=config.max_query_per_iter_input,
            use_vision=config.use_vision,
            headless=config.headless,
            use_own_browser=config.use_own_browser,
            chrome_cdp=config.chrome_cdp
        )
        
        results.update({
            "status": "completed",
            "markdown_content": markdown_content,
            "file_path": file_path
        })
        
        # Update task in MongoDB
        await db.update_task_status(
            task_id, 
            "completed", 
            {
                "markdown_content": markdown_content,
                "file_path": file_path
            }
        )
        
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Error in research task {task_id}: {error_details}")
        results.update({
            "status": "failed",
            "errors": error_details
        })
        
        # Update task in MongoDB with error
        await db.update_task_status(task_id, "failed", errors=error_details)
    
    logger.info(f"Research task {task_id} completed with status: {results['status']}")
    return results

@app.get("/task/{task_id}")
async def get_task_result(task_id: str):
    """Get the result of a task"""
    task = await db.get_task(task_id)
    if task:
        return {"tash": str(task)}
    else:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

@app.get("/research-file/{task_id}")
async def get_research_file(task_id: str):
    """Get the research file for a completed research task"""
    task = await db.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Task {task_id} is not completed")
        
    if "result" not in task or not task["result"] or "file_path" not in task["result"]:
        raise HTTPException(status_code=404, detail=f"No file found for task {task_id}")
        
    file_path = task["result"]["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found at {file_path}")
        
    return FileResponse(file_path)

@app.get("/tasks")
async def get_recent_tasks(limit: int = 10):
    """Get the most recent tasks"""
    tasks = db.get_recent_tasks(limit)
    return {"tasks": tasks}

# Add a middleware to handle errors
@app.middleware("http")
async def add_error_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        import traceback
        error_detail = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Unhandled exception: {error_detail}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e), "type": "InternalServerError"}
        )
    
def tool_usage_callback(browser_state, model_output, step_number):
    """
    Test callback function to log tool usage information to the console
    
    Args:
        browser_state: The current browser state
        model_output: The model's output containing actions/tools used
        step_number: The current step number
    """
    print(f"\n===== TOOL USAGE - STEP {step_number} =====")
    print(f"Thought: {model_output.current_state.thought}")
    
    # Log each action/tool
    for i, action in enumerate(model_output.action):
        try:
            # Convert action to a dict for display
            action_data = action.model_dump(exclude_unset=True)
            action_name = action_data.get("name", "unknown")
            action_params = action_data.get("parameters", {})
            
            print(f"\nTool {i+1}: {action_name}")
            print(f"Parameters: {action_params}")
        except Exception as e:
            print(f"\nTool {i+1}: Error serializing action - {str(e)}")
    
    print(f"\nSummary: {model_output.current_state.summary}")
    print(f"Task Progress: {model_output.current_state.task_progress}")
    print("======================================\n")

# Cleanup MongoDB connection when the app shuts down
@app.on_event("shutdown")
async def shutdown_event():
    db.close()

# Store connected WebSocket clients
connected_websockets = {}

@app.websocket("/agent/ws")
async def websocket_agent(websocket: WebSocket):
    await websocket.accept()
    
    # Generate a unique client ID
    client_id = f"client_{os.urandom(4).hex()}"
    connected_websockets[client_id] = websocket
    client_config = {}  # Store client configuration
    
    logger.info(f"🔌 New WebSocket connection established - Client ID: {client_id}")
    
    try:
        while True:  # Keep connection alive
            # Receive message
            data = await websocket.receive_text()
            logger.info(f"📩 Received message from client {client_id}:")
            logger.info(f"Raw message: {data}")
            
            message = json.loads(data)
            logger.info(f"Parsed message: {json.dumps(message, indent=2)}")
            
            # Handle different message types
            message_type = message.get("type", "")
            logger.info(f"Message type: {message_type}")
            
            if message_type == "connection_ack":
                logger.info(f"👋 Processing connection acknowledgment from client {client_id}")
                # Store the client configuration
                client_config = message.get("config", {})
                logger.info(f"📝 Stored client configuration: {json.dumps(client_config, indent=2)}")
                
                # Send confirmation
                response = {
                    "type": "connection_confirmed",
                    "data": {
                        "client_id": client_id,
                        "status": "ready",
                        "message": f"Connected and ready. Instance: {client_config.get('add_infos', 'unknown')}"
                    },
                    "timestamp": datetime.now().isoformat() + "Z"
                }
                logger.info(f"✅ Sending connection confirmation: {json.dumps(response, indent=2)}")
                await websocket.send_json(response)
                continue
                
            elif message_type == "create_task":
                logger.info(f"🎯 Processing create_task message from client {client_id}")
                # Use stored configuration if available, otherwise use message config
                config_data = message.get("config", client_config)
                
                # Extract task details
                task = config_data.get("task", {})
                add_infos = config_data.get("add_infos", "")
                
                # Extract config settings or use defaults
                config_settings = config_data.get("settings", {})
                agent_config = AgentConfig(
                    task=task,
                    add_infos=add_infos,
                    headless=config_settings.get("headless", True),
                    use_vision=config_settings.get("use_vision", True),
                    max_steps=config_settings.get("max_steps", 100),
                    max_actions_per_step=config_settings.get("max_actions_per_step", 10),
                    llm_api_key=os.getenv("OPENAI_API_KEY", None)
                )
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "task_created",
                    "data": {
                        "status": "starting",
                        "message": f"Starting agent with task: {task}"
                    },
                    "timestamp": datetime.now().isoformat() + "Z"
                })
                
                # Define the websocket callback for tool usage
                def ws_tool_usage_callback(browser_state, model_output, step_number):
                    """Send tool usage information to the WebSocket client"""
                    actions = []
                    for action in model_output.action:
                        try:
                            # Get the action type and parameters
                            action_data = action.model_dump(exclude_unset=True)
                            action_type = next(iter(action_data.keys())) if action_data else "unknown"
                            action_params = action_data.get(action_type, {})
                            
                            actions.append({
                                "type": action_type,
                                "parameters": action_params
                            })
                        except Exception as e:
                            actions.append({
                                "type": "error",
                                "error": str(e)
                            })
                    
                    # Create message with all relevant information
                    message = {
                        "type": "tool_usage",
                        "data": {
                            "step": step_number,
                            "thought": model_output.current_state.thought,
                            "summary": model_output.current_state.summary,
                            "task_progress": model_output.current_state.task_progress,
                            "future_plans": model_output.current_state.future_plans,
                            "actions": actions
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    }
                    
                    # Send to this specific client
                    asyncio.create_task(send_ws_message(websocket, message))
                
                # Helper function to send WebSocket messages asynchronously
                async def send_ws_message(ws, message):
                    try:
                        await ws.send_json(message)
                    except Exception as e:
                        logger.error(f"Error sending WebSocket message: {e}")
                
                # Run the agent with the WebSocket callback
                task_id = f"ws_task_{os.urandom(4).hex()}"
                
                # Store task in MongoDB
                await db.store_task(task_id, "agent", agent_config.dict())
                
                # Update client with task ID
                await websocket.send_json({
                    "type": "status",
                    "data": {
                        "status": "running",
                        "task_id": task_id,
                        "message": "Agent is running"
                    },
                    "timestamp": datetime.now().isoformat() + "Z"
                })
                
                # Run the agent
                result = await run_custom_agent(
                    llm=utils.get_llm_model(
                        provider=agent_config.llm_provider,
                        model_name=agent_config.llm_model_name,
                        num_ctx=agent_config.llm_num_ctx,
                        temperature=agent_config.llm_temperature,
                        base_url=agent_config.llm_base_url,
                        api_key=agent_config.llm_api_key,
                    ),
                    use_own_browser=agent_config.use_own_browser,
                    keep_browser_open=agent_config.keep_browser_open,
                    headless=agent_config.headless,
                    disable_security=agent_config.disable_security,
                    window_w=agent_config.window_w,
                    window_h=agent_config.window_h,
                    save_recording_path=agent_config.save_recording_path if agent_config.enable_recording else None,
                    save_agent_history_path=agent_config.save_agent_history_path,
                    save_trace_path=agent_config.save_trace_path,
                    task=task,
                    add_infos=add_infos,
                    max_steps=agent_config.max_steps,
                    use_vision=agent_config.use_vision,
                    max_actions_per_step=agent_config.max_actions_per_step,
                    tool_calling_method=agent_config.tool_calling_method,
                    chrome_cdp=agent_config.chrome_cdp,
                    register_new_step_callback=ws_tool_usage_callback
                )
                
                # Send final results
                final_result, errors, model_actions, model_thoughts, trace_file, history_file = result
                
                await websocket.send_json({
                    "type": "result",
                    "data": {
                        "final_result": final_result,
                        "errors": errors,
                    },
                    "timestamp": datetime.now().isoformat() + "Z"
                })
            elif message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat() + "Z"
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {
                        "message": f"Unknown message type: {message_type}"
                    },
                    "timestamp": datetime.now().isoformat() + "Z"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
                "details": traceback.format_exc(),
                "timestamp": datetime.now().isoformat() + "Z"
            })
        except:
            pass
    finally:
        # Clean up
        if client_id in connected_websockets:
            del connected_websockets[client_id]

# Store SSH terminal connections by client ID
ssh_terminals = {}

@app.websocket("/ssh/ws")
async def websocket_ssh_terminal(websocket: WebSocket):
    await websocket.accept()
    
    # Generate a unique client ID
    client_id = f"ssh_{os.urandom(4).hex()}"
    ssh_terminals[client_id] = SSHTerminal()
    
    logger.info(f"🔌 New SSH Terminal WebSocket connection established - Client ID: {client_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_confirmed",
            "data": {
                "client_id": client_id,
                "status": "ready",
                "message": "SSH Terminal connection established"
            },
            "timestamp": datetime.now().isoformat() + "Z"
        })
        
        while True:  # Keep connection alive
            # Receive message
            data = await websocket.receive_text()
            logger.info(f"📩 Received SSH message from client {client_id}")
            
            message = json.loads(data)
            message_type = message.get("type", "")
            
            if message_type == "connect":
                # Connect to SSH server
                connection_name = message.get("connection_name", f"conn_{os.urandom(4).hex()}")
                connection_params = message.get("connection", {})
                
                try:
                    # Create a callback function to send output to the WebSocket
                    async def output_callback(output: str):
                        await websocket.send_json({
                            "type": "terminal_output",
                            "data": {
                                "connection_name": connection_name,
                                "output": output
                            },
                            "timestamp": datetime.now().isoformat() + "Z"
                        })
                    
                    # Connect to SSH server
                    connection = ssh_terminals[client_id].connect(
                        name=connection_name,
                        host=connection_params.get("host", ""),
                        username=connection_params.get("username", ""),
                        password=connection_params.get("password"),
                        key_filename=connection_params.get("key_filename"),
                        port=connection_params.get("port", 22)
                    )
                    
                    # Set up the callback to stream output
                    def ws_callback(output):
                        asyncio.create_task(output_callback(output))
                    
                    connection.set_output_callback(ws_callback)
                    
                    # Send success response
                    await websocket.send_json({
                        "type": "ssh_connected",
                        "data": {
                            "connection_name": connection_name,
                            "status": "connected",
                            "message": f"Connected to {connection_params.get('host')}"
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    })
                    
                except Exception as e:
                    # Send error response
                    error_details = str(e) + "\n" + traceback.format_exc()
                    logger.error(f"SSH connection error: {error_details}")
                    
                    await websocket.send_json({
                        "type": "ssh_error",
                        "data": {
                            "connection_name": connection_name,
                            "error": str(e),
                            "details": error_details
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    })
            
            elif message_type == "execute":
                # Execute command on an existing connection
                connection_name = message.get("connection_name", "")
                command = message.get("command", "")
                
                try:
                    # Execute the command
                    ssh_terminals[client_id].execute_command(connection_name, command)
                    
                    # Command sent confirmation
                    await websocket.send_json({
                        "type": "command_sent",
                        "data": {
                            "connection_name": connection_name,
                            "command": command
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    })
                    
                except Exception as e:
                    # Send error response
                    error_details = str(e)
                    logger.error(f"SSH command error: {error_details}")
                    
                    await websocket.send_json({
                        "type": "ssh_error",
                        "data": {
                            "connection_name": connection_name,
                            "command": command,
                            "error": error_details
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    })
            
            elif message_type == "get_state":
                # Get current terminal state
                connection_name = message.get("connection_name", "")
                
                try:
                    terminal_state = ssh_terminals[client_id].get_terminal_state(connection_name)
                    
                    await websocket.send_json({
                        "type": "terminal_state",
                        "data": {
                            "connection_name": connection_name,
                            "state": terminal_state
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "ssh_error",
                        "data": {
                            "connection_name": connection_name,
                            "error": str(e)
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    })
            
            elif message_type == "disconnect":
                # Disconnect a specific SSH connection
                connection_name = message.get("connection_name", "")
                
                try:
                    ssh_terminals[client_id].disconnect(connection_name)
                    
                    await websocket.send_json({
                        "type": "ssh_disconnected",
                        "data": {
                            "connection_name": connection_name,
                            "status": "disconnected"
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    })
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "ssh_error",
                        "data": {
                            "connection_name": connection_name,
                            "error": str(e)
                        },
                        "timestamp": datetime.now().isoformat() + "Z"
                    })
            
            elif message_type == "ping":
                # Simple ping to keep the connection alive
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat() + "Z"
                })
            
            else:
                # Unknown message type
                await websocket.send_json({
                    "type": "error",
                    "data": {
                        "message": f"Unknown message type: {message_type}"
                    },
                    "timestamp": datetime.now().isoformat() + "Z"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket SSH client {client_id} disconnected")
    except Exception as e:
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"WebSocket SSH error: {error_details}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
                "details": error_details,
                "timestamp": datetime.now().isoformat() + "Z"
            })
        except:
            pass
    finally:
        # Clean up all SSH connections for this client
        if client_id in ssh_terminals:
            try:
                ssh_terminals[client_id].disconnect_all()
                del ssh_terminals[client_id]
            except Exception as e:
                logger.error(f"Error cleaning up SSH connections: {str(e)}")

# Add cleanup of SSH terminals when the app shuts down
@app.on_event("shutdown")
async def shutdown_ssh_terminals():
    for client_id, terminal in ssh_terminals.items():
        try:
            terminal.disconnect_all()
        except Exception as e:
            logger.error(f"Error disconnecting SSH terminal {client_id}: {str(e)}")
    ssh_terminals.clear()

# Entry point for running the app
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Browser Agent API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)


