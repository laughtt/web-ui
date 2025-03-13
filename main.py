from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
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

load_dotenv()

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
    agent_type: str = "custom"  # Default to custom agent as it's likely more feature-rich
    llm_provider: str = "openai"  # OpenAI is a common default
    llm_model_name: str = "gpt-4o"  # Modern capable model
    llm_num_ctx: int = 16384  # Reasonable context length for complex tasks
    llm_temperature: float = 0.7  # Balanced creativity vs determinism
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    use_own_browser: bool = False  # Simpler to start with managed browser
    keep_browser_open: bool = True  # Better performance between tasks
    headless: bool = True  # More efficient for API usage
    disable_security: bool = False  # Security should be on by default
    window_w: int = 1280  # Standard resolution width
    window_h: int = 720  # Standard resolution height
    save_recording_path: Optional[str] = "./tmp/record_videos"
    save_agent_history_path: Optional[str] = "./tmp/agent_history"
    save_trace_path: Optional[str] = "./tmp/traces"
    enable_recording: bool = False  # Recording off by default to save resources
    task: str = "Navigate to example.com and summarize the main content of the page."
    add_infos: Optional[str] = None
    max_steps: int = 15  # Reasonable limit for most tasks
    use_vision: bool = True  # Vision is valuable for browser automation
    max_actions_per_step: int = 5  # Balanced for most tasks
    tool_calling_method: str = "function_calling"  # Most reliable method
    chrome_cdp: Optional[str] = None

class ResearchConfig(BaseModel):
    research_task: str = "Compose a report on the use of Reinforcement Learning for training Large Language Models, encompassing its origins, current advancements, and future prospects."
    max_search_iteration_input: int = 3  # Good balance between depth and time
    max_query_per_iter_input: int = 3  # Multiple queries per iteration for breadth
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o"  # Research benefits from stronger models
    llm_num_ctx: int = 16384  # Larger context for research tasks
    llm_temperature: float = 0.5  # Lower temperature for more factual responses
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    use_vision: bool = True  # Vision helps with research on web pages
    use_own_browser: bool = False
    headless: bool = True  # Headless is more efficient for research
    chrome_cdp: Optional[str] = None

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
    
    background_tasks.add_task(
        run_browser_agent_task,
        task_id=task_id,
        config=config
    )
    
    return {"task_id": task_id, "status": "started"}

@app.get("/agent-status/{task_id}")
async def get_agent_status(task_id: str):
    """Get the status of a running agent task"""
    # This would need a task tracking system implementation
    # For simplicity, we'll return a placeholder
    return {"task_id": task_id, "status": "running"}

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
                chrome_cdp=config.chrome_cdp
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
        
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Error in agent task {task_id}: {error_details}")
        results.update({
            "status": "failed",
            "errors": error_details
        })
    finally:
        _global_agent = None
        if not config.keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None
                
            if _global_browser:
                await _global_browser.close()
                _global_browser = None
    
    # In a real implementation, store results in a database or cache
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

        controller = CustomController()

        # Initialize global browser if needed
        if ((_global_browser is None) or (cdp_url and cdp_url != "" and cdp_url != None)):
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if (_global_browser_context is None or (chrome_cdp and cdp_url != "" and cdp_url != None)):
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

async def run_research_task(task_id: str, config: ResearchConfig):
    from src.utils.deep_research import deep_research
    global _global_agent_state
    
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
        
    except Exception as e:
        import traceback
        error_details = str(e) + "\n" + traceback.format_exc()
        logger.error(f"Error in research task {task_id}: {error_details}")
        results.update({
            "status": "failed",
            "errors": error_details
        })
    
    # In a real implementation, store results in a database or cache
    logger.info(f"Research task {task_id} completed with status: {results['status']}")
    return results

@app.get("/task/{task_id}")
async def get_task_result(task_id: str):
    """
    Get the result of a task
    In a real implementation, this would retrieve from a database or cache
    """
    # Placeholder implementation - in a real app, retrieve from storage
    return {"task_id": task_id, "status": "unknown", "message": "Task retrieval not implemented"}

@app.get("/research-file/{task_id}")
async def get_research_file(task_id: str):
    """
    Get the research file for a completed research task
    In a real implementation, this would retrieve the file path from a database
    """
    # Placeholder implementation - in a real app, retrieve file path from storage
    return {"task_id": task_id, "status": "unknown", "message": "File retrieval not implemented"}

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

# Entry point for running the app
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Browser Agent API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    
    args = parser.parse_args()
    
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)