import os
import logging
from dotenv import load_dotenv
logger = logging.getLogger(__name__)
load_dotenv(override=True)

# Default configuration with values from environment variables
default_config = {
    "agent_type": os.getenv("AGENT_TYPE", "custom"),
    "max_steps": int(os.getenv("MAX_STEPS", "100")),
    "max_actions_per_step": int(os.getenv("MAX_ACTIONS_PER_STEP", "5")),
    "use_vision": os.getenv("USE_VISION", "false").lower() == "true",
    "tool_calling_method": os.getenv("TOOL_CALLING_METHOD", "auto"),
    
    # LLM configuration
    "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
    "llm_model_name": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "llm_num_ctx": int(os.getenv("LLM_NUM_CTX", "4096")),
    "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "llm_base_url": os.getenv("OPENAI_ENDPOINT"),
    "llm_api_key": os.getenv("OPENAI_API_KEY"),
    
    # Browser configuration
    "use_own_browser": os.getenv("CHROME_PERSISTENT_SESSION", "false").lower() == "true",
    "keep_browser_open": os.getenv("KEEP_BROWSER_OPEN", "false").lower() == "true",
    "headless": os.getenv("HEADLESS", "true").lower() == "true",
    "disable_security": os.getenv("DISABLE_SECURITY", "false").lower() == "true",
    "enable_recording": os.getenv("ENABLE_RECORDING", "true").lower() == "true",
    "window_w": int(os.getenv("RESOLUTION_WIDTH", "1280")),
    "window_h": int(os.getenv("RESOLUTION_HEIGHT", "720")),
    
    # Save paths
    "save_recording_path": os.getenv("SAVE_RECORDING_PATH"),
    "save_trace_path": os.getenv("SAVE_TRACE_PATH"),
    "save_agent_history_path": os.getenv("SAVE_AGENT_HISTORY_PATH", "./tmp/agent_history"),
    
    # Task
    "task": os.getenv("TASK", "go to google.com and type 'OpenAI' click search and give me the first url"),
}