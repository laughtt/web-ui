import os
import pickle
import uuid
import gradio as gr


# Default configuration as a simple dictionary
default_config = {
    "agent_type": "custom",
    "max_steps": 100,
    "max_actions_per_step": 5,
    "use_vision": False,
    "tool_calling_method": "auto",
    "llm_provider": "openai",
    "llm_model_name": "your-default-model",
    "llm_num_ctx": 4096,
    "llm_temperature": 0.7,
    "llm_base_url": None,
    "llm_api_key": "your-default-api-key",
    "use_own_browser": os.getenv("CHROME_PERSISTENT_SESSION", "false").lower() == "true",
    "keep_browser_open": False,
    "headless": True,
    "disable_security": False,
    "enable_recording": True,
    "window_w": 1280,
    "window_h": 720,
    "save_recording_path": None,
    "save_trace_path": None,
    "save_agent_history_path": "./tmp/agent_history",
    "task": "go to google.com and type 'OpenAI' click search and give me the first url",
}


def load_config_from_file(config_file):
    """Load settings from a UUID.pkl file."""
    try:
        with open(config_file, 'rb') as f:
            settings = pickle.load(f)
        return settings
    except Exception as e:
        return f"Error loading configuration: {str(e)}"


def save_config_to_file(settings, save_dir="./tmp/webui_settings"):
    """Save the current settings to a UUID.pkl file with a UUID name."""
    os.makedirs(save_dir, exist_ok=True)
    config_file = os.path.join(save_dir, f"{uuid.uuid4()}.pkl")
    with open(config_file, 'wb') as f:
        pickle.dump(settings, f)
    return f"Configuration saved to {config_file}"


def save_current_config(*args):
    current_config = {
        "agent_type": args[0],
        "max_steps": args[1],
        "max_actions_per_step": args[2],
        "use_vision": args[3],
        "tool_calling_method": args[4],
        "llm_provider": args[5],
        "llm_model_name": args[6],
        "llm_num_ctx": args[7],
        "llm_temperature": args[8],
        "llm_base_url": args[9],
        "llm_api_key": args[10],
        "use_own_browser": args[11],
        "keep_browser_open": args[12],
        "headless": args[13],
        "disable_security": args[14],
        "enable_recording": args[15],
        "window_w": args[16],
        "window_h": args[17],
        "save_recording_path": args[18],
        "save_trace_path": args[19],
        "save_agent_history_path": args[20],
        "task": args[21],
    }
    return save_config_to_file(current_config)


def update_ui_from_config(config_file):
    if config_file is not None:
        loaded_config = load_config_from_file(config_file.name)
        if isinstance(loaded_config, dict):
            return (
                gr.update(value=loaded_config.get("agent_type", "custom")),
                gr.update(value=loaded_config.get("max_steps", 100)),
                gr.update(value=loaded_config.get("max_actions_per_step", 5)),
                gr.update(value=loaded_config.get("use_vision", False)),
                gr.update(value=loaded_config.get("tool_calling_method", "auto")),
                gr.update(value=loaded_config.get("llm_provider", "openai")),
                gr.update(value=loaded_config.get("llm_model_name", "your-default-model")),
                gr.update(value=loaded_config.get("llm_num_ctx", 4096)),
                gr.update(value=loaded_config.get("llm_temperature", 0.7)),
                gr.update(value=loaded_config.get("llm_base_url", None)),
                gr.update(value=loaded_config.get("llm_api_key", "your-default-api-key")),
                gr.update(value=loaded_config.get("use_own_browser", False)),
                gr.update(value=loaded_config.get("keep_browser_open", False)),
                gr.update(value=loaded_config.get("headless", True)),
                gr.update(value=loaded_config.get("disable_security", False)),
                gr.update(value=loaded_config.get("enable_recording", True)),
                gr.update(value=loaded_config.get("window_w", 1280)),
                gr.update(value=loaded_config.get("window_h", 720)),
                gr.update(value=loaded_config.get("save_recording_path", None)),
                gr.update(value=loaded_config.get("save_trace_path", None)),
                gr.update(value=loaded_config.get("save_agent_history_path", "./tmp/agent_history")),
                gr.update(value=loaded_config.get("task", "go to google.com and type 'OpenAI' click search and give me the first url")),
                "Configuration loaded successfully."
            )
        else:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), "Error: Invalid configuration file."
            )
    return (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), "No file selected."
    )
