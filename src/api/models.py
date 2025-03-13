from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class MessageType(str, Enum):
    STATUS = "status"
    LOG = "log"
    BROWSER_STATE = "browser_state"
    RESULT = "result"
    ERROR = "error"


class TaskRequest(BaseModel):
    task: str = Field(..., description="The task description for the agent to execute")
    add_infos: Optional[str] = Field(None, description="Additional context or instructions")
    config: Optional[Dict[str, Any]] = Field(None, description="Optional configuration overrides")


class AgentUpdate(BaseModel):
    type: MessageType = Field(..., description="Type of the update message")
    data: Any = Field(..., description="The update data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the update")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow) 