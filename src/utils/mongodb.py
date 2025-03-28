from typing import Dict, Any, Optional, List
from pymongo import MongoClient
import motor.motor_asyncio
import os
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB utility class for storing and retrieving task data"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.tasks_collection = None
        self.connect()
        
    def connect(self):
        """Connect to MongoDB using environment variables"""
        try:
            mongo_uri = f"mongodb://{os.getenv('MONGODB_ROOT_USERNAME')}:{os.getenv('MONGODB_ROOT_PASSWORD')}@mongodb:27017/?authSource=admin"
            self.mongo_uri = mongo_uri
            # Create MongoDB client
            print(mongo_uri)
            self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri,
                                     serverSelectionTimeoutMS=5000)
            self.db = self.client["brownser"]
            self.tasks_collection = self.db["tasks"]
            self.instances = self.db["instances"]
            self.user_chat_context_collection = self.db["user_chat_context"]
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
    
    def is_connected(self) -> bool:
        print(self.client)
        print(self.db)
        print(self.tasks_collection)
        """Check if MongoDB is connected and available"""       
        if self.client is None:
            return False
        return True
    
    def _serialize_for_mongodb(self, data: Any) -> Any:
        """
        Recursively serialize data to ensure it's MongoDB compatible
        
        Args:
            data: Any data structure to serialize
            
        Returns:
            MongoDB compatible data structure
        """
        if data is None:
            return None
        
        if isinstance(data, (str, int, float, bool)):
            return data
        
        if isinstance(data, (datetime,)):
            return data
        
        if isinstance(data, dict):
            return {k: self._serialize_for_mongodb(v) for k, v in data.items()}
        
        if isinstance(data, list):
            return [self._serialize_for_mongodb(item) for item in data]
        
        if isinstance(data, tuple):
            return [self._serialize_for_mongodb(item) for item in data]
        
        # For custom objects or anything else, convert to string representation
        try:
            return str(data)
        except Exception as e:
            logger.warning(f"Could not serialize object of type {type(data)}: {str(e)}")
            return f"<Unserializable object of type {type(data).__name__}>"
    
    async def store_task(self, task_id: str, task_type: str, config: Dict[str, Any]) -> str:
        """
        Store a new task in the database
        
        Args:
            task_id: Unique identifier for the task
            task_type: Type of task (agent or research)
            config: Task configuration
            
        Returns:
            task_id: The ID of the stored task
        """
        if not self.is_connected():
            logger.warning(f"MongoDB not connected, skipping task storage for {task_id}")
            logger.warning(f"MongoDB not connected, {self.mongo_uri}")
            return task_id
            
        try:
            # Create task document
            task_doc = {
                "task_id": task_id,
                "task_type": task_type,
                "status": "started",
                "config": self._serialize_for_mongodb(config),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "result": None,
                "errors": None
            }
            
            # Insert or update task
            await self.tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": task_doc},
                upsert=True
            )
            
            return task_id
        except Exception as e:
            logger.error(f"Error storing task {task_id}: {str(e)}")
            return task_id
    
    async def update_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None, 
                          errors: Optional[str] = None) -> bool:
        """
        Update the status and results of a task
        
        Args:
            task_id: Unique identifier for the task
            status: New status (started, running, completed, failed)
            result: Task results (optional)
            errors: Error messages (optional)
            
        Returns:
            bool: True if update was successful
        """
        if not self.is_connected():
            logger.warning(f"MongoDB not connected, skipping task update for {task_id}")
            return False
            
        try:
            update_doc = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if result is not None:
                # Serialize complex objects in the result
                serialized_result = self._serialize_for_mongodb(result)
                update_doc["result"] = serialized_result
                
            if errors is not None:
                update_doc["errors"] = errors
            
            update_result = await self.tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": update_doc}
            )
            
            return update_result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {str(e)}")
            return False
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a task by its ID
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            dict: Task document or None if not found
        """
        if not self.is_connected():
            logger.warning(f"MongoDB not connected, cannot retrieve task {task_id}")
            logger.warning(f"MongoDB not connected, {self.mongo_uri}")
            return None
            
        try:
            task = await self.tasks_collection.find_one({"task_id": task_id})
            return task
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {str(e)}")
            return None
    
    async def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent tasks
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            list: List of task documents
        """
        if not self.is_connected():
            logger.warning("MongoDB not connected, cannot retrieve recent tasks")
            logger.warning(f"MongoDB not connected, {self.mongo_uri}")
            return []
            
        try:
            tasks = await self.tasks_collection.find().sort("created_at", -1).limit(limit)
            return list(tasks)
        except Exception as e:
            logger.error(f"Error retrieving recent tasks: {str(e)}")
            return []
    
    async def save_user_chat_context(self, person_name: str, screenshot: str, context: str):
        """
        Save user chat context to MongoDB
        
        Args:
            person_name: Name of the person the user is talking to
            screenshot: Screenshot data
            context: Chat context data
        """
        try:
            await self.user_chat_context_collection.insert_one({
                "person_name": person_name,
                "screenshot": screenshot,
                "context": context,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            logger.info(f"Saved chat context for conversation with: {person_name}")
        except Exception as e:
            logger.error(f"Error saving chat context for conversation with {person_name}: {str(e)}")
    
    async def get_latest_chat_context(self, person_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent chat context for a specific person
        
        Args:
            person_name: Name of the person to retrieve chat context for
            
        Returns:
            dict: Latest chat context or None if not found
        """
        if not self.is_connected():
            logger.warning(f"MongoDB not connected, cannot retrieve chat context for {person_name}")
            return None
            
        try:
            # Find the most recent chat context for this person
            chat_context = await self.user_chat_context_collection.find_one(
                {"person_name": person_name},
                sort=[("created_at", -1)]  # Sort by creation time, newest first
            )
            return chat_context
        except Exception as e:
            logger.error(f"Error retrieving chat context for {person_name}: {str(e)}")
            return None
    
    async def get_recent_chat_contexts(self, person_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve multiple recent chat contexts for a specific person
        
        Args:
            person_name: Name of the person to retrieve chat contexts for
            limit: Maximum number of chat contexts to return (default: 10)
            
        Returns:
            list: List of recent chat contexts sorted by recency (newest first)
        """
        if not self.is_connected():
            logger.warning(f"MongoDB not connected, cannot retrieve chat contexts for {person_name}")
            return []
            
        try:
            # Find recent chat contexts for this person
            cursor = self.user_chat_context_collection.find(
                {"person_name": person_name}
            ).sort("created_at", -1).limit(limit)
            
            # Convert cursor to list
            chat_contexts = await cursor.to_list(length=limit)
            
            logger.info(f"Retrieved {len(chat_contexts)} recent chat contexts for {person_name}")
            return chat_contexts
        except Exception as e:
            logger.error(f"Error retrieving chat contexts for {person_name}: {str(e)}")
            return []
    
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Create a singleton instance
db = MongoDB()