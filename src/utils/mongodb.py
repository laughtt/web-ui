from typing import Dict, Any, Optional, List
from pymongo import MongoClient
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
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://root:rootpassword@localhost:27017/brownser?authSource=admin")
            
            # Create MongoDB client
            self.client = MongoClient(mongodb_uri,
                                     serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Access database
            self.db = self.client[mongodb_database]
            
            # Access collections
            self.tasks_collection = self.db["tasks"]
            
            # Create indexes for faster queries
            self.tasks_collection.create_index("task_id", unique=True)
            self.tasks_collection.create_index("created_at")
            
            logger.info(f"Connected to MongoDB: {mongodb_uri}, database: {mongodb_database}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.client = None
            self.db = None
            self.tasks_collection = None
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected and available"""
        return self.client is not None and self.tasks_collection is not None
    
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
    
    def store_task(self, task_id: str, task_type: str, config: Dict[str, Any]) -> str:
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
            self.tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": task_doc},
                upsert=True
            )
            
            return task_id
        except Exception as e:
            logger.error(f"Error storing task {task_id}: {str(e)}")
            return task_id
    
    def update_task_status(self, task_id: str, status: str, result: Optional[Dict[str, Any]] = None, 
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
            
            update_result = self.tasks_collection.update_one(
                {"task_id": task_id},
                {"$set": update_doc}
            )
            
            return update_result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating task {task_id}: {str(e)}")
            return False
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a task by its ID
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            dict: Task document or None if not found
        """
        if not self.is_connected():
            logger.warning(f"MongoDB not connected, cannot retrieve task {task_id}")
            return None
            
        try:
            task = self.tasks_collection.find_one({"task_id": task_id})
            return task
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {str(e)}")
            return None
    
    def get_recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent tasks
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            list: List of task documents
        """
        if not self.is_connected():
            logger.warning("MongoDB not connected, cannot retrieve recent tasks")
            return []
            
        try:
            tasks = list(self.tasks_collection.find().sort("created_at", -1).limit(limit))
            return tasks
        except Exception as e:
            logger.error(f"Error retrieving recent tasks: {str(e)}")
            return []
    
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Create a singleton instance
db = MongoDB()