from typing import Dict, Any, Optional, List
from pymongo import MongoClient
import os
import logging
from datetime import datetime

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
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb://admin:password123@localhost:27017")
            mongodb_database = os.getenv("MONGODB_DATABASE", "browseruse")
            
            # Get authentication credentials if provided
            username = os.getenv("MONGO_ROOT_USERNAME", "admin")
            password = os.getenv("MONGO_ROOT_PASSWORD", "password123")
            
            # Create MongoDB client
            self.client = MongoClient(mongodb_uri, 
                                     username=username, 
                                     password=password,
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
                "config": config,
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
                update_doc["result"] = result
                
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