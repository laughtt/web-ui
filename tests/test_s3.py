# Using relative imports to go up one level and import from src
import sys
import os
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

print(project_root)
# Initialize the handler
from src.utils.s3 import S3FileHandler

s3_handler = S3FileHandler(
    bucket_name="test-brownser-use",
    prefix="llm-workspace",
    region="us-east-2"
)

# Write to a file
s3_handler.file_write("README.md", "This is new content")

# Read a file
content = s3_handler.file_read("README.md")

# Read specific lines
intro = s3_handler.file_read("README.md", start_line=0, end_line=10)

# Write to a file
s3_handler.file_write("README.md", "This is new content 2")

# Append to a file
s3_handler.file_write("README.md", "New log entry\n", append=True)

print(s3_handler.list_files())
# Find files by name
matching_files = s3_handler.find_file_by_name("README.md")

