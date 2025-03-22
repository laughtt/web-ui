# Using relative imports to go up one level and import from src
import sys
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
url1 = s3_handler.file_write("README.md", "This is new content")
print(f"File written, URL: {url1}")

# Read a file
content = s3_handler.file_read("README.md")
print(f"File content: {content}")

# Read specific lines
intro = s3_handler.file_read("README.md", start_line=0, end_line=10)
print(f"First 10 lines: {intro}")

# Write to a file
url2 = s3_handler.file_write("README.md", "This is new content 2")
print(f"File updated, URL: {url2}")

# Append to a file
url3 = s3_handler.file_write("README.md", "New log entry\n", append=True)
print(f"File appended, URL: {url3}")

files = s3_handler.list_files()
print(f"Files in bucket: {files}")

# Find files by name
matching_files = s3_handler.find_file_by_name("README.md")
print(f"Matching files: {matching_files}")

