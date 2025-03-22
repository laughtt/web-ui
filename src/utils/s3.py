import boto3
import os
from typing import List, Optional, Union, Tuple
from botocore.exceptions import ClientError
import logging
import os

class S3FileHandler:
    """
    A class to handle S3 file operations that makes S3 files appear as local files to an LLM.
    Provides methods for reading, writing, and finding files in an S3 bucket.
    """
    
    def __init__(self, bucket_name: str, prefix: str = "", region: str = "", 
                 aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None,
                 make_public: bool = True):
        """
        Initialize S3FileHandler with bucket and optional credentials.
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: Optional prefix path within the bucket (folder-like structure)
            region: AWS region for the S3 bucket
            aws_access_key_id: Optional AWS access key ID
            aws_secret_access_key: Optional AWS secret access key
            make_public: If True, all files uploaded will be made public by default
        """
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip('/') + '/' if prefix else ""
        self.make_public = make_public
        
        # Initialize S3 client
        self.s3 = boto3.client(
            's3',
            region_name=os.environ.get('AWS_REGION'),
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_full_path(self, file_path: str) -> str:
        """
        Constructs the full S3 key path from the given file path.
        
        Args:
            file_path: The relative file path
            
        Returns:
            The full S3 key including prefix
        """
        # Remove leading slash if present
        file_path = file_path.lstrip('/')
        return f"{self.prefix}{file_path}"
    
    def file_read(self, file_path: str, start_line: Optional[int] = None, 
                 end_line: Optional[int] = None) -> str:
        """
        Read content from a file in S3 with optional line range specification.
        
        Args:
            file_path: Path to the file in S3
            start_line: Optional line number to start reading from (0-indexed)
            end_line: Optional line number to stop reading at (inclusive)
            
        Returns:
            Content of the file as a string
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other S3 errors
        """
        full_path = self._get_full_path(file_path)
        
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=full_path)
            content = response['Body'].read().decode('utf-8')
            
            # If line range is specified, filter content
            if start_line is not None or end_line is not None:
                lines = content.splitlines()
                start = start_line if start_line is not None else 0
                end = end_line if end_line is not None else len(lines)
                
                # Adjust for 0-indexing and inclusivity
                content = '\n'.join(lines[start:end+1])
                
            return content
                
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"File not found: {file_path}")
            else:
                self.logger.error(f"Error reading file {file_path}: {str(e)}")
                raise Exception(f"Error reading file: {str(e)}")
    
    def file_write(self, file_path: str, content: str, append: bool = False, public: Optional[bool] = None) -> str:
        """
        Write content to a file in S3, either creating a new file, overwriting, or appending.
        
        Args:
            file_path: Path to the file in S3
            content: Content to write to the file
            append: If True, append to existing file; otherwise overwrite
            public: If True, make the file publicly accessible. If None, use the class default setting.
            
        Returns:
            The full S3 URL of the written file
            
        Raises:
            Exception: For S3 errors
        """
        full_path = self._get_full_path(file_path)
        
        try:
            if append:
                # If appending, first read existing content
                try:
                    existing_content = self.file_read(file_path)
                    content = existing_content + content
                except FileNotFoundError:
                    # If file doesn't exist, just create it
                    pass
            
            # Determine if file should be public
            make_public = self.make_public if public is None else public
            
            # Set up parameters for S3 put_object
            params = {
                'Bucket': self.bucket_name,
                'Key': full_path,
                'Body': content.encode('utf-8')
            }
            
            # Add ACL for public access if requested
            if make_public:
                params['ACL'] = 'public-read'
            
            # Write to S3
            self.s3.put_object(**params)
            
            return self.get_public_url(file_path)
            
        except Exception as e:
            self.logger.error(f"Error writing to file {file_path}: {str(e)}")
            raise Exception(f"Error writing to file: {str(e)}")
    
    def upload_local_file_to_s3(self, local_file_path: str, s3_file_path: Optional[str] = None, 
                               content_type: Optional[str] = None, public: Optional[bool] = None) -> str:
        """
        Upload a local file to S3.
        
        Args:
            local_file_path: Path to the local file to upload
            s3_file_path: Optional destination path in S3. If not provided, uses the filename from local_file_path
            content_type: Optional MIME type of the file. If not provided, will be guessed from the file extension
            public: If True, make the file publicly accessible. If None, use the class default setting
            
        Returns:
            The full S3 URL of the uploaded file
            
        Raises:
            FileNotFoundError: If the local file doesn't exist
            Exception: For S3 errors
        """
        # Check if local file exists
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")
        
        # Use filename as destination if s3_file_path not provided
        if s3_file_path is None:
            s3_file_path = os.path.basename(local_file_path)
        
        full_path = self._get_full_path(s3_file_path)
        
        # Determine if file should be public
        make_public = self.make_public if public is None else public
        
        # Guess content type if not provided
        if content_type is None:
            import mimetypes
            content_type, _ = mimetypes.guess_type(local_file_path)
        
        try:
            # Set up parameters for S3 upload_file
            extra_args = {}
            
            # Add content type if available
            if content_type:
                extra_args['ContentType'] = content_type
                
            # Add ACL for public access if requested
            if make_public:
                extra_args['ACL'] = 'public-read'
            
            # Upload to S3
            self.s3.upload_file(
                local_file_path, 
                self.bucket_name, 
                full_path,
                ExtraArgs=extra_args
            )
            
            self.logger.info(f"Successfully uploaded {local_file_path} to s3://{self.bucket_name}/{full_path}")
            return self.get_public_url(s3_file_path)
            
        except Exception as e:
            self.logger.error(f"Error uploading file {local_file_path}: {str(e)}")
            raise Exception(f"Error uploading file: {str(e)}")
    
    def make_file_public(self, file_path: str) -> str:
        """
        Make an existing file in S3 publicly accessible.
        
        Args:
            file_path: Path to the file in S3
            
        Returns:
            The full S3 URL of the file made public
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other S3 errors
        """
        full_path = self._get_full_path(file_path)
        
        try:
            # Check if file exists first
            self.s3.head_object(Bucket=self.bucket_name, Key=full_path)
            
            # Set public-read ACL
            self.s3.put_object_acl(
                Bucket=self.bucket_name,
                Key=full_path,
                ACL='public-read'
            )
            
            return self.get_public_url(file_path)
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                raise FileNotFoundError(f"File not found: {file_path}")
            else:
                self.logger.error(f"Error making file public {file_path}: {str(e)}")
                raise Exception(f"Error making file public: {str(e)}")
    
    def get_public_url(self, file_path: str) -> str:
        """
        Get the public URL for a file in S3.
        
        Args:
            file_path: Path to the file in S3
            
        Returns:
            Public URL for the file
        """
        full_path = self._get_full_path(file_path)
        
        # Generate the URL
        return f"https://{self.bucket_name}.s3.amazonaws.com/{full_path}"
    
    def find_file_by_name(self, file_name: str, subfolder: str = None) -> List[str]:
        """
        Find files in the S3 bucket that match the provided name.
        
        Args:
            file_name: Name or partial name of the file to find
            subfolder: Optional subfolder to search within
            
        Returns:
            List of file paths that match the search criteria
            
        Raises:
            Exception: For S3 errors
        """
        search_prefix = self.prefix
        if subfolder:
            subfolder = subfolder.strip('/')
            search_prefix = f"{self.prefix}{subfolder}/"
            
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=search_prefix
            )
            
            matching_files = []
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    # Extract just the filename from the full path
                    _, filename = os.path.split(key)
                    
                    if file_name in filename:
                        # Return path relative to prefix
                        relative_path = key
                        if key.startswith(self.prefix):
                            relative_path = key[len(self.prefix):]
                        matching_files.append(relative_path)
            
            return matching_files
            
        except Exception as e:
            self.logger.error(f"Error finding files with name {file_name}: {str(e)}")
            raise Exception(f"Error searching for files: {str(e)}")
    
    def list_files(self, subfolder: str = None) -> List[str]:
        """
        List all files in the S3 bucket/prefix or specific subfolder.
        
        Args:
            subfolder: Optional subfolder to list files from
            
        Returns:
            List of file paths
            
        Raises:
            Exception: For S3 errors
        """
        search_prefix = self.prefix
        if subfolder:
            subfolder = subfolder.strip('/')
            search_prefix = f"{self.prefix}{subfolder}/"
            
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=search_prefix
            )
            
            files = []
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    # Return path relative to prefix
                    relative_path = key
                    if key.startswith(self.prefix):
                        relative_path = key[len(self.prefix):]
                    files.append(relative_path)
            
            return files
            
        except Exception as e:
            self.logger.error(f"Error listing files: {str(e)}")
            raise Exception(f"Error listing files: {str(e)}")

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the S3 bucket.
        
        Args:
            file_path: Path to the file in S3
            
        Returns:
            True if the file exists, False otherwise
        """
        full_path = self._get_full_path(file_path)
        
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=full_path)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                self.logger.error(f"Error checking if file exists {file_path}: {str(e)}")
                raise Exception(f"Error checking if file exists: {str(e)}")