import requests
import logging
import os
import asyncio
import base64
from typing import Optional
import instructor
import google.generativeai as genai
from pydantic import BaseModel, Field


from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class ChatContextSchema(BaseModel):
    name_user_chat: str = Field(description="The name of the user is talking with")
    context_user_chat: str = Field(description="The context of the chat")
    messages: list[str] = Field(description="The messages of the chat")


client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="models/gemini-2.5-pro-exp-03-25"
    ),
    mode=instructor.Mode.GEMINI_JSON

def scan_url_with_jina(url):
    """
    Scan a URL using Jina AI's reader service and return the extracted content.
    
    Args:
        url (str): The URL to scan
        
    Returns:
        str: The extracted content in markdown format
    """
    api_key = os.environ.get("JINA_API_KEY", "jina_10f654635d4f494cac364015f93a91e8HtmMOkmrYcH_SebdNsI3_lhyVJBx")
    
    headers = {
        'Accept': 'text/event-stream',
        'Authorization': f'Bearer {api_key}',
        'X-Engine': 'browser',
        'X-Return-Format': 'markdown'
    }
    
    try:
        jina_url = f'https://r.jina.ai/{url}'
        logger.info(f"Scanning URL with Jina: {url}")
        response = requests.get(jina_url, headers=headers)
        
        if response.status_code == 200:
            return response.text
        else:
            error_msg = f"Jina API returned status code {response.status_code}: {response.text}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
    
    except Exception as e:
        error_msg = f"Error scanning URL with Jina: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}"


async def screenshot_url_analysis(screenshot: str) -> ChatContextSchema:
    """
    Analyze a screenshot of a webpage using Gemini's image analysis capabilities.
    
    Args:
        screenshot: Base64-encoded screenshot image
    
    Returns:
        Analysis of the screenshot content
    """
    # Get API key from environment variable or use a placeholder
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    
    try:
        # Convert base64 string to bytes
        image_bytes = base64.b64decode(screenshot)
        

        system_prompt = """
        You are SOCIAL MEDIA MANAGER, you are talking with a user, describe the context of the conversation in detail.
        - Do not include the header, footer, sidebar, etc.
        """
        # Create the prompt for image analysis
        prompt = """
        Describe chat context from the screenshot, if there is no chat context, return "no chat context"
        This can be usually a chat messages, return the context of the conversation , this will be saved to respond the user from the chat history.
        - Do not include the header, footer, sidebar, etc.
        """
        analysis = await client.chat.completions.create(
                response_model=ChatContextSchema,
                contents=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [prompt,image_bytes]}
                ]
            )

        logger.info(f"Screenshot analysis: {response.text}")
        # Return the analysis text
        return analysis
    except Exception as e:
        return f"Error analyzing screenshot: {str(e)}"
    