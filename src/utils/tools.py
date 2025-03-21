import requests
import logging
import os

logger = logging.getLogger(__name__)

def scan_url_with_jina(url):
    """
    Scan a URL using Jina AI's reader service and return the extracted content.
    
    Args:
        url (str): The URL to scan
        api_key (str, optional): Jina API key. If not provided, uses the default key.
        
    Returns:
        str: The extracted content in markdown format
    """
    api_key = os.environ.get("JINA_API_KEY", "jina_10f654635d4f494cac364015f93a91e8HtmMOkmrYcH_SebdNsI3_lhyVJBx")
    
    headers = {
        'Accept': 'text/event-stream',
        'Authorization': f'Bearer {api_key}',
        'X-Engine': 'browser',
        'X-Respond-With': 'readerlm-v2',
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