import asyncio
from utils.utils import OpenAIPayload, OpenAIAPIRequest, openai_compatible_request, process_stream_chunk
import urllib3
import os

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

async def main(
    prompt: str = "What is the color of the sky on a clear day?",
    api_url: str = "https://api.openai.com/v1",
    model: str = "gpt-3.5-turbo",
):
    """
    Make an asynchronous streaming chat request to the OpenAI-compatible API.

    Args:
        prompt (str): The user prompt to send in the chat request.

    Returns:
        int: Total number of characters received in the response, or None if the request fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key is required. Set the OPENAI_API_KEY environment variable.")
    
    api_url = f"{api_url}/chat/completions"
    

    # Construct the payload using OpenAIPayload
    payload = OpenAIPayload(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    # Construct the request using OpenAIAPIRequest
    request = OpenAIAPIRequest(
        api_url=api_url,
        api_key=api_key,
        payload=payload
    )

    try:
        total_chars = 0
        print("Making request...")

        # Make the asynchronous streaming request
        async for chunk in await openai_compatible_request(request):
            # Process the chunk to extract content
            processed_chunk = process_stream_chunk(chunk)
            content = processed_chunk.get("content", "")
            if content:
                print(content, end='', flush=True)  # Print content in real-time
                total_chars += len(content)

        print(f"\n\nTotal characters received: {total_chars}")
        return total_chars

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv
    load_dotenv()
    fire.Fire(main)