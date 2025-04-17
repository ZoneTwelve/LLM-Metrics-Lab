# utils.py
import time
import json
import asyncio
import aiohttp
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union, AsyncGenerator, Callable

# TODO: Consider relocating all the classes to utils/protocol.py or another suitable file

class OpenAIPayload(BaseModel):
    """
    Class for OpenAI API request payload.
    """
    model: str = Field(..., description="The model to use, e.g., gpt-3.5-turbo")
    messages: Optional[List[Dict[str, str]]] = Field(
        None, description="List of messages for chat completions"
    )
    prompt: Optional[str] = Field(None, description="Prompt for text completions")
    stream: bool = Field(False, description="Whether to stream the response")
    n: int = Field(1, description="Number of chat completion choices to generate. Only 1 is supported.")
    max_new_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate (maps to max_tokens)"
    )
    do_sample: bool = Field(
        False, description="Whether to use sampling (affects temperature/top_p)"
    )
    top_p: Optional[float] = Field(
        None, description="Nucleus sampling probability mass"
    )
    top_k: Optional[int] = Field(
        None, description="Top-k tokens to consider (not directly supported by OpenAI)"
    )
    min_p: Optional[float] = Field(
        None, description="Minimum token probability (not directly supported by OpenAI)"
    )
    temperature: Optional[float] = Field(
        None, description="Sampling temperature"
    )
    repetition_penalty: Optional[float] = Field(
        None, description="Penalty to discourage repetition (not directly supported by OpenAI)"
    )
    ignore_eos: Optional[bool] = Field(
        None, description="Whether to ignore EOS token (not directly supported by OpenAI)"
    )
    random_seed: Optional[int] = Field(
        None, description="Seed for sampling (not directly supported by OpenAI)"
    )
    stop_words: Optional[List[str]] = Field(
        None, description="Words to stop generation"
    )
    bad_words: Optional[List[str]] = Field(
        None, description="Words to never generate (not directly supported by OpenAI)"
    )
    stop_token_ids: Optional[List[int]] = Field(
        None, description="Token IDs to stop generation (maps to stop)"
    )
    bad_token_ids: Optional[List[int]] = Field(
        None, description="Token IDs to never generate (not directly supported by OpenAI)"
    )
    min_new_tokens: Optional[int] = Field(
        None, description="Minimum number of tokens to generate (not directly supported by OpenAI)"
    )
    skip_special_tokens: bool = Field(
        True, description="Whether to remove special tokens in decoding"
    )
    logprobs: Optional[int] = Field(
        None, description="Number of log probabilities to return per output token"
    )
    response_format: Optional[Dict[str, Any]] = Field(
        None, description="Response format, e.g., JSON schema or regex schema"
    )
    logits_processors: Optional[List[Callable]] = Field(
        None, description="Custom logit processors (not directly supported by OpenAI)"
    )

    def to_openai_dict(self) -> Dict[str, Any]:
        """
        Converts the payload to an OpenAI-compatible dictionary.
        """
        payload = {"model": self.model, "stream": self.stream}
        if self.messages is not None:
            payload["messages"] = self.messages
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        if self.max_new_tokens is not None:
            payload["max_tokens"] = self.max_new_tokens
        if self.n != 1:
            payload["n"] = self.n
        if self.do_sample:
            payload["temperature"] = self.temperature or 1.0
            if self.top_p is not None:
                payload["top_p"] = self.top_p
        else:
            payload["temperature"] = self.temperature or 0.0
        if self.stop_words is not None or self.stop_token_ids is not None:
            stop = []
            if self.stop_words:
                stop.extend(self.stop_words)
            if self.stop_token_ids:
                stop.extend([str(id) for id in self.stop_token_ids])
            if stop:
                payload["stop"] = stop
        if self.logprobs is not None:
            payload["logprobs"] = self.logprobs
        if self.response_format is not None:
            payload["response_format"] = self.response_format
        return payload

class ChoiceDelta(BaseModel):
    """
    Delta content for a choice in a streaming chat completion chunk.
    """
    role: Optional[str] = Field(None, description="Role of the message, e.g., assistant")
    content: Optional[str] = Field(None, description="Content of the message delta")

class Choice(BaseModel):
    """
    Choice in a chat completion chunk or response.
    """
    index: int = Field(..., description="Index of the choice")
    delta: Optional[ChoiceDelta] = Field(None, description="Delta for streaming chunks")
    message: Optional[Dict[str, str]] = Field(None, description="Message for non-streaming responses")
    logprobs: Optional[Any] = Field(None, description="Log probabilities, if requested")
    finish_reason: Optional[str] = Field(None, description="Reason for finishing, e.g., stop")

class Usage(BaseModel):
    """
    Usage statistics for the completion.
    """
    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    total_tokens: int = Field(..., description="Total number of tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")

class ChatCompletionChunk(BaseModel):
    """
    Model for a streaming chat completion chunk.
    """
    id: str = Field(..., description="Unique identifier for the chunk")
    object: str = Field(..., description="Object type, e.g., chat.completion.chunk")
    created: int = Field(..., description="Timestamp of creation")
    model: str = Field(..., description="Model used")
    choices: List[Choice] = Field(..., description="List of choices")
    usage: Optional[Usage] = Field(None, description="Usage statistics, if provided")

class ChatCompletionResponse(BaseModel):
    """
    Model for a non-streaming chat completion response.
    """
    id: str = Field(..., description="Unique identifier for the response")
    object: str = Field(..., description="Object type, e.g., chat.completion")
    created: int = Field(..., description="Timestamp of creation")
    model: str = Field(..., description="Model used")
    choices: List[Choice] = Field(..., description="List of choices")
    usage: Usage = Field(..., description="Usage statistics")

class TextCompletionChunk(BaseModel):
    """
    Model for a streaming text completion chunk.
    """
    id: str = Field(..., description="Unique identifier for the chunk")
    object: str = Field(..., description="Object type, e.g., text_completion.chunk")
    created: int = Field(..., description="Timestamp of creation")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="List of choices with text and finish_reason")
    usage: Optional[Usage] = Field(None, description="Usage statistics, if provided")

class TextCompletionResponse(BaseModel):
    """
    Model for a non-streaming text completion response.
    """
    id: str = Field(..., description="Unique identifier for the response")
    object: str = Field(..., description="Object type, e.g., text_completion")
    created: int = Field(..., description="Timestamp of creation")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="List of choices with text")
    usage: Usage = Field(..., description="Usage statistics")

class OpenAIAPIRequest(BaseModel):
    """
    Base class for OpenAI API requests.
    """
    api_url: str = Field(
        ..., description="The URL of the OpenAI API endpoint."
    )
    api_key: str = Field(
        ..., description="The API key for authentication."
    )
    payload: OpenAIPayload = Field(
        ..., description="The payload to be sent in the request."
    )

class OpenAIAPIResponse(BaseModel):
    """
    Base class for OpenAI API responses.
    """
    status_code: int = Field(
        ..., description="HTTP status code of the response."
    )
    content: Optional[Union[ChatCompletionResponse, TextCompletionResponse]] = Field(
        None, description="Content of the response."
    )
    error: Optional[str] = Field(
        None, description="Error message if any occurred."
    )
    timing: Optional[float] = Field(
        None, description="Total time taken for the request in seconds."
    )

class OpenAIAPIStreamChunk(BaseModel):
    """
    Class for individual chunks in streaming responses.
    """
    chunk: Union[ChatCompletionChunk, TextCompletionChunk] = Field(
        ..., description="The data chunk received from the streaming response."
    )
    timestamp: float = Field(
        ..., description="Timestamp when the chunk was received, in seconds since epoch."
    )

async def openai_compatible_request_non_streaming(
    request: OpenAIAPIRequest
) -> OpenAIAPIResponse:
    """
    Makes an asynchronous non-streaming request to the OpenAI API.
    """
    api_url = request.api_url
    api_key = request.api_key
    payload = request.payload.to_openai_dict()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        async with session.post(api_url, headers=headers, json=payload) as response:
            end_time = time.time()
            timing = end_time - start_time
            if response.status == 200:
                content = await response.json()
                response_model = (
                    ChatCompletionResponse(**content)
                    if "chat/completions" in api_url
                    else TextCompletionResponse(**content)
                )
                return OpenAIAPIResponse(status_code=200, content=response_model, timing=timing)
            else:
                error_text = await response.text()
                return OpenAIAPIResponse(
                    status_code=response.status, error=error_text, timing=timing
                )

async def openai_compatible_request_streaming(
    request: OpenAIAPIRequest
) -> AsyncGenerator[OpenAIAPIStreamChunk, None]:
    """
    Makes an asynchronous streaming request to the OpenAI API.
    """
    api_url = request.api_url
    api_key = request.api_key
    payload = request.payload.to_openai_dict()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=payload) as response:
            if response.status == 200:
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(data)
                            chunk_model = (
                                ChatCompletionChunk(**chunk_data)
                                if "chat/completions" in api_url
                                else TextCompletionChunk(**chunk_data)
                            )
                            timestamp = time.time()
                            yield OpenAIAPIStreamChunk(chunk=chunk_model, timestamp=timestamp)
                        except json.JSONDecodeError:
                            # Skip malformed chunks silently
                            pass
            else:
                error_text = await response.text()
                raise Exception(f"API request failed with status code {response.status}: {error_text}")

async def openai_compatible_request(
    request: OpenAIAPIRequest
) -> Union[OpenAIAPIResponse, AsyncGenerator[OpenAIAPIStreamChunk, None]]:
    """
    Wrapper for making asynchronous requests to the OpenAI API.
    """
    if request.payload.stream:
        return openai_compatible_request_streaming(request)
    else:
        return await openai_compatible_request_non_streaming(request)

def process_stream_chunk(chunk: OpenAIAPIStreamChunk) -> Dict[str, Any]:
    """
    Process a streaming chunk to extract relevant data for logging.
    """
    try:
        data = chunk.chunk.dict()
        content = data["choices"][0]["delta"]["content"] if data["choices"][0]["delta"] else ""
        return {
            "data": data,
            "timestamp": chunk.timestamp,
            "content": content
        }
    except Exception as e:
        return {"error": str(e), "timestamp": chunk.timestamp}

async def main(
    api_url: str = "https://api.openai.com/v1",
    model: str = "gpt-3.5-turbo",
):
    """
    Example usage of async OpenAI API requests.
    """
    import os
    from dotenv import load_dotenv
    import tqdm
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Non-streaming chat completion
    chat_payload = OpenAIPayload(
        model=model,
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    chat_request = OpenAIAPIRequest(
        api_url=f"{api_url}/chat/completions",
        api_key=API_KEY,
        payload=chat_payload
    )
    
    response = await openai_compatible_request(chat_request)
    print(f"Chat Completion Response: {response.content}, Time: {response.timing}s")
    print(f"Stop Reason: {response.content.choices[0].finish_reason}")
    
    # Streaming text completion
    text_payload = OpenAIPayload(
        model=model,
        prompt="Once upon a time",
        stream=True,
        max_new_tokens=50,
        stop_words=["the end"]
    )
    text_request = OpenAIAPIRequest(
        api_url=f"{api_url}/completions",
        api_key=API_KEY,
        payload=text_payload
    )
    start_time = time.time()

    pbar = tqdm.tqdm(total=0, unit="chunk", desc="Streaming Text Completion")
    completion_response_str = ""
    async for chunk in await openai_compatible_request(text_request):
        elapsed = chunk.timestamp - start_time
        pbar.update(1)
        content = chunk.chunk.choices[0]["text"]
        pbar.set_postfix_str(f"Elapsed: {elapsed:.2f}s; Output: `{content}`")
        completion_response_str += content
    pbar.close()
    print(f"Streaming Text Completion Response: {completion_response_str}")
    print(f"Stop Reason: {chunk.chunk.choices[0]['finish_reason']}")
    
if __name__ == "__main__":
    import fire
    fire.Fire(main)