import os
import json
import asyncio
import uuid
from dotenv import load_dotenv
import websockets
from datasets import load_dataset
from utils.url import normalize_url

class Template(str):
    pass

class Conversation(str):
    pass

load_dotenv()

def load_dataset_as_questions(dataset_name: str, key: Template | Conversation):
    dataset = load_dataset(dataset_name)['train']
    questions = []
    if isinstance(key, Template):
        for row in dataset:
            prompt = key.format(**row)
            questions.append(prompt)
    elif isinstance(key, Conversation):
        for row in dataset:
            messages = json.loads(row[key])
            # for simplicity, join conversation turns into one prompt
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            questions.append(text)
    else:
        raise ValueError("Invalid key type")
    return questions

async def websocket_handler(websocket):
    async for message in websocket:
        data = json.loads(message)
        if data.get("command") != "start":
            await websocket.send(json.dumps({"error": "Unknown command"}))
            continue

        params = data.get("params", {})
        api_url = params.get("api_url")
        dataset_name = params.get("dataset")
        tpl = params.get("template")
        conv = params.get("conversation")

        if not api_url or not (tpl or conv):
            await websocket.send(json.dumps({"error": "api_url and template/conversation required"}))
            continue

        api_url = normalize_url(api_url)
        # load questions
        if tpl:
            questions = load_dataset_as_questions(dataset_name, Template(tpl))
        else:
            questions = load_dataset_as_questions(dataset_name, Conversation(conv))

        # package tasks
        tasks = []
        for idx, prompt in enumerate(questions):
            tasks.append({
                "id": idx,
                "model": params.get("model", "gpt-3.5-turbo"),
                "api_url": api_url,
                "prompt": prompt
            })

        await websocket.send(json.dumps({"status": "started", "tasks": tasks}))

async def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8765))
    server = await websockets.serve(websocket_handler, host, port)
    print(f"WebSocket server running at ws://{host}:{port}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
