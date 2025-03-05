import json
import asyncio
from fastapi import FastAPI, WebSocket
from typing import Dict

app = FastAPI()
connections: Dict[str, WebSocket] = {}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = websocket

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            target_id = data["target"]
            if target_id in connections:
                await connections[target_id].send_text(json.dumps(data))
    except:
        del connections[client_id]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)