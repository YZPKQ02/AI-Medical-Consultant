from __future__ import annotations

import json

from fastapi import WebSocket, WebSocketDisconnect

from app.services.runtime import consultation_service


async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    consultation = consultation_service.create_consultation(chief_complaint="websocket chat")

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                payload = json.loads(raw_message)
                content = str(payload.get("content", ""))
            except json.JSONDecodeError:
                content = raw_message

            result = consultation_service.add_user_message(consultation["id"], content)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return
