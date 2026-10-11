"""Opt-in local annotated preview. No frame recordings or image history."""
import time
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response


def create_preview_router(monitor, *, clock=time.monotonic):
    frames = {}

    def local_only(request: Request):
        if (request.client is None or request.client.host not in {'127.0.0.1', '::1'}
                or request.url.hostname not in {'127.0.0.1', 'localhost', '::1'}
                or request.headers.get('origin') not in {None, str(request.base_url).rstrip('/')}):
            raise HTTPException(403, 'Camera preview is available on the local care computer only')

    router = APIRouter(prefix='/camera-preview', dependencies=[Depends(local_only)])

    def prune():
        for key, (_, received) in tuple(frames.items()):
            if clock() - received > 3:
                frames.pop(key, None)

    @router.post('/{room_id}/{source}')
    async def publish(room_id: str, source: str, request: Request):
        if room_id not in monitor.config.room_ids:
            raise HTTPException(404, 'Room not configured')
        if len(source) > 100 or request.headers.get('content-type') != 'image/jpeg':
            raise HTTPException(422, 'Send an annotated JPEG frame')
        frame = bytearray()
        async for chunk in request.stream():
            frame.extend(chunk)
            if len(frame) > 2_000_000:
                raise HTTPException(413, 'Preview frame too large')
        if not frame.startswith(b'\xff\xd8') or not frame.endswith(b'\xff\xd9'):
            raise HTTPException(422, 'Invalid JPEG frame')
        prune()
        if len(frames) >= 16 and (room_id, source) not in frames:
            raise HTTPException(429, 'Too many preview sources')
        frames[room_id, source] = bytes(frame), clock()
        return {'received': True}

    @router.get('/{room_id}/{source}')
    async def latest(room_id: str, source: str):
        prune()
        item = frames.get((room_id, source))
        if item is None:
            raise HTTPException(503, 'No current camera preview. Start the camera client with --dashboard-preview.')
        return Response(item[0], media_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    @router.delete('/{room_id}/{source}')
    async def clear(room_id: str, source: str):
        frames.pop((room_id, source), None)
        return {'cleared': True}

    return router
