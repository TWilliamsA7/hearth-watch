"""Current, consent-filtered display identities, independent of simulation admission."""
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hearth.monitor.detector import source_key


class DisplayIdentity(BaseModel):
    room_id: str
    source_type: str
    source: str
    scenario_id: str | None
    run_id: str | None
    person_id: str | None
    timestamp: datetime
    recognized: bool
    name: str | None


def create_identity_router(monitor) -> APIRouter:
    router = APIRouter(tags=["monitoring"])

    @router.get("/rooms/{room_id}/identities", response_model=list[DisplayIdentity])
    async def identities(room_id: str):
        if room_id not in monitor.config.room_ids:
            raise HTTPException(404, "Room not found")
        monitor.check_health()
        if monitor.identity_of is None:
            return []
        observations = [o for o in monitor.observations.values() if o.room_id == room_id]
        newest = {}
        for obs in observations:
            key = source_key(obs)
            newest[key] = max(newest.get(key, obs.timestamp), obs.timestamp)
        out = []
        for obs in observations:
            key = source_key(obs)
            health = monitor.health.get(key)
            if (not obs.person_detected or health is None or not health.status.online
                    or health.status.observation_stale
                    or (newest[key] - obs.timestamp).total_seconds() > monitor.config.max_observation_gap_seconds):
                continue
            identity = monitor.identity_of(obs)
            recognized = bool(identity.get("recognized"))
            if recognized:
                out.append(DisplayIdentity(
                    room_id=obs.room_id, source_type=obs.source_type, source=obs.source,
                    scenario_id=obs.scenario_id, run_id=obs.run_id, person_id=obs.person_id,
                    timestamp=obs.timestamp, recognized=True, name=identity.get("name")))
        return out

    return router
