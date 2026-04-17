"""Parameter constraints for VehicleControlAPI tools.

Business logic:
- startEngine requires pressBrakePedal to have been called first.
- setCruiseControl requires startEngine to have been called first.
- After estimate_distance returns a distance, estimate_drive_feasibility_by_mileage
  distance is constrained to that value.
"""

from __future__ import annotations

from constrained_agent import Session, ToolRegistry


def vehicle_control_constraints(session: Session, registry: ToolRegistry) -> None:
    # pressBrakePedal must precede startEngine
    if not session.tool("pressBrakePedal").has_run:
        try:
            registry["startEngine"].available = False
            registry["startEngine"].unavailable_reason = (
                "Call 'pressBrakePedal' first — the brake must be engaged to start the engine"
            )
        except KeyError:
            pass

    # startEngine must precede setCruiseControl
    if not session.tool("startEngine").has_run:
        try:
            registry["setCruiseControl"].available = False
            registry["setCruiseControl"].unavailable_reason = (
                "Call 'startEngine' first — the engine must be running to set cruise control"
            )
        except KeyError:
            pass

    # After estimate_distance, constrain distance on feasibility check
    if session.tool("estimate_distance").has_run:
        result = session.tool("estimate_distance").last_result
        if isinstance(result, dict) and "distance" in result:
            dist = result["distance"]
            try:
                if "distance" in registry["estimate_drive_feasibility_by_mileage"].params:
                    registry["estimate_drive_feasibility_by_mileage"].params["distance"].enum = [dist]
            except KeyError:
                pass
