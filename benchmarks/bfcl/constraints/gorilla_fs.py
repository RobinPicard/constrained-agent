"""Parameter constraints for GorillaFileSystem tools.

Business logic:
- After find returns file paths, constrain grep/cat/tail/sort file_name
  to the found files.
- After grep returns matches, constrain tail/sort to the searched file.
"""

from __future__ import annotations

from constrained_agent import Session, ToolRegistry


def gorilla_fs_constraints(session: Session, registry: ToolRegistry) -> None:
    # After find, constrain file_name on grep/cat/tail/sort
    if session.tool("find").has_run:
        result = session.tool("find").last_result
        if isinstance(result, dict) and "matches" in result:
            matches = result["matches"]
            if isinstance(matches, list) and matches:
                for tool_name in ("grep", "cat", "tail", "sort"):
                    try:
                        if "file_name" in registry[tool_name].params:
                            registry[tool_name].params["file_name"].enum = matches
                    except KeyError:
                        pass
