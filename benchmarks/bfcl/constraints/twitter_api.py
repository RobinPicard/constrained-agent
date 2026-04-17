"""Parameter constraints for TwitterAPI tools.

Business logic:
- authenticate_twitter must be called before post_tweet, retweet, comment,
  follow_user, unfollow_user (auth required for write actions).
- After post_tweet, comment and retweet are constrained to the returned
  tweet_id.
"""

from __future__ import annotations

from constrained_agent import Session, ToolRegistry


_WRITE_TOOLS = ("post_tweet", "retweet", "comment", "follow_user", "unfollow_user")


def twitter_api_constraints(session: Session, registry: ToolRegistry) -> None:
    # authenticate_twitter must precede write actions
    if not session.tool("authenticate_twitter").has_run:
        for tool_name in _WRITE_TOOLS:
            try:
                registry[tool_name].available = False
                registry[tool_name].unavailable_reason = (
                    "Call 'authenticate_twitter' first to log in before using this tool"
                )
            except KeyError:
                pass

    # After post_tweet, constrain tweet_id on comment and retweet
    if session.tool("post_tweet").has_run:
        result = session.tool("post_tweet").last_result
        if isinstance(result, dict) and "id" in result:
            tid = result["id"]
            for tool_name in ("comment", "retweet"):
                try:
                    if "tweet_id" in registry[tool_name].params:
                        registry[tool_name].params["tweet_id"].enum = [tid]
                except KeyError:
                    pass
