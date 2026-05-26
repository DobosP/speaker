from utils.conversation_router import (
    ConversationRouter,
    RouteAction,
    RouteContext,
    normalize_transcript,
)


def _ctx(text: str, *, partial: bool = False) -> RouteContext:
    return RouteContext(
        transcript=text,
        is_partial=partial,
        available_capabilities=("system.time", "debug.echo"),
    )


def test_normalize_transcript_collapses_noise():
    assert normalize_transcript(" Stop, stop! ") == "stop"
    assert normalize_transcript("That's enough.") == "thats enough"


def test_stop_routes_to_stop_output_not_shutdown():
    decision = ConversationRouter().route(_ctx("stop talking"))
    assert decision.action == RouteAction.STOP_OUTPUT
    assert decision.reason == "stop_output_phrase"


def test_quit_routes_to_shutdown():
    decision = ConversationRouter().route(_ctx("quit"))
    assert decision.action == RouteAction.SHUTDOWN


def test_near_miss_started_does_not_stop():
    decision = ConversationRouter().route(_ctx("started"))
    assert decision.action == RouteAction.LLM


def test_time_request_routes_to_capability():
    decision = ConversationRouter().route(_ctx("what time is it"))
    assert decision.action == RouteAction.CAPABILITY
    assert decision.capability == "system.time"


def test_partial_only_allows_control_commands():
    router = ConversationRouter()
    assert router.route_partial(_ctx("stop", partial=True)).action == RouteAction.STOP_OUTPUT
    assert router.route_partial(_ctx("what time is it", partial=True)).action == RouteAction.IGNORE
