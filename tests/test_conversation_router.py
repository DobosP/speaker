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


# -- action-brain trigger routing -------------------------------------------

def _agent_ctx(text: str) -> RouteContext:
    return RouteContext(
        transcript=text,
        available_capabilities=("system.time", "debug.echo", "agent.execute"),
    )


def _agent_router() -> ConversationRouter:
    return ConversationRouter(agent_trigger_phrases=("computer", "hey computer", "agent"))


def test_agent_trigger_routes_to_agent_execute_keeping_case():
    decision = _agent_router().route(_agent_ctx("Computer, list my Downloads folder."))
    assert decision.action == RouteAction.CAPABILITY
    assert decision.capability == "agent.execute"
    # original case + punctuation preserved for the action brain
    assert decision.payload["instruction"] == "list my Downloads folder."


def test_agent_multiword_trigger_strips_only_the_trigger():
    decision = _agent_router().route(_agent_ctx("Hey computer, what's my disk usage?"))
    assert decision.capability == "agent.execute"
    assert decision.payload["instruction"] == "what's my disk usage?"


def test_bare_trigger_falls_through_to_llm():
    assert _agent_router().route(_agent_ctx("computer")).action == RouteAction.LLM


def test_non_trigger_utterance_is_unaffected():
    assert _agent_router().route(_agent_ctx("what is the capital of france")).action == RouteAction.LLM


def test_trigger_ignored_when_agent_capability_unavailable():
    ctx = RouteContext(transcript="computer, list files", available_capabilities=("system.time",))
    assert _agent_router().route(ctx).action == RouteAction.LLM


def test_no_configured_trigger_means_no_agent_routing():
    # default router has no agent trigger phrases -> behaves exactly as before
    assert ConversationRouter().route(_agent_ctx("computer, list files")).action == RouteAction.LLM


def test_stop_still_wins_over_agent_trigger():
    # control phrases are checked before agent routing
    assert _agent_router().route(_agent_ctx("stop")).action == RouteAction.STOP_OUTPUT
