import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from pr_agent.config_loader import get_settings
from pr_agent.tools.pr_reviewer import PRReviewer


def _make_reviewer(git_provider=None):
    reviewer = PRReviewer.__new__(PRReviewer)
    reviewer.git_provider = git_provider or MagicMock()
    reviewer.pr_url = "https://example/pr/1"
    return reviewer


def test_should_publish_review_no_suggestions_respects_config():
    reviewer = _make_reviewer()
    settings = get_settings()
    original_publish_no_suggestions = settings.pr_reviewer.publish_output_no_suggestions

    try:
        settings.pr_reviewer.publish_output_no_suggestions = False
        assert reviewer._should_publish_review_no_suggestions("No major issues detected") is False
        assert reviewer._should_publish_review_no_suggestions("A major issue was detected") is True

        settings.pr_reviewer.publish_output_no_suggestions = True
        assert reviewer._should_publish_review_no_suggestions("No major issues detected") is True
    finally:
        settings.pr_reviewer.publish_output_no_suggestions = original_publish_no_suggestions


def test_can_run_incremental_review_skips_auto_mode_without_new_commit():
    reviewer = _make_reviewer()
    reviewer.is_auto = True
    reviewer.incremental = SimpleNamespace(first_new_commit_sha=None)

    assert reviewer._can_run_incremental_review() is False


def test_set_review_labels_replaces_stale_review_labels_and_keeps_user_labels():
    settings = get_settings()
    original = {
        "publish_output": settings.config.publish_output,
        "require_estimate_effort_to_review": settings.pr_reviewer.require_estimate_effort_to_review,
        "require_security_review": settings.pr_reviewer.require_security_review,
        "enable_review_labels_effort": settings.pr_reviewer.enable_review_labels_effort,
        "enable_review_labels_security": settings.pr_reviewer.enable_review_labels_security,
    }
    settings.config.publish_output = True
    settings.pr_reviewer.require_estimate_effort_to_review = True
    settings.pr_reviewer.require_security_review = True
    settings.pr_reviewer.enable_review_labels_effort = True
    settings.pr_reviewer.enable_review_labels_security = True
    git_provider = MagicMock()
    git_provider.get_pr_labels.return_value = ["Review effort 1/5", "Possible security concern", "keep-me"]
    reviewer = _make_reviewer(git_provider)
    data = {
        "review": {
            "estimated_effort_to_review_[1-5]": "3, moderate",
            "security_concerns": "yes",
        }
    }

    try:
        reviewer.set_review_labels(data)

        git_provider.publish_labels.assert_called_once_with([
            "Review effort 3/5",
            "Possible security concern",
            "keep-me",
        ])
    finally:
        settings.config.publish_output = original["publish_output"]
        settings.pr_reviewer.require_estimate_effort_to_review = original["require_estimate_effort_to_review"]
        settings.pr_reviewer.require_security_review = original["require_security_review"]
        settings.pr_reviewer.enable_review_labels_effort = original["enable_review_labels_effort"]
        settings.pr_reviewer.enable_review_labels_security = original["enable_review_labels_security"]


def test_get_user_answers_collects_question_and_answer_from_issue_comments():
    git_provider = MagicMock()
    git_provider.get_issue_comments.return_value = SimpleNamespace(reversed=[
        SimpleNamespace(body="Unrelated"),
        SimpleNamespace(body="Questions to better understand the PR:\n- Why?"),
        SimpleNamespace(body="/answer Because it fixes production."),
    ])
    reviewer = _make_reviewer(git_provider)
    reviewer.is_answer = True

    question, answer = reviewer._get_user_answers()

    assert question == "Questions to better understand the PR:\n- Why?"
    assert answer == "/answer Because it fixes production."


def test_pr_comment_context_keeps_human_rebuttals_and_filters_bot_output():
    settings = get_settings()
    original = settings.pr_reviewer.get("include_pr_comments", False)
    settings.pr_reviewer.include_pr_comments = True
    provider = MagicMock()
    provider.is_supported.return_value = True
    provider.get_issue_comments.return_value = [
        {
            "body": "The null guard is in parseConfig(); this finding is a false alarm.",
            "user": {"login": "alice"},
            "created_at": "2026-07-16T10:00:00Z",
        },
        {"body": "## PR Reviewer Guide\nOld generated finding", "user": {"login": "PR-Agent"}},
        {"body": "Third-party bot output", "user": {"login": "automation", "type": "Bot"}},
        {"body": "/review", "user": {"login": "alice"}},
        {"body": "/config --show", "user": {"login": "bob"}},
    ]
    reviewer = _make_reviewer(provider)

    try:
        context = reviewer._get_pr_comments_context()
    finally:
        settings.pr_reviewer.include_pr_comments = original

    assert "alice" in context
    assert "false alarm" in context
    assert "Old generated finding" not in context
    assert "Third-party bot output" not in context
    assert "/review" not in context
    assert "/config --show" not in context


def test_pr_comment_context_honors_total_character_limit():
    settings = get_settings()
    original_enabled = settings.pr_reviewer.get("include_pr_comments", False)
    original_limit = settings.pr_reviewer.get("max_pr_comment_chars", 12000)
    settings.pr_reviewer.include_pr_comments = True
    settings.pr_reviewer.max_pr_comment_chars = 40
    provider = MagicMock()
    provider.is_supported.return_value = True
    provider.get_issue_comments.return_value = [
        {"body": "Older implementation context", "user": {"login": "alice"}},
        {"body": "Newest rebuttal context", "user": {"login": "bob"}},
    ]
    reviewer = _make_reviewer(provider)

    try:
        context = reviewer._get_pr_comments_context()
    finally:
        settings.pr_reviewer.include_pr_comments = original_enabled
        settings.pr_reviewer.max_pr_comment_chars = original_limit

    assert len(context) <= 40
    assert "Newest rebuttal context" in context


def test_init_maps_user_question_and_answer_to_correct_prompt_vars(monkeypatch):
    """Behavioral regression for the swapped-unpacking bug (#2496).

    The bug lived in ``PRReviewer.__init__``: ``_get_user_answers()`` returns
    ``(question, answer)`` but the tuple was unpacked as ``answer, question``,
    so the review prompt rendered the user's answer under ``{{ question_str }}``
    and the question under ``{{ answer_str }}``. This drives the real ``__init__``
    (external collaborators stubbed) and asserts each value lands in ``self.vars``
    under the correct key — so it fails if the unpack is ever swapped again,
    regardless of how the line is formatted.
    """
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    provider = MagicMock()
    provider.is_supported.return_value = True
    provider.get_languages.return_value = {}
    provider.get_files.return_value = []
    provider.get_issue_comments.return_value = SimpleNamespace(reversed=[
        SimpleNamespace(body="Questions to better understand the PR:\n- Why?"),
        SimpleNamespace(body="/answer Because it fixes production."),
    ])
    provider.get_pr_description.return_value = ("desc", [])

    monkeypatch.setattr(pr_reviewer_module, "get_git_provider_with_context", lambda pr_url: provider)
    monkeypatch.setattr(pr_reviewer_module, "get_main_pr_language", lambda languages, files: "Python")
    monkeypatch.setattr(pr_reviewer_module, "TokenHandler", MagicMock())

    reviewer = PRReviewer(
        "https://example/pr/1",
        is_answer=True,
        ai_handler=lambda: SimpleNamespace(main_pr_language=None),
    )

    assert reviewer.vars["question_str"] == "Questions to better understand the PR:\n- Why?"
    assert reviewer.vars["answer_str"] == "/answer Because it fixes production."


def test_publish_review_as_commit_status_noop_when_flag_off():
    """The Gitea commit-status surface must stay off unless explicitly enabled,
    so it is a no-op for every provider by default."""
    settings = get_settings()
    original = settings.get("gitea.publish_review_as_status", False)
    try:
        settings.set("gitea.publish_review_as_status", False)
        provider = MagicMock()
        reviewer = _make_reviewer(provider)
        reviewer._publish_review_as_commit_status("## Review\n\nAll good")
        provider.publish_commit_status.assert_not_called()
    finally:
        settings.set("gitea.publish_review_as_status", original)


def test_publish_review_as_commit_status_noop_when_provider_lacks_capability():
    """Even with the flag on, a provider without publish_commit_status (e.g.
    GitHub/GitLab) must be untouched — no AttributeError, no call."""
    settings = get_settings()
    original = settings.get("gitea.publish_review_as_status", False)
    try:
        settings.set("gitea.publish_review_as_status", True)
        # spec=[] => hasattr is False for any attribute name.
        provider = MagicMock(spec=[])
        reviewer = _make_reviewer(provider)
        # Must not raise.
        reviewer._publish_review_as_commit_status("## Review\n\nAll good")
    finally:
        settings.set("gitea.publish_review_as_status", original)


def test_publish_review_as_commit_status_emits_success_when_enabled():
    """With the flag on and a Gitea-like provider, a success status is emitted
    on the PR head, mirroring GitHub's publish_as_check_run surface."""
    settings = get_settings()
    original = settings.get("gitea.publish_review_as_status", False)
    try:
        settings.set("gitea.publish_review_as_status", True)
        provider = MagicMock(spec=["publish_commit_status", "get_pr_url"])
        provider.get_pr_url.return_value = "https://gitea.example.com/o/r/pulls/1"
        reviewer = _make_reviewer(provider)

        reviewer._publish_review_as_commit_status("## PR Review 🔍\n\nSummary line here")

        provider.publish_commit_status.assert_called_once()
        _, kwargs = provider.publish_commit_status.call_args
        assert kwargs["state"] == "success"
        assert kwargs["context"] == "PR-Agent/review"
        # Summary is the first block, stripped of markdown heading chars.
        assert kwargs["description"] == "PR Review 🔍"
        assert kwargs["target_url"] == "https://gitea.example.com/o/r/pulls/1"
    finally:
        settings.set("gitea.publish_review_as_status", original)


def test_publish_review_as_commit_status_swallows_provider_errors():
    """A failing status emission must never break the review flow."""
    settings = get_settings()
    original = settings.get("gitea.publish_review_as_status", False)
    try:
        settings.set("gitea.publish_review_as_status", True)
        provider = MagicMock(spec=["publish_commit_status", "get_pr_url"])
        provider.get_pr_url.return_value = "u"
        provider.publish_commit_status.side_effect = Exception("boom")
        reviewer = _make_reviewer(provider)
        # Must not raise.
        reviewer._publish_review_as_commit_status("## Review\n\nbody")
    finally:
        settings.set("gitea.publish_review_as_status", original)


def test_publish_review_failure_edits_temporary_comment_and_redacts_secrets():
    settings = get_settings()
    original_publish_output = settings.config.publish_output
    original_status = settings.get("gitea.publish_review_as_status", False)
    try:
        settings.config.publish_output = True
        settings.set("gitea.publish_review_as_status", True)
        provider = MagicMock()
        provider.comments_list = [{"is_temporary": True, "comment_id": 7}]
        provider.get_pr_url.return_value = "https://gitea.example.com/o/r/pulls/1"
        reviewer = _make_reviewer(provider)

        reviewer._publish_review_failure(RuntimeError("refresh_token = secret-refresh-token"))

        provider.edit_comment.assert_called_once()
        body = provider.edit_comment.call_args.args[1]
        assert "PR-Agent 评审失败" in body
        assert "secret-refresh-token" not in body
        assert "<redacted>" in body
        provider.publish_comment.assert_not_called()
        provider.publish_commit_status.assert_called_once()
        assert provider.publish_commit_status.call_args.kwargs["state"] == "error"
    finally:
        settings.config.publish_output = original_publish_output
        settings.set("gitea.publish_review_as_status", original_status)


def test_run_publishes_failure_alert_when_prediction_fails(monkeypatch):
    from pr_agent.tools import pr_reviewer as pr_reviewer_module

    settings = get_settings()
    original_publish_output = settings.config.publish_output
    try:
        settings.config.publish_output = True
        provider = MagicMock()
        provider.get_files.return_value = ["file.py"]
        reviewer = _make_reviewer(provider)
        reviewer.incremental = SimpleNamespace(is_incremental=False)
        reviewer._publish_review_failure = MagicMock()

        async def _fail(*args, **kwargs):
            raise RuntimeError("auth failed")

        monkeypatch.setattr(pr_reviewer_module, "retry_with_fallback_models", _fail)

        asyncio.run(reviewer.run())

        reviewer._publish_review_failure.assert_called_once()
    finally:
        settings.config.publish_output = original_publish_output
