# Gitea ⇄ GitHub Provider Parity

This documents where `GiteaProvider` reaches full parity with `GithubProvider`,
and — importantly — which GitHub capabilities are **intentionally not ported**
because Gitea's server has no equivalent. "Not portable" here means *left
unimplemented by design*, not by omission.

## Implemented (parity or best-effort Gitea approximation)

| Capability | Status | Notes |
|-----------|--------|-------|
| `get_pr_file_content(file_path, branch)` | Full | Reads via `GET /raw/{path}?ref=`; `?ref=` accepts a branch. Fixes a live AttributeError in `/update_changelog`. |
| `create_or_update_pr_file(...)` | Full | Contents **base64-encoded**, update needs the **blob sha** from `GET /contents/{path}` (create omits it). Completes the `/update_changelog` push path. |
| `is_supported("publish_file_comments") → False` | Full | Gitea has no `subject_type=='file'` diff-view comment. Returning False (like GitLab) closes a latent AttributeError when `pr_description.inline_file_summary=true`. |
| `get_canonical_url_parts(...)` | Full | Gitea file-view uses `src/branch/{branch}` (not GitHub's `blob/`). Fixes `/help_docs` doc links. |
| `get_lines_link_original_file(...)` | Full | Commit-pinned `src/commit/{sha}` deep link. Completeness only — no active Gitea caller today. |
| `fetch_sub_issues(issue_url)` | **Approximation** | Gitea has no sub-issues; uses issue **dependencies** (`GET /issues/{index}/dependencies`), the closest native analogue. Single level, no pagination cursor. Only exercised by ticket-compliance with linked tickets. |
| Review → commit status (`gitea.publish_review_as_status`, default off) | **Approximation** | Wires the existing `publish_commit_status` into `pr_reviewer`. Gitea has no Checks API, so this is the commit-status surface (the portable subset of GitHub's `publish_as_check_run`), emitting a `success` status on the PR head so branch protection can require `PR-Agent/review`. Mirrors GitHub's neutral check-run conclusion (the status means "review ran/published", not a derived pass/fail verdict). |

## NOT PORTABLE (left unimplemented by design)

- **GitHub App installation-token auth** (`deployment_type=='app'`, `AppAuthentication`,
  `installation_id`, marketplace webhooks). Gitea has no App model — no app id, no
  per-installation JWT; auth is a single PAT. No server-side concept to map to.
- **Checks API / check runs** (`_publish_check_run`, `publish_as_check_run`, the
  `check_run` webhook branch). Gitea has no Checks API — only commit **statuses**.
  The commit-status surface (`publish_commit_status`, now wired behind
  `gitea.publish_review_as_status`) is the ceiling.
- **GraphQL sub-issues, verbatim.** Gitea has no GraphQL API and no sub-issue
  primitive. Issue **dependencies** (implemented above) are the REST analogue; the
  GraphQL query itself is not portable.
- **`get_notifications(since)`** — user-deployment notification polling. Consumed
  only by `github_polling.py`, which has no Gitea counterpart server. Out of scope
  for the webhook-driven Gitea app.
- **Draft-verify-delete inline-comment validation**
  (`_verify_code_comment`, `_publish_inline_comments_fallback_with_verification`,
  `_try_fix_invalid_inline_comments`). These exploit GitHub's create-PENDING →
  verify → DELETE-draft round-trip. Gitea accepts a submitted review directly; this
  fork deliberately submits `event=COMMENT` in one shot, which is the correct Gitea
  design. Not worth porting.
- **`validate_comments_inside_hunks`** — a GitHub committable-comment hard
  requirement. Gitea anchors positions to the PR head more leniently (this fork
  already omits `commit_id`), so there is no forcing requirement; porting adds risk
  without benefit.

## Intentionally retained

- `RepoApi.create_inline_comment` (the lower-level PENDING-draft path) is now
  effectively dead — `publish_inline_comments` submits with `event=COMMENT` instead.
  It is kept (rather than deleted) because two pre-existing unit tests pin its
  behavior; removing it would churn the passing baseline for no parity gain. It has
  no provider-level caller, so it is inert.
