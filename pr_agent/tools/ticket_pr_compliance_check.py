import re
import traceback

from pr_agent.config_loader import get_settings
from pr_agent.git_providers import GithubProvider
from pr_agent.git_providers import AzureDevopsProvider
from pr_agent.git_providers import GiteaProvider
from pr_agent.log import get_logger

# Compile the regex pattern once, outside the function
GITHUB_TICKET_PATTERN = re.compile(
     r'(https?://[^\s)>"\']+/[^/\s)>"\']+/[^/\s)>"\']+/issues/\d+)'
     r'|(\b([\w.-]+)/([\w.-]+)#(\d+)\b)|(#\d+)'
)
# Option A: issue number at start of branch or after /, followed by - or end (e.g. feature/1-test-issue, 123-fix)
BRANCH_ISSUE_PATTERN = re.compile(r"(?:^|/)(\d{1,6})(?=-|$)")
CLOSED_TICKET_EXPLICIT_CONTEXT_PATTERN = re.compile(
    r"(?i)(acceptance|requirements?|context|according to|based on|use(?:d|s|ing)?|"
    r"align(?:ed)? with|follow|against|validate|验收|需求|要求|上下文|按|根据|依据|对齐|参照|以.+为准)"
)

def find_jira_tickets(text):
    # Regular expression patterns for JIRA tickets
    patterns = [
        r'\b[A-Z]{2,10}-\d{1,7}\b',  # Standard JIRA ticket format (e.g., PROJ-123)
        r'(?:https?://[^\s/]+/browse/)?([A-Z]{2,10}-\d{1,7})\b'  # JIRA URL or just the ticket
    ]

    tickets = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                # If it's a tuple (from the URL pattern), take the last non-empty group
                ticket = next((m for m in reversed(match) if m), None)
            else:
                ticket = match
            if ticket:
                tickets.add(ticket)

    return list(tickets)


def extract_ticket_links_from_pr_description(pr_description, repo_path, base_url_html='https://github.com'):
    """
    Extract all ticket links from PR description
    """
    # Preserve first-seen order while de-duplicating, so the cap below selects a
    # deterministic subset (a plain set would slice an arbitrary, run-varying one).
    seen = set()
    github_tickets = []

    def _add(url):
        if url not in seen:
            seen.add(url)
            github_tickets.append(url)

    try:
        # Use the updated pattern to find matches
        matches = GITHUB_TICKET_PATTERN.findall(pr_description)

        for match in matches:
            if match[0]:  # Full URL match
                _add(match[0])
            elif match[1]:  # Shorthand notation match: owner/repo#issue_number
                owner, repo, issue_number = match[2], match[3], match[4]
                _add(f"{base_url_html.strip('/')}/{owner}/{repo}/issues/{issue_number}")
            else:  # #123 format
                issue_number = match[5][1:]  # remove #
                if issue_number.isdigit() and len(issue_number) < 5 and repo_path:
                    _add(f"{base_url_html.strip('/')}/{repo_path}/issues/{issue_number}")

        if len(github_tickets) > 3:
            get_logger().info(f"Too many tickets found in PR description: {len(github_tickets)}")
            # Limit the number of tickets to 3
            github_tickets = github_tickets[:3]
    except Exception as e:
        get_logger().error(f"Error extracting tickets error= {e}",
                           artifact={"traceback": traceback.format_exc()})

    return github_tickets

def extract_ticket_links_from_branch_name(branch_name, repo_path, base_url_html="https://github.com"):
    """
    Extract GitHub issue URLs from branch name. Numbers are matched at start of branch or after /,
    followed by - or end (e.g. feature/1-test-issue -> #1). Respects extract_issue_from_branch
    and optional branch_issue_regex (may be under [config] in TOML).
    """
    if not branch_name or not repo_path:
        return []
    if not isinstance(branch_name, str):
        return []
    settings = get_settings()
    if not settings.get("extract_issue_from_branch", settings.get("config.extract_issue_from_branch", True)):
        return []
    github_tickets = set()
    custom_regex_str = settings.get("branch_issue_regex") or settings.get("config.branch_issue_regex", "") or ""
    if custom_regex_str:
        try:
            pattern = re.compile(custom_regex_str)
            if pattern.groups < 1:
                get_logger().error(
                    "branch_issue_regex must contain at least one capturing group for the issue number; using default pattern."
                )
                pattern = BRANCH_ISSUE_PATTERN
        except re.error as e:
            get_logger().error(f"Invalid custom regex for branch issue extraction: {e}")
            return []
    else:
        pattern = BRANCH_ISSUE_PATTERN
    for match in pattern.finditer(branch_name):
        try:
            issue_number = match.group(1)
        except IndexError:
            continue
        if issue_number and issue_number.isdigit():
            github_tickets.add(
                f"{base_url_html.strip('/')}/{repo_path}/issues/{issue_number}"
            )
    return list(github_tickets)


def _issue_attr(issue, attr_name, default=None):
    if isinstance(issue, dict):
        return issue.get(attr_name, default)
    return getattr(issue, attr_name, default)


def _issue_labels(issue):
    labels = []
    try:
        for label in _issue_attr(issue, "labels", []) or []:
            if isinstance(label, dict):
                labels.append(label.get("name", ""))
            else:
                labels.append(label.name if hasattr(label, "name") else label)
    except Exception as e:
        get_logger().error(f"Error extracting labels error= {e}",
                           artifact={"traceback": traceback.format_exc()})
    return [str(label) for label in labels if label]


def _trim_ticket_body(body, max_characters):
    body = body or ""
    if len(body) > max_characters:
        return body[:max_characters] + "..."
    return body


def _issue_is_closed(issue):
    state = str(_issue_attr(issue, "state", "") or "").strip().lower()
    return state in ("closed", "done", "resolved")


def _ticket_issue_number(ticket_url):
    match = re.search(r"/issues/(\d+)(?:$|[?#])", ticket_url)
    return match.group(1) if match else ""


def _closed_ticket_explicitly_requested(user_description, ticket_url):
    issue_number = _ticket_issue_number(ticket_url)
    if not issue_number:
        return False
    for line in (user_description or "").splitlines():
        if (
            ticket_url in line
            or f"/issues/{issue_number}" in line
            or f"#{issue_number}" in line
        ) and CLOSED_TICKET_EXPLICIT_CONTEXT_PATTERN.search(line):
            return True
    return False


def _should_use_ticket(issue, ticket_url, user_description):
    if not _issue_is_closed(issue):
        return True
    if _closed_ticket_explicitly_requested(user_description, ticket_url):
        return True
    get_logger().info(
        f"Skipping closed ticket {ticket_url}; PR description did not request it as review context"
    )
    return False


def _build_ticket_content(ticket_url, issue, body, labels, sub_issues_content=None):
    return {
        'ticket_id': _issue_attr(issue, "number"),
        'ticket_url': ticket_url,
        'title': _issue_attr(issue, "title", ""),
        'body': body,
        'labels': ", ".join(labels),
        'sub_issues': sub_issues_content or [],
    }


async def extract_tickets(git_provider):
    MAX_TICKET_CHARACTERS = 10000
    try:
        if isinstance(git_provider, GithubProvider):
            user_description = git_provider.get_user_description()
            description_tickets = extract_ticket_links_from_pr_description(
                user_description, git_provider.repo, git_provider.base_url_html
            )
            branch_name = git_provider.get_pr_branch()
            branch_tickets = extract_ticket_links_from_branch_name(
                branch_name, git_provider.repo, git_provider.base_url_html
            )
            seen = set()
            merged = []
            for link in description_tickets + branch_tickets:
                if link not in seen:
                    seen.add(link)
                    merged.append(link)
            if len(merged) > 3:
                get_logger().info(f"Too many tickets (description + branch): {len(merged)}")
                tickets = merged[:3]
            else:
                tickets = merged
            tickets_content = []

            if tickets:

                for ticket in tickets:
                    repo_name, original_issue_number = git_provider._parse_issue_url(ticket)

                    try:
                        issue_main = git_provider.repo_obj.get_issue(original_issue_number)
                    except Exception as e:
                        get_logger().error(f"Error getting main issue: {e}",
                                           artifact={"traceback": traceback.format_exc()})
                        continue
                    if not _should_use_ticket(issue_main, ticket, user_description):
                        continue

                    issue_body_str = issue_main.body or ""
                    if len(issue_body_str) > MAX_TICKET_CHARACTERS:
                        issue_body_str = issue_body_str[:MAX_TICKET_CHARACTERS] + "..."

                    # Extract sub-issues
                    sub_issues_content = []
                    try:
                        sub_issues = git_provider.fetch_sub_issues(ticket)
                        for sub_issue_url in sub_issues:
                            try:
                                sub_repo, sub_issue_number = git_provider._parse_issue_url(sub_issue_url)
                                sub_issue = git_provider.repo_obj.get_issue(sub_issue_number)
                                if not _should_use_ticket(sub_issue, sub_issue_url, user_description):
                                    continue

                                sub_body = sub_issue.body or ""
                                if len(sub_body) > MAX_TICKET_CHARACTERS:
                                    sub_body = sub_body[:MAX_TICKET_CHARACTERS] + "..."

                                # Extract sub-issue labels
                                sub_labels = []
                                try:
                                    for label in sub_issue.labels:
                                        sub_labels.append(label.name if hasattr(label, 'name') else label)
                                except Exception as e:
                                    get_logger().error(f"Error extracting labels error= {e}",
                                                       artifact={"traceback": traceback.format_exc()})

                                sub_issues_content.append({
                                    'ticket_url': sub_issue_url,
                                    'title': sub_issue.title,
                                    'body': sub_body,
                                    'labels': ", ".join(sub_labels)
                                })
                            except Exception as e:
                                get_logger().warning(f"Failed to fetch sub-issue content for {sub_issue_url}: {e}")

                    except Exception as e:
                        get_logger().warning(f"Failed to fetch sub-issues for {ticket}: {e}")

                    # Extract labels
                    labels = _issue_labels(issue_main)

                    tickets_content.append({
                        'ticket_id': issue_main.number,
                        'ticket_url': ticket,
                        'title': issue_main.title,
                        'body': issue_body_str,
                        'labels': ", ".join(labels),
                        'sub_issues': sub_issues_content  # Store sub-issues content
                    })

                return tickets_content

        elif isinstance(git_provider, GiteaProvider):
            user_description = git_provider.get_user_description()
            repo_path = f"{git_provider.owner}/{git_provider.repo}"
            description_tickets = extract_ticket_links_from_pr_description(
                user_description, repo_path, git_provider.base_url
            )
            branch_name = git_provider.get_pr_branch()
            branch_tickets = extract_ticket_links_from_branch_name(
                branch_name, repo_path, git_provider.base_url
            )
            seen = set()
            tickets = []
            for link in description_tickets + branch_tickets:
                if link not in seen:
                    seen.add(link)
                    tickets.append(link)
            if len(tickets) > 3:
                get_logger().info(f"Too many tickets (description + branch): {len(tickets)}")
                tickets = tickets[:3]

            tickets_content = []
            for ticket in tickets:
                try:
                    owner, repo, issue_number = git_provider._parse_issue_url(ticket)
                    issue_main = git_provider.repo_api.get_issue(
                        owner=owner, repo=repo, index=issue_number
                    )
                except Exception as e:
                    get_logger().error(f"Error getting main issue: {e}",
                                       artifact={"traceback": traceback.format_exc()})
                    continue
                if not _should_use_ticket(issue_main, ticket, user_description):
                    continue

                issue_body_str = _trim_ticket_body(
                    _issue_attr(issue_main, "body", ""), MAX_TICKET_CHARACTERS
                )

                sub_issues_content = []
                try:
                    sub_issues = git_provider.fetch_sub_issues(ticket)
                    for sub_issue_url in sub_issues:
                        try:
                            sub_owner, sub_repo, sub_issue_number = git_provider._parse_issue_url(sub_issue_url)
                            sub_issue = git_provider.repo_api.get_issue(
                                owner=sub_owner, repo=sub_repo, index=sub_issue_number
                            )
                            if not _should_use_ticket(sub_issue, sub_issue_url, user_description):
                                continue
                            sub_issues_content.append(_build_ticket_content(
                                sub_issue_url,
                                sub_issue,
                                _trim_ticket_body(
                                    _issue_attr(sub_issue, "body", ""), MAX_TICKET_CHARACTERS
                                ),
                                _issue_labels(sub_issue),
                            ))
                        except Exception as e:
                            get_logger().warning(f"Failed to fetch sub-issue content for {sub_issue_url}: {e}")
                except Exception as e:
                    get_logger().warning(f"Failed to fetch sub-issues for {ticket}: {e}")

                tickets_content.append(_build_ticket_content(
                    ticket,
                    issue_main,
                    issue_body_str,
                    _issue_labels(issue_main),
                    sub_issues_content,
                ))

            return tickets_content

        elif isinstance(git_provider, AzureDevopsProvider):
            tickets_info = git_provider.get_linked_work_items()
            tickets_content = []
            for ticket in tickets_info:
                try:
                    ticket_body_str = ticket.get("body", "")
                    if len(ticket_body_str) > MAX_TICKET_CHARACTERS:
                        ticket_body_str = ticket_body_str[:MAX_TICKET_CHARACTERS] + "..."

                    tickets_content.append(
                        {
                            "ticket_id": ticket.get("id"),
                            "ticket_url": ticket.get("url"),
                            "title": ticket.get("title"),
                            "body": ticket_body_str,
                            "requirements": ticket.get("acceptance_criteria", ""),
                            "labels": ", ".join(ticket.get("labels", [])),
                        }
                    )
                except Exception as e:
                    get_logger().error(
                        f"Error processing Azure DevOps ticket: {e}",
                        artifact={"traceback": traceback.format_exc()},
                    )
            return tickets_content

    except Exception as e:
        get_logger().error(f"Error extracting tickets error= {e}",
                           artifact={"traceback": traceback.format_exc()})


async def extract_and_cache_pr_tickets(git_provider, vars):
    if not get_settings().get('pr_reviewer.require_ticket_analysis_review', False):
        return

    related_tickets = get_settings().get('related_tickets', [])

    if not related_tickets:
        tickets_content = await extract_tickets(git_provider)

        if tickets_content:
            # Store sub-issues along with main issues
            for ticket in tickets_content:
                if "sub_issues" in ticket and ticket["sub_issues"]:
                    for sub_issue in ticket["sub_issues"]:
                        related_tickets.append(sub_issue)  # Add sub-issues content

                related_tickets.append(ticket)

            get_logger().info("Extracted tickets and sub-issues from PR description",
                              artifact={"tickets": related_tickets})

            vars['related_tickets'] = related_tickets
            get_settings().set('related_tickets', related_tickets)
    else:
        get_logger().info("Using cached tickets", artifact={"tickets": related_tickets})
        vars['related_tickets'] = related_tickets


def check_tickets_relevancy():
    return True
