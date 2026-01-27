"""
Bloom Houston Collective Knowledge MCP Server

An MCP server that queries Pinboard's API v2 for bookmarks tagged
bloom-houston, exposing collective learning and contributor relationships.

Requires PINBOARD_API_TOKEN environment variable (format: username:token)
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urlencode

import httpx

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Constants
PINBOARD_API_BASE = "https://api.pinboard.in/v2"
COLLECTIVE_TAG = "bloom-houston"
CACHE_TTL_SECONDS = 300  # 5 minutes


@dataclass
class RateLimitInfo:
    """Track rate limit status from API headers."""
    requests_remaining: Optional[int] = None
    reset_time: Optional[str] = None
    reset_timestamp: Optional[int] = None


class PinboardAPIError(Exception):
    """Custom exception for Pinboard API errors."""
    def __init__(self, message: str, error_code: Optional[int] = None, error_type: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        self.error_type = error_type
        super().__init__(self.message)


class PinboardClient:
    """Client for Pinboard API v2 with authentication and rate limiting."""

    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self.rate_limit = RateLimitInfo()
        self._cache: dict = {}
        self._cache_timestamps: dict = {}

    def _get_cache_key(self, endpoint: str, params: dict) -> str:
        """Generate cache key from endpoint and params."""
        param_str = urlencode(sorted(params.items()))
        return f"{endpoint}?{param_str}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[cache_key]) < CACHE_TTL_SECONDS

    def _update_rate_limit(self, response: httpx.Response) -> None:
        """Update rate limit info from response headers."""
        if "X-Requests-Remaining" in response.headers:
            self.rate_limit.requests_remaining = int(response.headers["X-Requests-Remaining"])
        if "X-Reset-Time" in response.headers:
            self.rate_limit.reset_time = response.headers["X-Reset-Time"]
        if "X-Reset-Timestamp" in response.headers:
            self.rate_limit.reset_timestamp = int(response.headers["X-Reset-Timestamp"])

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        use_cache: bool = True
    ) -> dict:
        """Make authenticated request to Pinboard API v2."""
        if params is None:
            params = {}

        cache_key = self._get_cache_key(endpoint, params)

        # Check cache for GET requests
        if method == "GET" and use_cache and self._is_cache_valid(cache_key):
            return self._cache[cache_key]

        url = f"{PINBOARD_API_BASE}{endpoint}"

        # Use header-based authentication
        headers = {
            "X-Auth-Token": self.auth_token,
            "User-Agent": "BloomHoustonMCP/1.0"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=headers,
                    timeout=30.0
                )
            except httpx.TimeoutException:
                raise PinboardAPIError("Request timed out", error_code=504)
            except httpx.RequestError as e:
                raise PinboardAPIError(f"Network error: {str(e)}", error_code=503)

            # Update rate limit info
            self._update_rate_limit(response)

            # Handle HTTP errors
            if response.status_code == 401:
                raise PinboardAPIError("Authentication failed. Check your API token.", error_code=401)
            elif response.status_code == 402:
                raise PinboardAPIError("Payment required or account locked.", error_code=402)
            elif response.status_code == 429:
                reset_info = ""
                if self.rate_limit.reset_time:
                    reset_info = f" Resets at {self.rate_limit.reset_time}"
                raise PinboardAPIError(f"Rate limited.{reset_info}", error_code=429)
            elif response.status_code == 503:
                raise PinboardAPIError("Pinboard service unavailable.", error_code=503)
            elif response.status_code >= 400:
                raise PinboardAPIError(f"HTTP error {response.status_code}", error_code=response.status_code)

            # Parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError:
                raise PinboardAPIError("Invalid JSON response from API")

            # Check for API-level errors
            if data.get("status") == "error":
                error_msg = data.get("error_message", data.get("error", "Unknown error"))
                error_code = data.get("error_code")
                raise PinboardAPIError(error_msg, error_code=int(error_code) if error_code else None)

            # Cache successful GET responses
            if method == "GET" and use_cache:
                self._cache[cache_key] = data
                self._cache_timestamps[cache_key] = time.time()

            return data

    async def get_tag_bookmarks(self, tag: str = COLLECTIVE_TAG, count: int = 100) -> list[dict]:
        """Get bookmarks for a specific tag from sitewide public bookmarks."""
        params = {"count": count}
        data = await self._request("GET", f"/site/tag/{tag}", params)
        return data.get("bookmarks", data.get("posts", []))

    async def search_site(self, query: str, tag: Optional[str] = None, count: int = 100) -> list[dict]:
        """Search sitewide public bookmarks."""
        params = {"query": query, "count": count}
        if tag:
            params["tag"] = tag
        data = await self._request("GET", "/site/search/", params)
        return data.get("bookmarks", data.get("posts", []))

    async def get_recent(self, tag: Optional[str] = None, count: int = 20) -> list[dict]:
        """Get recent sitewide bookmarks, optionally filtered by tag."""
        params = {"count": count}
        if tag:
            params["tag"] = tag
        data = await self._request("GET", "/site/recent/", params)
        return data.get("bookmarks", data.get("posts", []))

    async def get_url_info(self, url: str) -> dict:
        """Get information about a specific URL."""
        params = {"url": url}
        data = await self._request("GET", "/url/", params)
        return data

    def get_rate_limit_status(self) -> str:
        """Get human-readable rate limit status."""
        if self.rate_limit.requests_remaining is not None:
            return f"Requests remaining: {self.rate_limit.requests_remaining}"
        return "Rate limit info not available"


# Global client instance (initialized on first use)
_client: Optional[PinboardClient] = None


def get_client() -> PinboardClient:
    """Get or create the Pinboard client."""
    global _client
    if _client is None:
        auth_token = os.environ.get("PINBOARD_API_TOKEN")
        if not auth_token:
            raise PinboardAPIError(
                "PINBOARD_API_TOKEN environment variable not set. "
                "Set it to your Pinboard API token (format: username:token)",
                error_code=401
            )
        _client = PinboardClient(auth_token)
    return _client


def normalize_url(url: str) -> str:
    """Normalize URL for comparison (following Pinboard's normalization)."""
    parsed = urlparse(url)
    # Remove trailing slashes and normalize case
    path = parsed.path.rstrip('/')
    return f"{parsed.scheme}://{parsed.netloc}{path}".lower()


def format_bookmark(b: dict) -> list[str]:
    """Format a bookmark for display."""
    lines = []
    # Handle both old feed format and new API format
    title = b.get("description", b.get("d", "No title"))
    url = b.get("href", b.get("u", "N/A"))
    user = b.get("user", b.get("a", "Unknown"))
    tags = b.get("tags", b.get("t", []))
    if isinstance(tags, str):
        tags = tags.split()
    extended = b.get("extended", b.get("n", ""))
    date = b.get("time", b.get("dt", "Unknown"))

    lines.append(f"**{title}**")
    lines.append(f"  URL: {url}")
    lines.append(f"  Saved by: {user}")
    lines.append(f"  Tags: {', '.join(tags) if tags else 'none'}")
    if extended:
        lines.append(f"  Notes: {extended}")
    lines.append(f"  Date: {date}")
    lines.append("")
    return lines


# Create the MCP server
server = Server("bloom-houston-collective")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for the Bloom Houston collective."""
    return [
        Tool(
            name="search_collective",
            description="Search across all bloom-houston bookmarks by text, tag, or user. "
                       "Returns matching bookmarks with titles, URLs, descriptions, and who saved them.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to match against title, description, tags, or URL"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by specific tag (in addition to bloom-houston)"
                    },
                    "user": {
                        "type": "string",
                        "description": "Filter by username who saved the bookmark"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_recent",
            description="Get the most recently shared bookmarks in the bloom-houston collective. "
                       "Optionally filter by an additional tag.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Optional additional tag to filter by"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)",
                        "default": 10
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="who_knows_about",
            description="Find contributors who have saved bookmarks about a specific topic. "
                       "Returns users ranked by how many relevant bookmarks they've shared.",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to search for (matches against title, description, and tags)"
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="find_connections",
            description="Find interest overlaps between members of the collective. "
                       "Shows which users share similar interests based on common tags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "Optional: Find connections for a specific user. "
                                      "If not provided, shows overall network connections."
                    },
                    "min_overlap": {
                        "type": "integer",
                        "description": "Minimum number of shared tags to consider a connection (default: 2)",
                        "default": 2
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="who_else_saved",
            description="Check if a URL was saved by others in the bloom-houston network. "
                       "Useful for finding who else is interested in a specific resource.",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to check"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="get_network_map",
            description="Generate a relationship graph showing connections between members "
                       "based on shared interests. Returns JSON or Mermaid diagram format.",
            inputSchema={
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["json", "mermaid"],
                        "description": "Output format: 'json' for structured data or 'mermaid' for diagram",
                        "default": "json"
                    },
                    "min_shared_tags": {
                        "type": "integer",
                        "description": "Minimum shared tags to show a connection (default: 1)",
                        "default": 1
                    }
                },
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "search_collective":
            return await search_collective(
                query=arguments.get("query"),
                tag=arguments.get("tag"),
                user=arguments.get("user"),
                limit=arguments.get("limit", 20)
            )
        elif name == "get_recent":
            return await get_recent(
                tag=arguments.get("tag"),
                limit=arguments.get("limit", 10)
            )
        elif name == "who_knows_about":
            return await who_knows_about(topic=arguments["topic"])
        elif name == "find_connections":
            return await find_connections(
                user=arguments.get("user"),
                min_overlap=arguments.get("min_overlap", 2)
            )
        elif name == "who_else_saved":
            return await who_else_saved(url=arguments["url"])
        elif name == "get_network_map":
            return await get_network_map(
                format=arguments.get("format", "json"),
                min_shared_tags=arguments.get("min_shared_tags", 1)
            )
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except PinboardAPIError as e:
        error_msg = f"Pinboard API Error: {e.message}"
        if e.error_code:
            error_msg += f" (code: {e.error_code})"
        return [TextContent(type="text", text=error_msg)]


async def fetch_collective_bookmarks() -> list[dict]:
    """Fetch all bloom-houston tagged bookmarks."""
    client = get_client()
    return await client.get_tag_bookmarks(COLLECTIVE_TAG)


def extract_bookmark_field(b: dict, *fields: str) -> str:
    """Extract a field from bookmark, trying multiple possible keys."""
    for field in fields:
        if field in b and b[field]:
            return b[field]
    return ""


def get_bookmark_tags(b: dict) -> list[str]:
    """Extract tags from bookmark, handling different formats."""
    tags = b.get("tags", b.get("t", []))
    if isinstance(tags, str):
        return tags.split()
    return tags or []


def get_bookmark_user(b: dict) -> str:
    """Extract username from bookmark."""
    return b.get("user", b.get("a", "Unknown"))


async def search_collective(
    query: Optional[str] = None,
    tag: Optional[str] = None,
    user: Optional[str] = None,
    limit: int = 20
) -> list[TextContent]:
    """Search across all bloom-houston bookmarks."""
    client = get_client()

    # If we have a text query, use the search endpoint
    if query:
        # Search with bloom-houston tag constraint
        search_tag = f"{COLLECTIVE_TAG}"
        if tag:
            search_tag = f"{COLLECTIVE_TAG}+{tag}"
        bookmarks = await client.search_site(query, tag=search_tag, count=limit * 2)
    else:
        # Get all collective bookmarks and filter locally
        bookmarks = await client.get_tag_bookmarks(COLLECTIVE_TAG)

    results = []
    for bookmark in bookmarks:
        # Apply user filter
        if user and get_bookmark_user(bookmark).lower() != user.lower():
            continue

        # Apply tag filter (if not already applied via API)
        if tag and not query:
            bookmark_tags = [t.lower() for t in get_bookmark_tags(bookmark)]
            if tag.lower() not in bookmark_tags:
                continue

        results.append(bookmark)
        if len(results) >= limit:
            break

    if not results:
        return [TextContent(type="text", text="No bookmarks found matching your criteria.")]

    output_lines = [f"Found {len(results)} bookmark(s):\n"]
    for b in results:
        output_lines.extend(format_bookmark(b))

    # Add rate limit info
    output_lines.append(f"\n_{client.get_rate_limit_status()}_")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def get_recent(tag: Optional[str] = None, limit: int = 10) -> list[TextContent]:
    """Get most recently shared bookmarks."""
    client = get_client()

    # Combine bloom-houston with optional additional tag
    combined_tag = COLLECTIVE_TAG
    if tag:
        combined_tag = f"{COLLECTIVE_TAG}+{tag}"

    bookmarks = await client.get_recent(tag=combined_tag, count=limit)

    if not bookmarks:
        return [TextContent(type="text", text="No recent bookmarks found.")]

    output_lines = [f"Most recent {len(bookmarks)} bookmark(s):\n"]
    for b in bookmarks:
        output_lines.extend(format_bookmark(b))

    output_lines.append(f"\n_{client.get_rate_limit_status()}_")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def who_knows_about(topic: str) -> list[TextContent]:
    """Find contributors by topic."""
    client = get_client()

    # Search for the topic within collective bookmarks
    bookmarks = await client.search_site(topic, tag=COLLECTIVE_TAG, count=100)

    # Group by user
    user_matches = defaultdict(list)
    for b in bookmarks:
        user = get_bookmark_user(b)
        user_matches[user].append(b)

    if not user_matches:
        return [TextContent(type="text", text=f"No contributors found with bookmarks about '{topic}'.")]

    # Sort by number of relevant bookmarks
    sorted_users = sorted(user_matches.items(), key=lambda x: len(x[1]), reverse=True)

    output_lines = [f"Contributors who know about '{topic}':\n"]
    for user, user_bookmarks in sorted_users:
        output_lines.append(f"**{user}** - {len(user_bookmarks)} bookmark(s)")
        for b in user_bookmarks[:3]:
            title = extract_bookmark_field(b, "description", "d") or "No title"
            output_lines.append(f"  - {title}")
        if len(user_bookmarks) > 3:
            output_lines.append(f"  - ... and {len(user_bookmarks) - 3} more")
        output_lines.append("")

    output_lines.append(f"\n_{client.get_rate_limit_status()}_")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def find_connections(user: Optional[str] = None, min_overlap: int = 2) -> list[TextContent]:
    """Find interest overlaps between members."""
    bookmarks = await fetch_collective_bookmarks()
    client = get_client()

    # Build tag sets per user (excluding bloom-houston itself)
    user_tags = defaultdict(set)
    for b in bookmarks:
        username = get_bookmark_user(b)
        tags = set(
            t.lower() for t in get_bookmark_tags(b)
            if t.lower() != COLLECTIVE_TAG
        )
        user_tags[username].update(tags)

    if user:
        # Find connections for specific user
        user_lower = user.lower()
        matching_user = None
        for u in user_tags:
            if u.lower() == user_lower:
                matching_user = u
                break

        if not matching_user:
            return [TextContent(type="text", text=f"User '{user}' not found in the collective.")]

        target_tags = user_tags[matching_user]
        connections = []

        for other_user, other_tags in user_tags.items():
            if other_user.lower() == user_lower:
                continue

            shared = target_tags & other_tags
            if len(shared) >= min_overlap:
                connections.append((other_user, shared))

        if not connections:
            return [TextContent(type="text", text=f"No connections found for '{user}' with at least {min_overlap} shared tags.")]

        connections.sort(key=lambda x: len(x[1]), reverse=True)

        output_lines = [f"Connections for {matching_user}:\n"]
        for other, shared in connections:
            output_lines.append(f"**{other}** - {len(shared)} shared interest(s)")
            output_lines.append(f"  Shared tags: {', '.join(sorted(shared))}")
            output_lines.append("")

        output_lines.append(f"\n_{client.get_rate_limit_status()}_")
        return [TextContent(type="text", text="\n".join(output_lines))]

    else:
        # Show overall network connections
        all_connections = []
        users = list(user_tags.keys())

        for i, user1 in enumerate(users):
            for user2 in users[i+1:]:
                shared = user_tags[user1] & user_tags[user2]
                if len(shared) >= min_overlap:
                    all_connections.append((user1, user2, shared))

        if not all_connections:
            return [TextContent(type="text", text=f"No connections found with at least {min_overlap} shared tags.")]

        all_connections.sort(key=lambda x: len(x[2]), reverse=True)

        output_lines = [f"Network connections (min {min_overlap} shared tags):\n"]
        for user1, user2, shared in all_connections[:20]:
            output_lines.append(f"**{user1}** <-> **{user2}** ({len(shared)} shared)")
            output_lines.append(f"  Tags: {', '.join(sorted(shared))}")
            output_lines.append("")

        output_lines.append(f"\n_{client.get_rate_limit_status()}_")
        return [TextContent(type="text", text="\n".join(output_lines))]


async def who_else_saved(url: str) -> list[TextContent]:
    """Check if URL was saved by others in the network."""
    client = get_client()

    # Use the URL endpoint to get info about this URL
    try:
        url_info = await client.get_url_info(url)
    except PinboardAPIError:
        # Fall back to searching collective bookmarks
        url_info = None

    # Also check our collective bookmarks
    bookmarks = await fetch_collective_bookmarks()
    normalized_target = normalize_url(url)

    savers = []
    for b in bookmarks:
        bookmark_url = extract_bookmark_field(b, "href", "u")
        if normalize_url(bookmark_url) == normalized_target:
            savers.append({
                "user": get_bookmark_user(b),
                "title": extract_bookmark_field(b, "description", "d") or "No title",
                "tags": get_bookmark_tags(b),
                "notes": extract_bookmark_field(b, "extended", "n"),
                "date": extract_bookmark_field(b, "time", "dt") or "Unknown"
            })

    if not savers:
        msg = f"No one in the bloom-houston collective has saved this URL: {url}"
        if url_info and url_info.get("status") == "ok":
            total_saves = url_info.get("total_saves", url_info.get("count", 0))
            if total_saves:
                msg += f"\n\n(Note: {total_saves} users sitewide have saved this URL)"
        return [TextContent(type="text", text=msg)]

    output_lines = [f"Found {len(savers)} member(s) who saved this URL:\n"]
    for s in savers:
        output_lines.append(f"**{s['user']}**")
        output_lines.append(f"  Title: {s['title']}")
        output_lines.append(f"  Tags: {', '.join(s['tags']) if s['tags'] else 'none'}")
        if s['notes']:
            output_lines.append(f"  Notes: {s['notes']}")
        output_lines.append(f"  Saved: {s['date']}")
        output_lines.append("")

    output_lines.append(f"\n_{client.get_rate_limit_status()}_")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def get_network_map(format: str = "json", min_shared_tags: int = 1) -> list[TextContent]:
    """Generate a relationship graph."""
    bookmarks = await fetch_collective_bookmarks()
    client = get_client()

    # Build user data
    user_tags = defaultdict(set)
    user_bookmark_count = defaultdict(int)

    for b in bookmarks:
        username = get_bookmark_user(b)
        tags = set(
            t.lower() for t in get_bookmark_tags(b)
            if t.lower() != COLLECTIVE_TAG
        )
        user_tags[username].update(tags)
        user_bookmark_count[username] += 1

    # Build connections
    users = list(user_tags.keys())
    connections = []

    for i, user1 in enumerate(users):
        for user2 in users[i+1:]:
            shared = user_tags[user1] & user_tags[user2]
            if len(shared) >= min_shared_tags:
                connections.append({
                    "source": user1,
                    "target": user2,
                    "shared_tags": sorted(shared),
                    "weight": len(shared)
                })

    if format == "mermaid":
        # Generate Mermaid flowchart
        lines = ["graph TD"]

        # Add nodes with bookmark counts
        for username in users:
            safe_name = username.replace("-", "_").replace(".", "_").replace(" ", "_")
            lines.append(f"    {safe_name}[\"{username}<br/>{user_bookmark_count[username]} bookmarks\"]")

        # Add edges
        for conn in connections:
            source = conn["source"].replace("-", "_").replace(".", "_").replace(" ", "_")
            target = conn["target"].replace("-", "_").replace(".", "_").replace(" ", "_")
            weight = conn["weight"]
            lines.append(f"    {source} ---|{weight} shared| {target}")

        lines.append("")
        lines.append(f"_{client.get_rate_limit_status()}_")

        return [TextContent(type="text", text="\n".join(lines))]

    else:
        # JSON format
        network = {
            "nodes": [
                {
                    "id": username,
                    "bookmark_count": user_bookmark_count[username],
                    "tags": sorted(user_tags[username])
                }
                for username in users
            ],
            "edges": connections,
            "stats": {
                "total_members": len(users),
                "total_connections": len(connections),
                "total_bookmarks": sum(user_bookmark_count.values())
            },
            "rate_limit": {
                "requests_remaining": client.rate_limit.requests_remaining,
                "reset_time": client.rate_limit.reset_time
            }
        }

        return [TextContent(type="text", text=json.dumps(network, indent=2))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
