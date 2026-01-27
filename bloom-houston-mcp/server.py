"""
Bloom Houston Collective Knowledge MCP Server

An MCP server that queries Pinboard's public feeds and API v1 for bookmarks tagged
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
PINBOARD_API_BASE = "https://api.pinboard.in/v1"
PINBOARD_FEEDS_BASE = "https://feeds.pinboard.in"
COLLECTIVE_TAG = "bloom-houston"
CACHE_TTL_SECONDS = 300  # 5 minutes


@dataclass
class CachedData:
    """Cached API response with timestamp."""
    data: any
    timestamp: float


class PinboardClient:
    """Client for Pinboard API v1 and public feeds."""

    def __init__(self, auth_token: str):
        self.auth_token = auth_token
        self._cache: dict[str, CachedData] = {}

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache:
            return False
        return (time.time() - self._cache[cache_key].timestamp) < CACHE_TTL_SECONDS

    def _get_cached(self, cache_key: str) -> Optional[any]:
        """Get cached data if valid."""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key].data
        return None

    def _set_cache(self, cache_key: str, data: any) -> None:
        """Cache data with current timestamp."""
        self._cache[cache_key] = CachedData(data=data, timestamp=time.time())

    async def get_tag_feed(self, tag: str = COLLECTIVE_TAG, count: int = 100) -> list[dict]:
        """Get public bookmarks for a tag using the JSON feed."""
        cache_key = f"feed:{tag}:{count}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Use public JSON feed - no auth required
        url = f"{PINBOARD_FEEDS_BASE}/json/t:{tag}/"
        params = {"count": count}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                self._set_cache(cache_key, data)
                return data
            except httpx.HTTPStatusError as e:
                raise Exception(f"Feed error: {e.response.status_code}")
            except httpx.RequestError as e:
                raise Exception(f"Network error: {str(e)}")
            except json.JSONDecodeError:
                raise Exception("Invalid JSON in feed response")

    async def get_recent_feed(self, count: int = 50) -> list[dict]:
        """Get recent public bookmarks from the recent feed."""
        cache_key = f"recent:{count}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        url = f"{PINBOARD_FEEDS_BASE}/json/recent/"
        params = {"count": count}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                self._set_cache(cache_key, data)
                return data
            except Exception as e:
                raise Exception(f"Feed error: {str(e)}")

    async def api_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make authenticated request to Pinboard API v1."""
        if params is None:
            params = {}
        
        # Add auth token and format
        params["auth_token"] = self.auth_token
        params["format"] = "json"

        url = f"{PINBOARD_API_BASE}{endpoint}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, params=params, timeout=30.0)
                
                if response.status_code == 401:
                    raise Exception("Authentication failed. Check your API token.")
                elif response.status_code == 429:
                    raise Exception("Rate limited. Please wait before making more requests.")
                
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                raise Exception(f"API error: {e.response.status_code}")
            except httpx.RequestError as e:
                raise Exception(f"Network error: {str(e)}")


# Global client instance
_client: Optional[PinboardClient] = None


def get_client() -> PinboardClient:
    """Get or create the Pinboard client."""
    global _client
    if _client is None:
        auth_token = os.environ.get("PINBOARD_API_TOKEN")
        if not auth_token:
            raise Exception(
                "PINBOARD_API_TOKEN environment variable not set. "
                "Set it to your Pinboard API token (format: username:token)"
            )
        _client = PinboardClient(auth_token)
    return _client


def normalize_url(url: str) -> str:
    """Normalize URL for comparison."""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    return f"{parsed.scheme}://{parsed.netloc}{path}".lower()


def extract_field(bookmark: dict, *keys) -> Optional[str]:
    """Extract a field from bookmark, trying multiple possible keys."""
    for key in keys:
        if key in bookmark and bookmark[key]:
            return bookmark[key]
    return None


def get_bookmark_user(bookmark: dict) -> str:
    """Extract username from bookmark."""
    return extract_field(bookmark, "a", "user", "author") or "Unknown"


def get_bookmark_tags(bookmark: dict) -> list[str]:
    """Extract tags from bookmark."""
    tags = extract_field(bookmark, "t", "tags")
    if tags is None:
        return []
    if isinstance(tags, str):
        return tags.split() if tags else []
    if isinstance(tags, list):
        return tags
    return []


def get_bookmark_url(bookmark: dict) -> str:
    """Extract URL from bookmark."""
    return extract_field(bookmark, "u", "href", "url") or ""


def get_bookmark_title(bookmark: dict) -> str:
    """Extract title from bookmark."""
    return extract_field(bookmark, "d", "description", "title") or "No title"


def get_bookmark_description(bookmark: dict) -> str:
    """Extract description/notes from bookmark."""
    return extract_field(bookmark, "n", "extended", "notes") or ""


def get_bookmark_date(bookmark: dict) -> str:
    """Extract date from bookmark."""
    return extract_field(bookmark, "dt", "time", "date") or "Unknown"


def format_bookmark(b: dict) -> list[str]:
    """Format a bookmark for display."""
    lines = []
    title = get_bookmark_title(b)
    url = get_bookmark_url(b)
    user = get_bookmark_user(b)
    tags = get_bookmark_tags(b)
    description = get_bookmark_description(b)
    date = get_bookmark_date(b)

    lines.append(f"**{title}**")
    lines.append(f"  URL: {url}")
    lines.append(f"  Saved by: {user}")
    lines.append(f"  Tags: {', '.join(tags) if tags else 'none'}")
    if description:
        lines.append(f"  Notes: {description}")
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


async def fetch_collective_bookmarks() -> list[dict]:
    """Fetch all bloom-houston tagged bookmarks from public feed."""
    client = get_client()
    return await client.get_tag_feed(COLLECTIVE_TAG, count=100)


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
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def search_collective(
    query: Optional[str] = None,
    tag: Optional[str] = None,
    user: Optional[str] = None,
    limit: int = 20
) -> list[TextContent]:
    """Search across collective bookmarks."""
    bookmarks = await fetch_collective_bookmarks()

    # Filter bookmarks
    results = []
    for b in bookmarks:
        # Filter by user
        if user and get_bookmark_user(b).lower() != user.lower():
            continue

        # Filter by additional tag
        if tag:
            bookmark_tags = [t.lower() for t in get_bookmark_tags(b)]
            if tag.lower() not in bookmark_tags:
                continue

        # Filter by search query
        if query:
            query_lower = query.lower()
            title = get_bookmark_title(b).lower()
            description = get_bookmark_description(b).lower()
            url = get_bookmark_url(b).lower()
            tags = " ".join(get_bookmark_tags(b)).lower()

            if not any(query_lower in field for field in [title, description, url, tags]):
                continue

        results.append(b)

        if len(results) >= limit:
            break

    if not results:
        return [TextContent(type="text", text="No matching bookmarks found.")]

    output_lines = [f"Found {len(results)} matching bookmark(s):\n"]
    for b in results:
        output_lines.extend(format_bookmark(b))

    return [TextContent(type="text", text="\n".join(output_lines))]


async def get_recent(tag: Optional[str] = None, limit: int = 10) -> list[TextContent]:
    """Get most recent bookmarks from the collective."""
    bookmarks = await fetch_collective_bookmarks()

    # Filter by additional tag if specified
    if tag:
        bookmarks = [
            b for b in bookmarks
            if tag.lower() in [t.lower() for t in get_bookmark_tags(b)]
        ]

    # Take most recent (feed is usually already sorted)
    recent = bookmarks[:limit]

    if not recent:
        msg = "No recent bookmarks found"
        if tag:
            msg += f" with tag '{tag}'"
        return [TextContent(type="text", text=msg + ".")]

    output_lines = [f"Recent bloom-houston bookmarks:\n"]
    for b in recent:
        output_lines.extend(format_bookmark(b))

    return [TextContent(type="text", text="\n".join(output_lines))]


async def who_knows_about(topic: str) -> list[TextContent]:
    """Find contributors who know about a topic."""
    bookmarks = await fetch_collective_bookmarks()
    topic_lower = topic.lower()

    # Count matches per user
    user_matches = defaultdict(list)

    for b in bookmarks:
        title = get_bookmark_title(b).lower()
        description = get_bookmark_description(b).lower()
        tags = " ".join(get_bookmark_tags(b)).lower()

        if topic_lower in title or topic_lower in description or topic_lower in tags:
            username = get_bookmark_user(b)
            user_matches[username].append(b)

    if not user_matches:
        return [TextContent(type="text", text=f"No one in the collective has bookmarks about '{topic}'.")]

    # Sort by number of matches
    sorted_users = sorted(user_matches.items(), key=lambda x: len(x[1]), reverse=True)

    output_lines = [f"Contributors who know about '{topic}':\n"]
    for username, matches in sorted_users:
        output_lines.append(f"**{username}** - {len(matches)} bookmark(s)")
        for b in matches[:3]:  # Show up to 3 examples
            output_lines.append(f"  • {get_bookmark_title(b)}")
        if len(matches) > 3:
            output_lines.append(f"  ... and {len(matches) - 3} more")
        output_lines.append("")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def find_connections(user: Optional[str] = None, min_overlap: int = 2) -> list[TextContent]:
    """Find interest overlaps between members."""
    bookmarks = await fetch_collective_bookmarks()

    # Build user -> tags mapping
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

        return [TextContent(type="text", text="\n".join(output_lines))]


async def who_else_saved(url: str) -> list[TextContent]:
    """Check if URL was saved by others in the network."""
    bookmarks = await fetch_collective_bookmarks()
    normalized_target = normalize_url(url)

    savers = []
    for b in bookmarks:
        bookmark_url = get_bookmark_url(b)
        if normalize_url(bookmark_url) == normalized_target:
            savers.append({
                "user": get_bookmark_user(b),
                "title": get_bookmark_title(b),
                "tags": get_bookmark_tags(b),
                "notes": get_bookmark_description(b),
                "date": get_bookmark_date(b)
            })

    if not savers:
        return [TextContent(type="text", text=f"No one in the bloom-houston collective has saved this URL: {url}")]

    output_lines = [f"Found {len(savers)} member(s) who saved this URL:\n"]
    for s in savers:
        output_lines.append(f"**{s['user']}**")
        output_lines.append(f"  Title: {s['title']}")
        output_lines.append(f"  Tags: {', '.join(s['tags']) if s['tags'] else 'none'}")
        if s['notes']:
            output_lines.append(f"  Notes: {s['notes']}")
        output_lines.append(f"  Saved: {s['date']}")
        output_lines.append("")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def get_network_map(format: str = "json", min_shared_tags: int = 1) -> list[TextContent]:
    """Generate a relationship graph."""
    bookmarks = await fetch_collective_bookmarks()

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
