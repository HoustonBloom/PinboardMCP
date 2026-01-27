"""
Bloom Houston Collective Knowledge MCP Server

An MCP server that queries Pinboard's public feed for bookmarks tagged
bloom-houston, exposing collective learning and contributor relationships.
"""

import asyncio
import httpx
from collections import defaultdict
from typing import Optional
from urllib.parse import urlparse

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Constants
PINBOARD_FEED_URL = "https://feeds.pinboard.in/json/t:bloom-houston/"
CACHE_TTL_SECONDS = 300  # 5 minutes

# Global cache
_cache = {
    "data": None,
    "timestamp": 0
}


async def fetch_bookmarks() -> list[dict]:
    """Fetch bookmarks from Pinboard public feed with caching."""
    import time

    current_time = time.time()
    if _cache["data"] is not None and (current_time - _cache["timestamp"]) < CACHE_TTL_SECONDS:
        return _cache["data"]

    async with httpx.AsyncClient() as client:
        response = await client.get(PINBOARD_FEED_URL, timeout=30.0)
        response.raise_for_status()
        bookmarks = response.json()

    _cache["data"] = bookmarks
    _cache["timestamp"] = current_time

    return bookmarks


def normalize_url(url: str) -> str:
    """Normalize URL for comparison."""
    parsed = urlparse(url)
    # Remove trailing slashes and normalize
    path = parsed.path.rstrip('/')
    return f"{parsed.scheme}://{parsed.netloc}{path}".lower()


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


async def search_collective(
    query: Optional[str] = None,
    tag: Optional[str] = None,
    user: Optional[str] = None,
    limit: int = 20
) -> list[TextContent]:
    """Search across all bloom-houston bookmarks."""
    bookmarks = await fetch_bookmarks()
    results = []

    for bookmark in bookmarks:
        # Apply filters
        if user and bookmark.get("a", "").lower() != user.lower():
            continue

        if tag:
            tags = [t.lower() for t in bookmark.get("t", [])]
            if tag.lower() not in tags:
                continue

        if query:
            query_lower = query.lower()
            searchable = " ".join([
                bookmark.get("d", ""),  # title/description
                bookmark.get("n", ""),  # extended description
                bookmark.get("u", ""),  # url
                " ".join(bookmark.get("t", []))  # tags
            ]).lower()

            if query_lower not in searchable:
                continue

        results.append(bookmark)

        if len(results) >= limit:
            break

    if not results:
        return [TextContent(type="text", text="No bookmarks found matching your criteria.")]

    output_lines = [f"Found {len(results)} bookmark(s):\n"]
    for b in results:
        tags = ", ".join(b.get("t", []))
        output_lines.append(f"**{b.get('d', 'No title')}**")
        output_lines.append(f"  URL: {b.get('u', 'N/A')}")
        output_lines.append(f"  Saved by: {b.get('a', 'Unknown')}")
        output_lines.append(f"  Tags: {tags}")
        if b.get("n"):
            output_lines.append(f"  Notes: {b.get('n')}")
        output_lines.append("")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def get_recent(tag: Optional[str] = None, limit: int = 10) -> list[TextContent]:
    """Get most recently shared bookmarks."""
    bookmarks = await fetch_bookmarks()

    # Filter by tag if specified
    if tag:
        bookmarks = [
            b for b in bookmarks
            if tag.lower() in [t.lower() for t in b.get("t", [])]
        ]

    # Pinboard feed is already sorted by most recent
    results = bookmarks[:limit]

    if not results:
        return [TextContent(type="text", text="No recent bookmarks found.")]

    output_lines = [f"Most recent {len(results)} bookmark(s):\n"]
    for b in results:
        tags = ", ".join(b.get("t", []))
        output_lines.append(f"**{b.get('d', 'No title')}**")
        output_lines.append(f"  URL: {b.get('u', 'N/A')}")
        output_lines.append(f"  Saved by: {b.get('a', 'Unknown')}")
        output_lines.append(f"  Tags: {tags}")
        output_lines.append(f"  Date: {b.get('dt', 'Unknown')}")
        output_lines.append("")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def who_knows_about(topic: str) -> list[TextContent]:
    """Find contributors by topic."""
    bookmarks = await fetch_bookmarks()
    topic_lower = topic.lower()

    # Count bookmarks per user matching the topic
    user_matches = defaultdict(list)

    for b in bookmarks:
        searchable = " ".join([
            b.get("d", ""),
            b.get("n", ""),
            " ".join(b.get("t", []))
        ]).lower()

        if topic_lower in searchable:
            user = b.get("a", "Unknown")
            user_matches[user].append(b)

    if not user_matches:
        return [TextContent(type="text", text=f"No contributors found with bookmarks about '{topic}'.")]

    # Sort by number of relevant bookmarks
    sorted_users = sorted(user_matches.items(), key=lambda x: len(x[1]), reverse=True)

    output_lines = [f"Contributors who know about '{topic}':\n"]
    for user, user_bookmarks in sorted_users:
        output_lines.append(f"**{user}** - {len(user_bookmarks)} bookmark(s)")
        for b in user_bookmarks[:3]:  # Show up to 3 examples
            output_lines.append(f"  - {b.get('d', 'No title')}")
        if len(user_bookmarks) > 3:
            output_lines.append(f"  - ... and {len(user_bookmarks) - 3} more")
        output_lines.append("")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def find_connections(user: Optional[str] = None, min_overlap: int = 2) -> list[TextContent]:
    """Find interest overlaps between members."""
    bookmarks = await fetch_bookmarks()

    # Build tag sets per user
    user_tags = defaultdict(set)
    for b in bookmarks:
        username = b.get("a", "Unknown")
        tags = set(t.lower() for t in b.get("t", []) if t.lower() != "bloom-houston")
        user_tags[username].update(tags)

    if user:
        # Find connections for specific user
        if user not in user_tags:
            return [TextContent(type="text", text=f"User '{user}' not found in the collective.")]

        target_tags = user_tags[user]
        connections = []

        for other_user, other_tags in user_tags.items():
            if other_user == user:
                continue

            shared = target_tags & other_tags
            if len(shared) >= min_overlap:
                connections.append((other_user, shared))

        if not connections:
            return [TextContent(type="text", text=f"No connections found for '{user}' with at least {min_overlap} shared tags.")]

        connections.sort(key=lambda x: len(x[1]), reverse=True)

        output_lines = [f"Connections for {user}:\n"]
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
        for user1, user2, shared in all_connections[:20]:  # Limit to top 20
            output_lines.append(f"**{user1}** <-> **{user2}** ({len(shared)} shared)")
            output_lines.append(f"  Tags: {', '.join(sorted(shared))}")
            output_lines.append("")

        return [TextContent(type="text", text="\n".join(output_lines))]


async def who_else_saved(url: str) -> list[TextContent]:
    """Check if URL was saved by others in the network."""
    bookmarks = await fetch_bookmarks()
    normalized_target = normalize_url(url)

    savers = []
    for b in bookmarks:
        if normalize_url(b.get("u", "")) == normalized_target:
            savers.append({
                "user": b.get("a", "Unknown"),
                "title": b.get("d", "No title"),
                "tags": b.get("t", []),
                "notes": b.get("n", ""),
                "date": b.get("dt", "Unknown")
            })

    if not savers:
        return [TextContent(type="text", text=f"No one in the collective has saved this URL: {url}")]

    output_lines = [f"Found {len(savers)} member(s) who saved this URL:\n"]
    for s in savers:
        output_lines.append(f"**{s['user']}**")
        output_lines.append(f"  Title: {s['title']}")
        output_lines.append(f"  Tags: {', '.join(s['tags'])}")
        if s['notes']:
            output_lines.append(f"  Notes: {s['notes']}")
        output_lines.append(f"  Saved: {s['date']}")
        output_lines.append("")

    return [TextContent(type="text", text="\n".join(output_lines))]


async def get_network_map(format: str = "json", min_shared_tags: int = 1) -> list[TextContent]:
    """Generate a relationship graph."""
    bookmarks = await fetch_bookmarks()

    # Build user data
    user_tags = defaultdict(set)
    user_bookmark_count = defaultdict(int)

    for b in bookmarks:
        username = b.get("a", "Unknown")
        tags = set(t.lower() for t in b.get("t", []) if t.lower() != "bloom-houston")
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
            safe_name = username.replace("-", "_").replace(".", "_")
            lines.append(f"    {safe_name}[{username}<br/>{user_bookmark_count[username]} bookmarks]")

        # Add edges
        for conn in connections:
            source = conn["source"].replace("-", "_").replace(".", "_")
            target = conn["target"].replace("-", "_").replace(".", "_")
            weight = conn["weight"]
            lines.append(f"    {source} ---|{weight} shared| {target}")

        return [TextContent(type="text", text="\n".join(lines))]

    else:
        # JSON format
        import json

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
