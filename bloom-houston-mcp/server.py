#!/usr/bin/env python3
"""
Bloom Houston Pinboard MCP Server - Complete Version

Includes ALL original functionality PLUS new smart tools for reduced API calls.

ORIGINAL TOOLS:
- search_collective: Search bloom-houston bookmarks
- get_recent: Get recent bloom-houston bookmarks
- who_knows_about: Find contributors by topic
- find_connections: Find interest overlaps
- who_else_saved: Check who saved a URL
- get_network_map: Generate relationship graph
- search_public_pinboard: Search all public Pinboard bookmarks
- search_my_bookmarks: Search your personal bookmarks
- get_my_tags: Get your personal tag list
- search_user_bookmarks: Search a specific user's bookmarks
- save_bookmark: Save a new bookmark
- suggest_tags: Get tag suggestions for a URL
- get_feed_url: Generate RSS/JSON feed URLs

NEW SMART TOOLS (reduce API calls):
- get_user_profile: Complete user profile in ONE call
- get_collective_overview: Full collective snapshot in ONE call
- find_common_ground: Compare two users in ONE call
- clear_cache: Clear in-memory cache
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, quote

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

# Configuration
PINBOARD_API_TOKEN = os.environ.get("PINBOARD_API_TOKEN", "")
COLLECTIVE_TAG = "bloom-houston"
CACHE_TTL_SECONDS = 300  # 5 minutes

# Persistent storage for known members
DATA_DIR = Path(os.environ.get("BLOOM_DATA_DIR", Path.home() / ".bloom-houston"))
KNOWN_MEMBERS_FILE = DATA_DIR / "known_members.json"

# Simple in-memory cache
_cache = {}
_cache_timestamps = {}


# =============================================================================
# PERSISTENT MEMBER TRACKING
# =============================================================================

def ensure_data_dir():
    """Ensure the data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_known_members() -> dict:
    """Load known members from persistent storage."""
    ensure_data_dir()
    if KNOWN_MEMBERS_FILE.exists():
        try:
            with open(KNOWN_MEMBERS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"members": {}, "last_check": None}
    return {"members": {}, "last_check": None}


def save_known_members(data: dict):
    """Save known members to persistent storage."""
    ensure_data_dir()
    with open(KNOWN_MEMBERS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_known_member(username: str, first_seen: str, first_bookmark: dict):
    """Add a new member to the known members list."""
    data = load_known_members()
    if username.lower() not in {k.lower() for k in data["members"]}:
        data["members"][username] = {
            "first_seen": first_seen,
            "first_bookmark_title": first_bookmark.get("title", ""),
            "first_bookmark_url": first_bookmark.get("url", "")
        }
        save_known_members(data)
        return True
    return False

server = Server("bloom-houston")


# =============================================================================
# CACHING LAYER
# =============================================================================

def get_cached(key: str):
    """Get cached value if still valid."""
    if key in _cache and key in _cache_timestamps:
        if time.time() - _cache_timestamps[key] < CACHE_TTL_SECONDS:
            return _cache[key]
    return None


def set_cached(key: str, value):
    """Store value in cache."""
    _cache[key] = value
    _cache_timestamps[key] = time.time()


# =============================================================================
# PINBOARD API HELPERS
# =============================================================================

async def fetch_public_tag_feed(tag: str = COLLECTIVE_TAG, count: int = 100) -> list:
    """Fetch public bookmarks for a tag from Pinboard feeds."""
    cache_key = f"public_feed:{tag}:{count}"
    cached = get_cached(cache_key)
    if cached:
        return cached
    
    url = f"https://feeds.pinboard.in/json/t:{tag}/?count={count}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            set_cached(cache_key, data)
            return data
    return []


async def fetch_public_recent(count: int = 100) -> list:
    """Fetch recent public bookmarks from all of Pinboard."""
    cache_key = f"public_recent:{count}"
    cached = get_cached(cache_key)
    if cached:
        return cached
    
    url = f"https://feeds.pinboard.in/json/recent/?count={count}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            set_cached(cache_key, data)
            return data
    return []


async def fetch_user_bookmarks(username: str, count: int = 100, tag: str = None) -> list:
    """Fetch public bookmarks for a specific user."""
    cache_key = f"user_bookmarks:{username}:{tag}:{count}"
    cached = get_cached(cache_key)
    if cached:
        return cached
    
    if tag:
        url = f"https://feeds.pinboard.in/json/u:{username}/t:{tag}/?count={count}"
    else:
        url = f"https://feeds.pinboard.in/json/u:{username}/?count={count}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            set_cached(cache_key, data)
            return data
    return []


async def fetch_my_bookmarks(tag: str = None, count: int = 100) -> list:
    """Fetch authenticated user's bookmarks."""
    if not PINBOARD_API_TOKEN:
        return []
    
    cache_key = f"my_bookmarks:{tag}:{count}"
    cached = get_cached(cache_key)
    if cached:
        return cached
    
    url = f"https://api.pinboard.in/v1/posts/all?auth_token={PINBOARD_API_TOKEN}&format=json&count={count}"
    if tag:
        url += f"&tag={tag}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            set_cached(cache_key, data)
            return data
    return []


async def fetch_my_tags() -> dict:
    """Fetch authenticated user's tags with counts."""
    if not PINBOARD_API_TOKEN:
        return {}
    
    cache_key = "my_tags"
    cached = get_cached(cache_key)
    if cached:
        return cached
    
    url = f"https://api.pinboard.in/v1/tags/get?auth_token={PINBOARD_API_TOKEN}&format=json"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            set_cached(cache_key, data)
            return data
    return {}


async def save_bookmark_to_pinboard(url: str, title: str, description: str = "", 
                                     tags: list = None, private: bool = False, 
                                     toread: bool = False) -> dict:
    """Save a bookmark to the authenticated user's Pinboard."""
    if not PINBOARD_API_TOKEN:
        return {"error": "No API token configured"}
    
    params = {
        "auth_token": PINBOARD_API_TOKEN,
        "format": "json",
        "url": url,
        "description": title,  # Pinboard calls title "description"
        "extended": description,  # Pinboard calls notes "extended"
        "shared": "no" if private else "yes",
        "toread": "yes" if toread else "no"
    }
    
    if tags:
        params["tags"] = " ".join(tags)
    
    api_url = "https://api.pinboard.in/v1/posts/add"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(api_url, params=params)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"API error: {resp.status_code}"}


async def get_tag_suggestions(url: str) -> dict:
    """Get tag suggestions for a URL from Pinboard."""
    if not PINBOARD_API_TOKEN:
        return {"popular": [], "recommended": []}
    
    api_url = f"https://api.pinboard.in/v1/posts/suggest?auth_token={PINBOARD_API_TOKEN}&format=json&url={quote(url)}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(api_url)
        if resp.status_code == 200:
            data = resp.json()
            # Pinboard returns [{"popular": [...]}, {"recommended": [...]}]
            result = {"popular": [], "recommended": []}
            for item in data:
                if "popular" in item:
                    result["popular"] = item["popular"]
                if "recommended" in item:
                    result["recommended"] = item["recommended"]
            return result
    return {"popular": [], "recommended": []}


# =============================================================================
# DATA EXTRACTION HELPERS
# =============================================================================

def get_bookmark_user(b: dict) -> str:
    return b.get("a") or b.get("user") or b.get("author") or "unknown"


def get_bookmark_title(b: dict) -> str:
    return b.get("d") or b.get("description") or b.get("title") or "Untitled"


def get_bookmark_url(b: dict) -> str:
    return b.get("u") or b.get("href") or b.get("url") or ""


def get_bookmark_description(b: dict) -> str:
    return b.get("n") or b.get("extended") or b.get("notes") or ""


def get_bookmark_tags(b: dict) -> list:
    tags = b.get("t") or b.get("tags") or []
    if isinstance(tags, str):
        tags = tags.split()
    return [t for t in tags if t]


def get_bookmark_date(b: dict) -> str:
    return b.get("dt") or b.get("time") or b.get("date") or ""


def normalize_url(url: str) -> str:
    """Normalize URL for comparison."""
    parsed = urlparse(url.lower().rstrip("/"))
    return f"{parsed.netloc}{parsed.path}"


# =============================================================================
# SMART AGGREGATION FUNCTIONS
# =============================================================================

def build_user_profile(bookmarks: list, username: str) -> dict:
    """Build comprehensive profile from a user's bookmarks."""
    tag_counts = defaultdict(int)
    recent_bookmarks = []
    domains = defaultdict(int)
    
    for b in bookmarks:
        # For user feeds, all bookmarks belong to that user
        # Count tags
        for tag in get_bookmark_tags(b):
            if tag.lower() != COLLECTIVE_TAG:
                tag_counts[tag] += 1
        
        # Track domains
        url = get_bookmark_url(b)
        if url:
            domain = urlparse(url).netloc
            domains[domain] += 1
        
        # Collect recent bookmarks
        recent_bookmarks.append({
            "title": get_bookmark_title(b),
            "url": get_bookmark_url(b),
            "tags": [t for t in get_bookmark_tags(b) if t.lower() != COLLECTIVE_TAG],
            "notes": get_bookmark_description(b),
            "date": get_bookmark_date(b)
        })
    
    # Sort tags by count
    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
    top_tags = [{"tag": t, "count": c} for t, c in sorted_tags[:20]]
    
    # Top domains
    sorted_domains = sorted(domains.items(), key=lambda x: -x[1])
    top_domains = [{"domain": d, "count": c} for d, c in sorted_domains[:10]]
    
    return {
        "username": username,
        "total_bookmarks": len(recent_bookmarks),
        "tags": top_tags,
        "top_interests": [t["tag"] for t in top_tags[:10]],
        "top_domains": top_domains,
        "recent_bookmarks": recent_bookmarks[:20]
    }


def build_collective_overview(bookmarks: list) -> dict:
    """Build overview of entire collective from bookmarks."""
    contributors = defaultdict(lambda: {"count": 0, "tags": set(), "latest": ""})
    tag_counts = defaultdict(int)
    recent = []
    
    for b in bookmarks:
        user = get_bookmark_user(b)
        date = get_bookmark_date(b)
        tags = get_bookmark_tags(b)
        
        # Track contributor
        contributors[user]["count"] += 1
        contributors[user]["tags"].update(t for t in tags if t.lower() != COLLECTIVE_TAG)
        if date > contributors[user]["latest"]:
            contributors[user]["latest"] = date
        
        # Count tags
        for tag in tags:
            if tag.lower() != COLLECTIVE_TAG:
                tag_counts[tag] += 1
        
        # Recent bookmarks
        recent.append({
            "title": get_bookmark_title(b),
            "url": get_bookmark_url(b),
            "user": user,
            "tags": [t for t in tags if t.lower() != COLLECTIVE_TAG],
            "date": date
        })
    
    # Format contributors
    contributor_list = []
    for user, data in sorted(contributors.items(), key=lambda x: -x[1]["count"]):
        contributor_list.append({
            "username": user,
            "bookmark_count": data["count"],
            "interests": list(data["tags"])[:10],
            "last_active": data["latest"]
        })
    
    # Format tags
    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
    tag_list = [{"tag": t, "count": c} for t, c in sorted_tags[:30]]
    
    return {
        "total_bookmarks": len(bookmarks),
        "total_contributors": len(contributors),
        "contributors": contributor_list,
        "popular_tags": tag_list,
        "recent_activity": recent[:15]
    }


def find_user_overlaps(profile_a: dict, profile_b: dict) -> dict:
    """Find common ground between two user profiles."""
    tags_a = set(t["tag"].lower() for t in profile_a.get("tags", []))
    tags_b = set(t["tag"].lower() for t in profile_b.get("tags", []))
    
    shared_tags = tags_a & tags_b
    
    # Find shared URLs
    urls_a = {normalize_url(b["url"]): b for b in profile_a.get("recent_bookmarks", [])}
    urls_b = {normalize_url(b["url"]): b for b in profile_b.get("recent_bookmarks", [])}
    shared_urls = set(urls_a.keys()) & set(urls_b.keys())
    
    shared_bookmarks = []
    for url in shared_urls:
        shared_bookmarks.append({
            "url": urls_a[url]["url"],
            "title": urls_a[url]["title"],
            "saved_by_both": True
        })
    
    return {
        "user_a": profile_a["username"],
        "user_b": profile_b["username"],
        "shared_interests": list(shared_tags),
        "shared_interest_count": len(shared_tags),
        "unique_to_a": list(tags_a - tags_b)[:10],
        "unique_to_b": list(tags_b - tags_a)[:10],
        "shared_bookmarks": shared_bookmarks,
        "compatibility_score": round(len(shared_tags) / max(len(tags_a | tags_b), 1) * 100, 1)
    }


# =============================================================================
# MCP TOOL DEFINITIONS
# =============================================================================

@server.list_tools()
async def list_tools():
    return [
        # =================================================================
        # NEW SMART OVERVIEW TOOLS (reduce API calls)
        # =================================================================
        Tool(
            name="get_user_profile",
            description=(
                "Get a COMPLETE profile of a Pinboard user in ONE call. "
                "Returns all their tags (with counts), top interests, recent bookmarks, "
                "and favorite domains. Use this FIRST before asking specific questions "
                "about a user - it eliminates the need for multiple tag searches."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Pinboard username to look up"
                    }
                },
                "required": ["username"]
            }
        ),
        Tool(
            name="get_collective_overview",
            description=(
                "Get a COMPLETE snapshot of the bloom-houston collective in ONE call. "
                "Returns all contributors (with bookmark counts and interests), popular tags, "
                "and recent activity. Use this to understand the community landscape "
                "before drilling into specifics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max bookmarks to analyze (default 100)",
                        "default": 100
                    }
                }
            }
        ),
        Tool(
            name="find_common_ground",
            description=(
                "Compare two users and find their shared interests in ONE call. "
                "Returns overlapping tags, shared bookmarks, unique interests, "
                "and a compatibility score. Much faster than fetching profiles separately."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "user_a": {
                        "type": "string",
                        "description": "First Pinboard username"
                    },
                    "user_b": {
                        "type": "string",
                        "description": "Second Pinboard username"
                    }
                },
                "required": ["user_a", "user_b"]
            }
        ),
        
        # =================================================================
        # COLLECTIVE (bloom-houston) TOOLS
        # =================================================================
        Tool(
            name="search_collective",
            description=(
                "Search across all bloom-houston bookmarks by text, tag, or user. "
                "Returns matching bookmarks with titles, URLs, descriptions, and who saved them."
            ),
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
                }
            }
        ),
        Tool(
            name="get_recent",
            description=(
                "Get the most recently shared bookmarks in the bloom-houston collective. "
                "Optionally filter by an additional tag."
            ),
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
                }
            }
        ),
        Tool(
            name="who_knows_about",
            description=(
                "Find contributors who have saved bookmarks about a specific topic. "
                "Returns users ranked by how many relevant bookmarks they've shared."
            ),
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
            description=(
                "Find interest overlaps between members of the collective. "
                "Shows which users share similar interests based on common tags."
            ),
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
                }
            }
        ),
        Tool(
            name="who_else_saved",
            description=(
                "Check if a URL was saved by others in the bloom-houston network. "
                "Useful for finding who else is interested in a specific resource."
            ),
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
            description=(
                "Generate a relationship graph showing connections between members "
                "based on shared interests. Returns JSON or Mermaid diagram format."
            ),
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
                }
            }
        ),
        
        # =================================================================
        # PERSONAL BOOKMARKS TOOLS
        # =================================================================
        Tool(
            name="search_my_bookmarks",
            description=(
                "Search YOUR personal Pinboard bookmarks (all bookmarks, any tag). "
                "This searches your entire Pinboard collection, not just bloom-houston tagged items."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to match against title, description, tags, or URL"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by specific tag"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 20)",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="get_my_tags",
            description=(
                "Get a list of all tags you've used in your Pinboard, with counts. "
                "Useful for exploring your own bookmark organization."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        # =================================================================
        # PUBLIC PINBOARD TOOLS
        # =================================================================
        Tool(
            name="search_user_bookmarks",
            description=(
                "Search any Pinboard user's PUBLIC bookmarks by username. "
                "Uses the efficient JSON feed endpoint (no auth required). "
                "Great for exploring a friend's bookmarks or finding experts on a topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Pinboard username to search (e.g., 'cyberchucktx')"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search term to match against title, description, tags, or URL"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by specific tag"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 20)",
                        "default": 20
                    }
                },
                "required": ["username"]
            }
        ),
        Tool(
            name="search_public_pinboard",
            description=(
                "Search recent PUBLIC bookmarks across all of Pinboard (not just bloom-houston). "
                "Useful for discovering what others are bookmarking on any topic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search term to match against title, description, tags, or URL"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by specific tag (searches public feed for this tag)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 20)",
                        "default": 20
                    }
                }
            }
        ),
        
        # =================================================================
        # BOOKMARK MANAGEMENT TOOLS
        # =================================================================
        Tool(
            name="save_bookmark",
            description=(
                "Save a URL to YOUR Pinboard account with optional tags and description. "
                "Use this after finding interesting bookmarks from other users or searches."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to bookmark"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the bookmark"
                    },
                    "description": {
                        "type": "string",
                        "description": "Extended description or notes about the bookmark"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of tags (e.g., ['ai', 'community', 'houston'])"
                    },
                    "private": {
                        "type": "boolean",
                        "description": "Make bookmark private (default: false, bookmark is public)",
                        "default": False
                    },
                    "toread": {
                        "type": "boolean",
                        "description": "Mark as 'to read' / unread (default: false)",
                        "default": False
                    }
                },
                "required": ["url", "title"]
            }
        ),
        Tool(
            name="suggest_tags",
            description=(
                "Get tag suggestions for a URL before saving it. "
                "Returns popular tags (used by others for this URL) and recommended tags "
                "(from your own tag vocabulary that might fit)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to get tag suggestions for"
                    }
                },
                "required": ["url"]
            }
        ),
        
        # =================================================================
        # UTILITY TOOLS
        # =================================================================
        Tool(
            name="get_feed_url",
            description=(
                "Generate RSS or JSON feed URLs for subscribing to Pinboard content. "
                "Use RSS URLs for feed readers (Feedly, Inoreader, etc.) or automation (Zapier, IFTTT). "
                "Use JSON URLs for programmatic access."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "Pinboard username (optional - omit to get tag-only feed)"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Tag to filter by (optional)"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["rss", "json"],
                        "description": "Feed format: 'rss' for feed readers, 'json' for code (default: rss)",
                        "default": "rss"
                    }
                }
            }
        ),
        Tool(
            name="clear_cache",
            description=(
                "Clear the in-memory cache to force fresh API calls. "
                "Use this if you've just added bookmarks and want to see them immediately."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        # =================================================================
        # MEMBER DISCOVERY TOOLS
        # =================================================================
        Tool(
            name="discover_new_members",
            description=(
                "Check for NEW contributors to any bloom-related tag. "
                "Compares current contributors against a stored list of known members "
                "and returns anyone who hasn't been seen before. "
                "Great for community onboarding - discover new people joining the network!"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tag": {
                        "type": "string",
                        "description": "Tag to monitor (default: 'bloom-houston'). Use 'bloom' for broader discovery.",
                        "default": "bloom-houston"
                    },
                    "auto_add": {
                        "type": "boolean",
                        "description": "Automatically add new members to known list (default: true)",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="list_known_members",
            description=(
                "List all known members of the bloom network that have been discovered. "
                "Shows when they were first seen and their first bookmark."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sort_by": {
                        "type": "string",
                        "enum": ["first_seen", "username"],
                        "description": "Sort order (default: first_seen, newest first)",
                        "default": "first_seen"
                    }
                }
            }
        ),
        Tool(
            name="reset_known_members",
            description=(
                "Clear the known members list to start fresh. "
                "Use with caution - this forgets all previously discovered members."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to confirm reset"
                    }
                },
                "required": ["confirm"]
            }
        )
    ]


# =============================================================================
# MCP TOOL IMPLEMENTATIONS
# =============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    
    # =================================================================
    # NEW SMART OVERVIEW TOOLS
    # =================================================================
    
    if name == "get_user_profile":
        username = arguments.get("username", "")
        if not username:
            return [TextContent(type="text", text="Error: username is required")]
        
        bookmarks = await fetch_user_bookmarks(username, 200)
        if not bookmarks:
            return [TextContent(type="text", text=f"No public bookmarks found for user '{username}'")]
        
        profile = build_user_profile(bookmarks, username)
        
        lines = [
            f"# Profile: {profile['username']}",
            f"**Total Bookmarks:** {profile['total_bookmarks']}",
            "",
            "## Top Interests",
            ", ".join(profile['top_interests']) if profile['top_interests'] else "None",
            "",
            "## All Tags (with counts)"
        ]
        for t in profile['tags']:
            lines.append(f"- {t['tag']}: {t['count']}")
        
        lines.extend(["", "## Top Domains"])
        for d in profile['top_domains']:
            lines.append(f"- {d['domain']}: {d['count']}")
        
        lines.extend(["", "## Recent Bookmarks"])
        for b in profile['recent_bookmarks'][:10]:
            tags_str = ", ".join(b['tags'][:5]) if b['tags'] else "none"
            lines.append(f"- **{b['title']}**")
            lines.append(f"  Tags: {tags_str}")
            if b['notes']:
                lines.append(f"  Notes: {b['notes'][:100]}...")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "get_collective_overview":
        limit = arguments.get("limit", 100)
        bookmarks = await fetch_public_tag_feed(COLLECTIVE_TAG, limit)
        
        if not bookmarks:
            return [TextContent(type="text", text="No bookmarks found in the bloom-houston collective")]
        
        overview = build_collective_overview(bookmarks)
        
        lines = [
            "# Bloom Houston Collective Overview",
            f"**Total Bookmarks:** {overview['total_bookmarks']}",
            f"**Contributors:** {overview['total_contributors']}",
            "",
            "## Contributors (by activity)"
        ]
        for c in overview['contributors'][:15]:
            interests = ", ".join(c['interests'][:5]) if c['interests'] else "none"
            lines.append(f"- **{c['username']}** ({c['bookmark_count']} bookmarks)")
            lines.append(f"  Interests: {interests}")
        
        lines.extend(["", "## Popular Tags"])
        for t in overview['popular_tags'][:20]:
            lines.append(f"- {t['tag']}: {t['count']}")
        
        lines.extend(["", "## Recent Activity"])
        for b in overview['recent_activity'][:10]:
            lines.append(f"- **{b['title']}** (by {b['user']})")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "find_common_ground":
        user_a = arguments.get("user_a", "")
        user_b = arguments.get("user_b", "")
        
        if not user_a or not user_b:
            return [TextContent(type="text", text="Error: both user_a and user_b are required")]
        
        # Fetch both profiles in parallel
        bookmarks_a, bookmarks_b = await asyncio.gather(
            fetch_user_bookmarks(user_a, 200),
            fetch_user_bookmarks(user_b, 200)
        )
        
        if not bookmarks_a:
            return [TextContent(type="text", text=f"No bookmarks found for user '{user_a}'")]
        if not bookmarks_b:
            return [TextContent(type="text", text=f"No bookmarks found for user '{user_b}'")]
        
        profile_a = build_user_profile(bookmarks_a, user_a)
        profile_b = build_user_profile(bookmarks_b, user_b)
        
        overlap = find_user_overlaps(profile_a, profile_b)
        
        lines = [
            f"# Common Ground: {overlap['user_a']} & {overlap['user_b']}",
            f"**Compatibility Score:** {overlap['compatibility_score']}%",
            f"**Shared Interests:** {overlap['shared_interest_count']}",
            "",
            "## Shared Tags",
            ", ".join(overlap['shared_interests']) if overlap['shared_interests'] else "None found",
            "",
            f"## Unique to {overlap['user_a']}",
            ", ".join(overlap['unique_to_a']) if overlap['unique_to_a'] else "None",
            "",
            f"## Unique to {overlap['user_b']}",
            ", ".join(overlap['unique_to_b']) if overlap['unique_to_b'] else "None",
        ]
        
        if overlap['shared_bookmarks']:
            lines.extend(["", "## Bookmarks Saved by Both"])
            for b in overlap['shared_bookmarks'][:5]:
                lines.append(f"- {b['title']}")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    # =================================================================
    # COLLECTIVE (bloom-houston) TOOLS
    # =================================================================
    
    elif name == "search_collective":
        query = arguments.get("query", "").lower()
        tag_filter = arguments.get("tag", "").lower()
        user_filter = arguments.get("user", "").lower()
        limit = arguments.get("limit", 20)
        
        bookmarks = await fetch_public_tag_feed(COLLECTIVE_TAG, 200)
        
        results = []
        for b in bookmarks:
            # Text match
            title = get_bookmark_title(b).lower()
            desc = get_bookmark_description(b).lower()
            url = get_bookmark_url(b).lower()
            text_match = not query or query in title or query in desc or query in url
            
            # Tag match
            b_tags = [t.lower() for t in get_bookmark_tags(b)]
            tag_match = not tag_filter or tag_filter in b_tags
            
            # User match
            b_user = get_bookmark_user(b).lower()
            user_match = not user_filter or user_filter == b_user
            
            if text_match and tag_match and user_match:
                results.append({
                    "title": get_bookmark_title(b),
                    "url": get_bookmark_url(b),
                    "user": get_bookmark_user(b),
                    "tags": [t for t in get_bookmark_tags(b) if t.lower() != COLLECTIVE_TAG],
                    "notes": get_bookmark_description(b),
                    "date": get_bookmark_date(b)
                })
            
            if len(results) >= limit:
                break
        
        if not results:
            return [TextContent(type="text", text="No matching bookmarks found in bloom-houston")]
        
        lines = [f"Found {len(results)} matching bookmark(s):\n"]
        for r in results:
            lines.append(f"**{r['title']}**")
            lines.append(f"  URL: {r['url']}")
            lines.append(f"  Saved by: {r['user']}")
            lines.append(f"  Tags: {', '.join(r['tags']) if r['tags'] else 'none'}")
            if r['notes']:
                lines.append(f"  Notes: {r['notes'][:150]}")
            lines.append("")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "get_recent":
        tag_filter = arguments.get("tag", "").lower()
        limit = arguments.get("limit", 10)
        
        bookmarks = await fetch_public_tag_feed(COLLECTIVE_TAG, 100)
        
        results = []
        for b in bookmarks:
            b_tags = [t.lower() for t in get_bookmark_tags(b)]
            if not tag_filter or tag_filter in b_tags:
                results.append({
                    "title": get_bookmark_title(b),
                    "url": get_bookmark_url(b),
                    "user": get_bookmark_user(b),
                    "tags": [t for t in get_bookmark_tags(b) if t.lower() != COLLECTIVE_TAG],
                    "notes": get_bookmark_description(b),
                    "date": get_bookmark_date(b)
                })
            
            if len(results) >= limit:
                break
        
        if not results:
            return [TextContent(type="text", text="No recent bookmarks found")]
        
        lines = [f"Recent bloom-houston bookmarks:\n"]
        for r in results:
            lines.append(f"**{r['title']}**")
            lines.append(f"  URL: {r['url']}")
            lines.append(f"  Saved by: {r['user']} on {r['date']}")
            lines.append(f"  Tags: {', '.join(r['tags']) if r['tags'] else 'none'}")
            if r['notes']:
                lines.append(f"  Notes: {r['notes'][:100]}")
            lines.append("")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "who_knows_about":
        topic = arguments.get("topic", "").lower()
        if not topic:
            return [TextContent(type="text", text="Error: topic is required")]
        
        bookmarks = await fetch_public_tag_feed(COLLECTIVE_TAG, 200)
        
        user_matches = defaultdict(list)
        for b in bookmarks:
            title = get_bookmark_title(b).lower()
            desc = get_bookmark_description(b).lower()
            tags = [t.lower() for t in get_bookmark_tags(b)]
            
            if topic in title or topic in desc or topic in tags:
                user = get_bookmark_user(b)
                user_matches[user].append(b)
        
        if not user_matches:
            return [TextContent(type="text", text=f"No one in the collective has bookmarks about '{topic}'")]
        
        sorted_users = sorted(user_matches.items(), key=lambda x: -len(x[1]))
        
        lines = [f"Contributors who know about '{topic}':\n"]
        for username, matches in sorted_users:
            lines.append(f"**{username}** - {len(matches)} bookmark(s)")
            for b in matches[:3]:
                lines.append(f"  • {get_bookmark_title(b)}")
            if len(matches) > 3:
                lines.append(f"  ... and {len(matches) - 3} more")
            lines.append("")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "find_connections":
        target_user = arguments.get("user", "").lower()
        min_overlap = arguments.get("min_overlap", 2)
        
        bookmarks = await fetch_public_tag_feed(COLLECTIVE_TAG, 200)
        
        # Build user -> tags mapping
        user_tags = defaultdict(set)
        for b in bookmarks:
            username = get_bookmark_user(b)
            tags = set(t.lower() for t in get_bookmark_tags(b) if t.lower() != COLLECTIVE_TAG)
            user_tags[username].update(tags)
        
        if target_user:
            # Find connections for specific user
            matching = None
            for u in user_tags:
                if u.lower() == target_user:
                    matching = u
                    break
            
            if not matching:
                return [TextContent(type="text", text=f"User '{target_user}' not found in the collective")]
            
            target_tags = user_tags[matching]
            connections = []
            
            for other, other_tags in user_tags.items():
                if other.lower() == target_user:
                    continue
                shared = target_tags & other_tags
                if len(shared) >= min_overlap:
                    connections.append((other, shared))
            
            if not connections:
                return [TextContent(type="text", text=f"No connections found for '{matching}' with at least {min_overlap} shared tags")]
            
            connections.sort(key=lambda x: -len(x[1]))
            
            lines = [f"Connections for {matching}:\n"]
            for other, shared in connections:
                lines.append(f"**{other}** - {len(shared)} shared interest(s)")
                lines.append(f"  Shared: {', '.join(sorted(shared))}")
                lines.append("")
            
            return [TextContent(type="text", text="\n".join(lines))]
        
        else:
            # Show all network connections
            all_connections = []
            users = list(user_tags.keys())
            
            for i, u1 in enumerate(users):
                for u2 in users[i+1:]:
                    shared = user_tags[u1] & user_tags[u2]
                    if len(shared) >= min_overlap:
                        all_connections.append((u1, u2, shared))
            
            if not all_connections:
                return [TextContent(type="text", text=f"No connections found with at least {min_overlap} shared tags")]
            
            all_connections.sort(key=lambda x: -len(x[2]))
            
            lines = [f"Network connections (min {min_overlap} shared tags):\n"]
            for u1, u2, shared in all_connections[:20]:
                lines.append(f"**{u1}** ↔ **{u2}** ({len(shared)} shared)")
                lines.append(f"  Tags: {', '.join(sorted(shared))}")
                lines.append("")
            
            return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "who_else_saved":
        url = arguments.get("url", "")
        if not url:
            return [TextContent(type="text", text="Error: url is required")]
        
        bookmarks = await fetch_public_tag_feed(COLLECTIVE_TAG, 200)
        normalized = normalize_url(url)
        
        savers = []
        for b in bookmarks:
            if normalize_url(get_bookmark_url(b)) == normalized:
                savers.append({
                    "user": get_bookmark_user(b),
                    "title": get_bookmark_title(b),
                    "notes": get_bookmark_description(b),
                    "date": get_bookmark_date(b)
                })
        
        if not savers:
            return [TextContent(type="text", text=f"No one in the collective has saved this URL")]
        
        lines = [f"Found {len(savers)} member(s) who saved this URL:\n"]
        for s in savers:
            lines.append(f"**{s['user']}**")
            if s['notes']:
                lines.append(f"  Notes: {s['notes']}")
            lines.append(f"  Saved: {s['date']}")
            lines.append("")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "get_network_map":
        min_shared = arguments.get("min_shared_tags", 1)
        fmt = arguments.get("format", "json")
        
        bookmarks = await fetch_public_tag_feed(COLLECTIVE_TAG, 200)
        
        # Build user -> tags mapping
        user_tags = defaultdict(set)
        for b in bookmarks:
            username = get_bookmark_user(b)
            tags = set(t.lower() for t in get_bookmark_tags(b) if t.lower() != COLLECTIVE_TAG)
            user_tags[username].update(tags)
        
        # Find connections
        connections = []
        users = list(user_tags.keys())
        for i, u1 in enumerate(users):
            for u2 in users[i+1:]:
                shared = user_tags[u1] & user_tags[u2]
                if len(shared) >= min_shared:
                    connections.append({"a": u1, "b": u2, "shared": list(shared), "count": len(shared)})
        
        if fmt == "json":
            import json
            return [TextContent(type="text", text=json.dumps({
                "users": list(user_tags.keys()),
                "connections": connections
            }, indent=2))]
        
        # Mermaid format
        lines = ["```mermaid", "graph LR"]
        
        # Add nodes
        for user in users:
            safe_id = user.replace("-", "_").replace(".", "_")
            lines.append(f"    {safe_id}(({user}))")
        
        # Add connections
        for conn in connections[:30]:  # Limit for readability
            a = conn["a"].replace("-", "_").replace(".", "_")
            b = conn["b"].replace("-", "_").replace(".", "_")
            lines.append(f"    {a} ---|{conn['count']} shared| {b}")
        
        lines.append("```")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    # =================================================================
    # PERSONAL BOOKMARKS TOOLS
    # =================================================================
    
    elif name == "search_my_bookmarks":
        query = arguments.get("query", "").lower()
        tag_filter = arguments.get("tag")
        limit = arguments.get("limit", 20)
        
        bookmarks = await fetch_my_bookmarks(tag=tag_filter, count=200)
        
        if not bookmarks:
            if not PINBOARD_API_TOKEN:
                return [TextContent(type="text", text="Error: No Pinboard API token configured")]
            return [TextContent(type="text", text="No bookmarks found in your Pinboard")]
        
        results = []
        for b in bookmarks:
            title = get_bookmark_title(b).lower()
            desc = get_bookmark_description(b).lower()
            url = get_bookmark_url(b).lower()
            
            if not query or query in title or query in desc or query in url:
                results.append({
                    "title": get_bookmark_title(b),
                    "url": get_bookmark_url(b),
                    "tags": get_bookmark_tags(b),
                    "notes": get_bookmark_description(b),
                    "date": get_bookmark_date(b)
                })
            
            if len(results) >= limit:
                break
        
        if not results:
            return [TextContent(type="text", text="No matching bookmarks found")]
        
        lines = [f"Found {len(results)} bookmark(s) in your Pinboard:\n"]
        for r in results:
            lines.append(f"**{r['title']}**")
            lines.append(f"  URL: {r['url']}")
            lines.append(f"  Tags: {', '.join(r['tags']) if r['tags'] else 'none'}")
            if r['notes']:
                lines.append(f"  Notes: {r['notes'][:150]}")
            lines.append("")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "get_my_tags":
        tags = await fetch_my_tags()
        
        if not tags:
            if not PINBOARD_API_TOKEN:
                return [TextContent(type="text", text="Error: No Pinboard API token configured")]
            return [TextContent(type="text", text="No tags found in your Pinboard")]
        
        # Sort by count (descending)
        sorted_tags = sorted(tags.items(), key=lambda x: -int(x[1]))
        
        lines = ["Your Pinboard tags:\n"]
        for tag, count in sorted_tags[:50]:
            lines.append(f"- **{tag}**: {count}")
        
        if len(sorted_tags) > 50:
            lines.append(f"\n... and {len(sorted_tags) - 50} more tags")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    # =================================================================
    # PUBLIC PINBOARD TOOLS
    # =================================================================
    
    elif name == "search_user_bookmarks":
        username = arguments.get("username", "")
        query = arguments.get("query", "").lower()
        tag_filter = arguments.get("tag")
        limit = arguments.get("limit", 20)
        
        if not username:
            return [TextContent(type="text", text="Error: username is required")]
        
        bookmarks = await fetch_user_bookmarks(username, 200, tag=tag_filter)
        
        if not bookmarks:
            return [TextContent(type="text", text=f"No public bookmarks found for user '{username}'")]
        
        results = []
        for b in bookmarks:
            title = get_bookmark_title(b).lower()
            desc = get_bookmark_description(b).lower()
            url = get_bookmark_url(b).lower()
            
            if not query or query in title or query in desc or query in url:
                results.append({
                    "title": get_bookmark_title(b),
                    "url": get_bookmark_url(b),
                    "tags": get_bookmark_tags(b),
                    "notes": get_bookmark_description(b),
                    "date": get_bookmark_date(b)
                })
            
            if len(results) >= limit:
                break
        
        if not results:
            return [TextContent(type="text", text=f"No matching bookmarks found for '{username}'")]
        
        lines = [f"Found {len(results)} bookmark(s) from {username}:\n"]
        for r in results:
            lines.append(f"**{r['title']}**")
            lines.append(f"  URL: {r['url']}")
            lines.append(f"  Tags: {', '.join(r['tags']) if r['tags'] else 'none'}")
            if r['notes']:
                lines.append(f"  Notes: {r['notes'][:150]}")
            lines.append("")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "search_public_pinboard":
        query = arguments.get("query", "").lower()
        tag_filter = arguments.get("tag")
        limit = arguments.get("limit", 20)
        
        if tag_filter:
            bookmarks = await fetch_public_tag_feed(tag_filter, 100)
        else:
            bookmarks = await fetch_public_recent(100)
        
        if not bookmarks:
            return [TextContent(type="text", text="No public bookmarks found")]
        
        results = []
        for b in bookmarks:
            title = get_bookmark_title(b).lower()
            desc = get_bookmark_description(b).lower()
            url = get_bookmark_url(b).lower()
            
            if not query or query in title or query in desc or query in url:
                results.append({
                    "title": get_bookmark_title(b),
                    "url": get_bookmark_url(b),
                    "user": get_bookmark_user(b),
                    "tags": get_bookmark_tags(b),
                    "notes": get_bookmark_description(b),
                    "date": get_bookmark_date(b)
                })
            
            if len(results) >= limit:
                break
        
        if not results:
            return [TextContent(type="text", text="No matching public bookmarks found")]
        
        lines = [f"Found {len(results)} public bookmark(s):\n"]
        for r in results:
            lines.append(f"**{r['title']}**")
            lines.append(f"  URL: {r['url']}")
            lines.append(f"  Saved by: {r['user']}")
            lines.append(f"  Tags: {', '.join(r['tags']) if r['tags'] else 'none'}")
            lines.append("")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    # =================================================================
    # BOOKMARK MANAGEMENT TOOLS
    # =================================================================
    
    elif name == "save_bookmark":
        url = arguments.get("url", "")
        title = arguments.get("title", "")
        description = arguments.get("description", "")
        tags = arguments.get("tags", [])
        private = arguments.get("private", False)
        toread = arguments.get("toread", False)
        
        if not url or not title:
            return [TextContent(type="text", text="Error: url and title are required")]
        
        result = await save_bookmark_to_pinboard(
            url=url,
            title=title,
            description=description,
            tags=tags,
            private=private,
            toread=toread
        )
        
        if "error" in result:
            return [TextContent(type="text", text=f"Error saving bookmark: {result['error']}")]
        
        # Clear cache so new bookmark shows up
        _cache.clear()
        _cache_timestamps.clear()
        
        visibility = "private" if private else "public"
        tags_str = ", ".join(tags) if tags else "none"
        
        return [TextContent(type="text", text=(
            f"✓ Bookmark saved!\n\n"
            f"**Title:** {title}\n"
            f"**URL:** {url}\n"
            f"**Tags:** {tags_str}\n"
            f"**Visibility:** {visibility}\n"
            f"**To Read:** {'yes' if toread else 'no'}"
        ))]
    
    
    elif name == "suggest_tags":
        url = arguments.get("url", "")
        if not url:
            return [TextContent(type="text", text="Error: url is required")]
        
        suggestions = await get_tag_suggestions(url)
        
        lines = [f"Tag suggestions for: {url}\n"]
        
        if suggestions["popular"]:
            lines.append("**Popular tags** (used by others for this URL):")
            lines.append(", ".join(suggestions["popular"]))
            lines.append("")
        else:
            lines.append("**Popular tags:** none found\n")
        
        if suggestions["recommended"]:
            lines.append("**Recommended tags** (from your vocabulary):")
            lines.append(", ".join(suggestions["recommended"]))
        else:
            lines.append("**Recommended tags:** none found")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    # =================================================================
    # UTILITY TOOLS
    # =================================================================
    
    elif name == "get_feed_url":
        username = arguments.get("username")
        tag = arguments.get("tag")
        fmt = arguments.get("format", "rss")
        
        base = f"https://feeds.pinboard.in/{fmt}/"
        
        if username and tag:
            url = f"{base}u:{username}/t:{tag}/"
        elif username:
            url = f"{base}u:{username}/"
        elif tag:
            url = f"{base}t:{tag}/"
        else:
            return [TextContent(type="text", text="Error: provide username and/or tag")]
        
        return [TextContent(type="text", text=f"Feed URL ({fmt.upper()}):\n{url}")]
    
    
    elif name == "clear_cache":
        _cache.clear()
        _cache_timestamps.clear()
        return [TextContent(type="text", text="✓ Cache cleared. Next API calls will fetch fresh data.")]
    
    
    # =================================================================
    # MEMBER DISCOVERY TOOLS
    # =================================================================
    
    elif name == "discover_new_members":
        tag = arguments.get("tag", COLLECTIVE_TAG)
        auto_add = arguments.get("auto_add", True)
        
        # Fetch current bookmarks for the tag
        bookmarks = await fetch_public_tag_feed(tag, 200)
        
        if not bookmarks:
            return [TextContent(type="text", text=f"No bookmarks found with tag '{tag}'")]
        
        # Load known members
        known_data = load_known_members()
        known_usernames = {k.lower() for k in known_data["members"]}
        
        # Find current contributors
        current_contributors = {}
        for b in bookmarks:
            username = get_bookmark_user(b)
            if username.lower() not in current_contributors:
                current_contributors[username.lower()] = {
                    "username": username,
                    "title": get_bookmark_title(b),
                    "url": get_bookmark_url(b),
                    "date": get_bookmark_date(b),
                    "tags": get_bookmark_tags(b)
                }
        
        # Find new members
        new_members = []
        for username_lower, data in current_contributors.items():
            if username_lower not in known_usernames:
                new_members.append(data)
                
                if auto_add:
                    add_known_member(
                        data["username"],
                        datetime.utcnow().isoformat() + "Z",
                        {"title": data["title"], "url": data["url"]}
                    )
        
        # Update last check time
        known_data["last_check"] = datetime.utcnow().isoformat() + "Z"
        save_known_members(known_data)
        
        # Format response
        if not new_members:
            return [TextContent(type="text", text=(
                f"No new members found for tag '{tag}'.\n\n"
                f"**Known members:** {len(known_usernames)}\n"
                f"**Current contributors:** {len(current_contributors)}\n"
                f"**Last check:** {known_data.get('last_check', 'never')}"
            ))]
        
        lines = [
            f"🎉 **{len(new_members)} new member(s) discovered!**\n",
            f"Tag: `{tag}`\n"
        ]
        
        for m in new_members:
            lines.append(f"### {m['username']}")
            lines.append(f"**First bookmark:** {m['title']}")
            lines.append(f"**URL:** {m['url']}")
            lines.append(f"**Tags:** {', '.join(m['tags']) if m['tags'] else 'none'}")
            lines.append(f"**Date:** {m['date']}")
            lines.append("")
        
        if auto_add:
            lines.append("✓ Added to known members list")
        else:
            lines.append("ℹ️ Not added to known members (auto_add=false)")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "list_known_members":
        sort_by = arguments.get("sort_by", "first_seen")
        
        known_data = load_known_members()
        members = known_data.get("members", {})
        
        if not members:
            return [TextContent(type="text", text=(
                "No known members yet.\n\n"
                "Use `discover_new_members` to start tracking contributors to bloom tags."
            ))]
        
        # Sort members
        member_list = [(username, data) for username, data in members.items()]
        
        if sort_by == "first_seen":
            member_list.sort(key=lambda x: x[1].get("first_seen", ""), reverse=True)
        else:
            member_list.sort(key=lambda x: x[0].lower())
        
        lines = [
            f"# Known Bloom Network Members",
            f"**Total:** {len(members)}",
            f"**Last check:** {known_data.get('last_check', 'never')}",
            ""
        ]
        
        for username, data in member_list:
            lines.append(f"### {username}")
            lines.append(f"- **First seen:** {data.get('first_seen', 'unknown')}")
            lines.append(f"- **First bookmark:** {data.get('first_bookmark_title', 'unknown')}")
            if data.get('first_bookmark_url'):
                lines.append(f"- **URL:** {data.get('first_bookmark_url')}")
            lines.append("")
        
        return [TextContent(type="text", text="\n".join(lines))]
    
    
    elif name == "reset_known_members":
        confirm = arguments.get("confirm", False)
        
        if not confirm:
            return [TextContent(type="text", text=(
                "⚠️ This will clear all known members and cannot be undone.\n\n"
                "To confirm, call with `confirm: true`"
            ))]
        
        # Reset the file
        save_known_members({"members": {}, "last_check": None})
        
        return [TextContent(type="text", text="✓ Known members list has been reset. All members cleared.")]
    
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# =============================================================================
# MAIN
# =============================================================================

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
