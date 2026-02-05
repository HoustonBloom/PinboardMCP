# Bloom Pinboard MCP Server

A Model Context Protocol (MCP) server that connects Claude to Pinboard, enabling AI-assisted bookmark management and collective knowledge discovery for the Bloom network.

## The Bloom Tagging Convention

**`bloom`** is a signal tag that means: *"I want this resource surfaced to others in the network."*

Pair it with a location tag to help people find local resources:
- `bloom` + `houston`
- `bloom` + `austin`
- `bloom` + `detroit`
- `bloom` + `global` (for non-location-specific resources)

This creates a decentralized, federated knowledge layer. Anyone can participate by tagging their bookmarks with `bloom`. No central authority required — just a shared convention.

## What This Does

This MCP server lets Claude:
- Search and analyze your Pinboard bookmarks
- Discover what others tagging `bloom` are saving
- Find connections between community members based on shared interests
- Save new bookmarks with smart tagging
- Track new members joining the bloom network

## Prerequisites

- **Python 3.10+** installed on your computer
- **Claude Desktop** app (not the web version)
- **Pinboard account** with API access

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR-USERNAME/bloom-pinboard-mcp.git
cd bloom-pinboard-mcp
```

### 2. Install Python dependencies

```bash
pip install httpx mcp
```

### 3. Get your Pinboard API token

1. Log into Pinboard
2. Go to https://pinboard.in/settings/password
3. Copy your API token (looks like: `username:ABC123DEF456...`)

### 4. Find your Python path

**Windows:**
```bash
where python
```

**Mac/Linux:**
```bash
which python3
```

Copy the full path that appears.

### 5. Configure Claude Desktop

Claude Desktop needs a configuration file to know about MCP servers.

**Config file location:**
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`

Create or edit this file with the following (replace the placeholders):

```json
{
  "mcpServers": {
    "pinboard": {
      "command": "C:\\Path\\To\\Your\\python.exe",
      "args": [
        "C:\\Path\\To\\Your\\bloom-pinboard-mcp\\server.py"
      ],
      "env": {
        "PINBOARD_API_TOKEN": "your-username:YOUR_API_TOKEN_HERE"
      }
    }
  }
}
```

**Important:** 
- Use double backslashes (`\\`) on Windows
- Use the full absolute path to both Python and server.py
- Keep your API token private — never commit it to git!

### 6. Restart Claude Desktop

Completely quit Claude Desktop (don't just close the window — use Cmd+Q on Mac or right-click the system tray icon on Windows) and reopen it.

## Verify It's Working

In Claude Desktop, try asking:
- "Show me my Pinboard tags"
- "Search my bookmarks for houston"
- "Find bloom-tagged resources about mutual aid"

If Claude can answer these using your Pinboard data, you're all set!

## Available Tools

### Personal Bookmarks
| Tool | Description |
|------|-------------|
| `search_my_bookmarks` | Search your personal Pinboard collection |
| `get_my_tags` | List all your tags with counts |
| `save_bookmark` | Save a new bookmark with tags |
| `suggest_tags` | Get tag suggestions for a URL |

### Bloom Collective Discovery
| Tool | Description |
|------|-------------|
| `search_collective` | Search bloom-tagged bookmarks across the network |
| `get_recent` | Get recent bloom bookmarks |
| `get_collective_overview` | Full snapshot of the collective in one call |
| `who_knows_about` | Find contributors by topic |
| `find_connections` | Find interest overlaps between members |
| `get_network_map` | Generate relationship graph |

### User Discovery
| Tool | Description |
|------|-------------|
| `get_user_profile` | Complete profile of any Pinboard user |
| `search_user_bookmarks` | Search a specific user's public bookmarks |
| `find_common_ground` | Compare two users' shared interests |
| `who_else_saved` | Check who else saved a URL |

### Member Tracking
| Tool | Description |
|------|-------------|
| `discover_new_members` | Find new contributors using bloom tags |
| `list_known_members` | List all tracked bloom network members |
| `reset_known_members` | Clear the known members list |

### Utilities
| Tool | Description |
|------|-------------|
| `get_feed_url` | Generate RSS/JSON feed URLs for any user or tag |
| `clear_cache` | Clear the in-memory cache |
| `search_public_pinboard` | Search all public Pinboard bookmarks |

## Customizing for Your Location

The server is configured with a `COLLECTIVE_TAG` variable in `server.py`. You can change this to match your local bloom chapter or search for `bloom` broadly:

```python
COLLECTIVE_TAG = "bloom"  # Find all bloom-tagged resources globally
```

The tools will find anyone using the bloom convention, and you can filter by location in your queries.

## Troubleshooting

### "python.exe is not recognized"
Your Python path is wrong. Run `where python` (Windows) or `which python3` (Mac/Linux) and use that exact path.

### Server not showing up in Claude
1. Check your JSON syntax (missing commas, quotes, etc.)
2. Make sure paths are absolute, not relative
3. Completely quit and restart Claude Desktop

### "No bookmarks found"
- Check that your API token is correct
- Try `clear_cache` to force fresh API calls

### Check the logs
Claude Desktop logs MCP errors to:
- **Windows:** `%APPDATA%\Claude\logs\mcp-server-pinboard.log`
- **Mac:** `~/Library/Logs/Claude/mcp-server-pinboard.log`

## Security Notes

- **Never commit your API token to git** — use the example config as a template
- Your token is stored only in your local Claude Desktop config
- The server makes read/write calls to Pinboard's API on your behalf

## About Bloom Network

[Bloom Network](https://bloomnetwork.org/) is a grassroots movement building regenerative communities worldwide. 

The `bloom` tag on Pinboard serves as a shared knowledge layer — a way for organizers, activists, and community builders to surface resources to each other without any central platform. Tag something `bloom` and it becomes discoverable. Add your city and it becomes locally findable.

This is infrastructure for collective intelligence: simple, decentralized, and owned by no one.

## License

MIT

## Contributing

Pull requests welcome! Ideas for improvement:
- Dynamic location filtering (search `bloom` + any location)
- Cross-location discovery (find resources tagged in multiple cities)
- Digest/notification tools for new bloom content
- Integration with other knowledge-sharing tools

If you start a bloom chapter in your city, let us know!
