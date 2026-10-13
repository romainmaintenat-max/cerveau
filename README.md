# Cerveau

**A local, sovereign long-term memory for AI agents.**
Index your notes, let your agents remember what matters, and search everything by meaning —
on your own machine, with no cloud service and no API bill.

Cerveau ("brain" in French) plugs into any MCP client (Claude Desktop, Claude Code, and others) and gives it:

- `search_memory` — semantic search across your documents and the agent's memories,
- `remember` — store a durable fact, decision or lesson,
- `brain_status` — what is in there.

*[Version française : README.fr.md](README.fr.md)*

## Why another memory tool?

Cerveau's design comes from a personal memory system that has run for months on a home AI server.
This package is a clean, standalone rewrite of its core ideas — each one a lesson from something that went wrong:

- **The journal is the source of truth.** Every memory is written to an append-only, fsync'ed journal
  *before* it reaches the vector database. Lose Qdrant, change the embedding model: `cerveau replay`
  rebuilds every memory (documents are rebuilt by `cerveau index`, from your files). The journal stores
  text, never vectors. Each line carries a sequence number and a checksum: a loss in the middle of the
  journal is detected, while the damage a crash leaves at the end (a half-written line, NUL bytes) is
  recognised, kept aside and never glued to the next record.
- **Sealed collections.** The embedding model and vector size are recorded at creation. Starting with
  another model, or on collections Cerveau did not create, is refused before anything is written.
- **No silent truncation.** A text too long for the model is rejected with a clear error, never cut.
- **Agents add, humans remove.** Through MCP an agent can search and remember. Archiving, restoring,
  indexing and rebuilding are command-line only. Archiving is reversible; only `cerveau purge` erases, on your command.
- **Secure by default.** Qdrant requires an API key and only listens on 127.0.0.1. See [SECURITY.md](SECURITY.md).

## Requirements

| What | Version | Notes |
|---|---|---|
| Python | 3.11+ | macOS ships an older `python3`, and so do Ubuntu 22.04 (3.10) and Debian 11 (3.9): install a recent one. Debian/Ubuntu: `sudo apt install python3-venv` |
| Docker + Docker Compose v2 | recent | runs Qdrant (`docker compose`, not `docker-compose`) |
| [Ollama](https://ollama.com) | recent | runs the embedding model; it must be running |
| Embedding model | `bge-m3` (default): multilingual, 100+ languages, MIT license<br>`snowflake-arctic-embed2` (alternative): same size and same dimension; switching models requires re-indexing | runs on CPU; a GPU only makes indexing faster |

To use the alternative: `ollama pull snowflake-arctic-embed2` and set `CERVEAU_EMBED_MODEL=snowflake-arctic-embed2` in `.env` (`CERVEAU_VECTOR_SIZE` stays `1024`).

## Quick start

```bash
git clone https://github.com/romainmaintenat-max/cerveau.git
cd cerveau
python3 -m venv .venv && source .venv/bin/activate    # Windows: see below
pip install -e .

cp .env.example .env && chmod 600 .env                # Windows: copy .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste it as CERVEAU_QDRANT_API_KEY in .env

docker compose up -d                                  # give Qdrant a few seconds to start
ollama pull bge-m3

cerveau init
cerveau index examples/notes
cerveau search "how hot should the oven be for bread?"
```

On Windows (PowerShell): `py -m venv .venv`, then `.venv\Scripts\Activate.ps1` (if scripts are blocked:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`), and `copy .env.example .env`. Instead of `chmod 600`:
`icacls .env /inheritance:r /grant:r "${env:USERNAME}:F"` (removes inherited rights and gives full access to you alone).

The first result should come from `bread.md` and mention 240 °C. Then index your own folders.
To stop Qdrant: `docker compose down` (never `down -v`: it deletes the Qdrant volume; the journal
survives, but you would have to `init`, `replay` and re-index).
Run `cerveau` from the folder that holds `.env`, or set `CERVEAU_ENV_FILE` (see Configuration).
In a new terminal, activate the virtual environment again (`source .venv/bin/activate`, or
`.venv\Scripts\Activate.ps1` on Windows), or call `.venv/bin/cerveau` by its full path.
`.env` holds the Qdrant key: keep it readable by you alone (`chmod 600`, or `icacls` on Windows). If you choose a key yourself, avoid `$`: Docker Compose
would read it differently from Cerveau. Keep `.env` in UTF-8: Docker Compose cannot read UTF-16.

## Command line

| Command | What it does |
|---|---|
| `cerveau init` | create the collections, check the configuration |
| `cerveau index <folder or file> [--prune]` | index `.md`, `.markdown`, `.txt` files; unchanged files are skipped; files deleted since the last run stay searchable until `--prune` (a warning says so); a whole disk, your home folder, the folder of all homes and system folders (`/etc`, `/usr`, `C:\Windows`…) are refused; exit code 1 if a file could not be indexed (listed in `errors`); `--prune` (folders only) removes what is no longer indexable there: deleted files, and files now skipped (hidden, excluded); skipped if a sub-folder could not be read, or if the folder holds no text file at all (an unmounted disk looks empty: use `unindex` if the files are really gone) |
| `cerveau unindex <path>` | remove what was indexed at or under a path, even if it no longer exists (renamed or moved folder) |
| `cerveau remember "<text>" [--kind decision]` | store a memory |
| `cerveau search "<query>" [-k 5 (1-50; 1-20 through MCP)] [--scope all\|docs\|memories] [--include-archived] [--json]` | search by meaning |
| `cerveau list [--actor mcp] [--include-archived] [--json]` | every memory of the journal, in full, with who wrote it (`mcp` = an AI client, `cli` = you, `adopted` = recovered by `adopt-orphans`, shown with the author recorded in Qdrant) |
| `cerveau list --orphans` | memories found in Qdrant but not in this journal |
| `cerveau archive <id>` / `cerveau restore <id>` | hide / show a memory (reversible; the text stays in the journal) |
| `cerveau purge <id> --confirm` | removes a memory from the active brain (journal and Qdrant), for a secret saved by mistake: rewrites the journal without it and removes it from Qdrant, in every Cerveau prefix of that Qdrant. Backups made before, and Qdrant's own files until it rewrites them, still hold the text: see [SECURITY.md](SECURITY.md) to erase those too |
| `cerveau replay` | write every memory of the journal back to Qdrant |
| `cerveau status` | counts and configuration; warns when Qdrant is behind the journal |
| `cerveau adopt-orphans` | copy into the journal the memories found only in Qdrant (see `list --orphans`); their author becomes `adopted` |
| `cerveau reset-collections --confirm <prefix> [--discard-orphans]` | delete this prefix's Qdrant collections (journal and files kept), to start again after a wrong setup. Refused while Qdrant holds memories missing from the journal: `adopt-orphans` them first, or `--discard-orphans` to lose them |
| `cerveau --version` | |

A text starting with `-` goes after `--`: `cerveau remember -- "-5 °C tonight"`.

Files must be UTF-8 (or UTF-16 with BOM, Windows Notepad's "Unicode"); other encodings, and binary files, are reported, not guessed.
Under the folder you give, hidden files and folders (`.git`, `.draft.md`…), `node_modules`, `venv`,
`site-packages`, `__pycache__` and symbolic links are skipped (a folder you name yourself is
used even if it is hidden; a hidden *file* named directly is refused). Files over 20 MB are skipped and reported. A folder that cannot be read is reported and the others are indexed.
A document is identified by its absolute path: indexing a folder and then one of its sub-folders does
not create duplicates. `index --prune` and `unindex` remove chunks of *documents*, never memories.

## Connect an AI client

The MCP server is started by the client (stdio), from a folder you do not control. It therefore
**never** reads a `.env` from its current folder: give it `CERVEAU_ENV_FILE`, the absolute path of your `.env`.
Run `cerveau init` once before connecting a client: through MCP, nothing is ever created (not even by `remember`). Configuration errors are printed on the server's
stderr (most clients show it in their MCP logs).

**Claude Desktop** — in `claude_desktop_config.json` (menu Settings → Developer → Edit Config; on macOS
`~/Library/Application Support/Claude/`, on Windows `%APPDATA%\Claude\`):

```json
{
  "mcpServers": {
    "cerveau": {
      "command": "/absolute/path/to/cerveau/.venv/bin/cerveau-mcp",
      "env": { "CERVEAU_ENV_FILE": "/absolute/path/to/cerveau/.env" }
    }
  }
}
```

On Windows, use `C:\\path\\to\\cerveau\\.venv\\Scripts\\cerveau-mcp.exe` and `C:\\path\\to\\cerveau\\.env`:
in JSON every backslash is written twice.

**Claude Code**:

```bash
claude mcp add -s user cerveau -e CERVEAU_ENV_FILE=/absolute/path/to/cerveau/.env -- /absolute/path/to/cerveau/.venv/bin/cerveau-mcp
```

On Windows: `claude mcp add -s user cerveau -e CERVEAU_ENV_FILE=C:\path\to\cerveau\.env -- C:\path\to\cerveau\.venv\Scripts\cerveau-mcp.exe`

## Configuration

Settings live in `.env` (see [.env.example](.env.example)). Only `CERVEAU_*` lines are read, and a
`CERVEAU_*` name close to a real setting is refused: a typo never silently falls back to a default.
The file used is `CERVEAU_ENV_FILE` if set (it must exist); otherwise the command line — not the MCP
server — reads `.env` in the current folder. Non-empty environment variables win over the file. Every
name starting with `CERVEAU` is checked: one within 3 edits of a setting (typo, wrong separator), one made of
some of a setting's words or more (`CERVEAU_ALLOW_WRITE`, `CERVEAU_DATA`), or one about writing or read-only
mode (`CERVEAU_READONLY`) is an error; any other (`CERVEAU_TOKEN`) belongs to another program and is ignored
with a warning.

| Variable | Default | |
|---|---|---|
| `CERVEAU_ENV_FILE` | — | absolute path of the `.env` to load (environment variable only) |
| `CERVEAU_QDRANT_URL` | `http://127.0.0.1:6333` | |
| `CERVEAU_QDRANT_API_KEY` | — | required for any non-local Qdrant |
| `CERVEAU_OLLAMA_URL` | `http://127.0.0.1:11434` | a non-local URL prints a warning: every text goes there |
| `CERVEAU_OLLAMA_TIMEOUT` | `300` | seconds per embedding request; raise it on a slow CPU |
| `CERVEAU_EMBEDDER` | `ollama` | `hash` is a fake, non-semantic embedder for tests only |
| `CERVEAU_EMBED_MODEL` / `CERVEAU_VECTOR_SIZE` | `bge-m3` / `1024` | must match each other |
| `CERVEAU_MAX_TEXT_CHARS` / `CERVEAU_CHUNK_MAX_CHARS` | `3000` / `1500` | chunk size must leave 256 characters for the heading. Dense text (code, tables) may need lower values |
| `CERVEAU_PREFIX` | `cerveau` | collections are `<prefix>_docs`, `_memories`, `_meta`. It does **not** separate journals: see below |
| `CERVEAU_DATA_DIR` | `~/.cerveau` | the journal lives here — **back it up**. variables are expanded the platform's way (`$VAR` on Linux/macOS, `%VAR%` on Windows); a relative path in `.env` is relative to the `.env` file |
| `CERVEAU_MCP_ALLOW_WRITE` | `true` | `false`: no `remember` tool, and the MCP server writes nothing at all (but its log file) |
| `CERVEAU_LOG` | `on` | `off`: no log file (see [Reporting a bug](#reporting-a-bug)) |
| `CERVEAU_QDRANT_PORT` | `6333` | port of Qdrant on this machine, read by `docker-compose.yml` only (change `CERVEAU_QDRANT_URL` with it) |

The MCP server reads its settings once, when the AI client starts it: **after changing the `.env`,
restart the AI client** (Claude Desktop, Claude Code…). Until then the old settings stay in force,
`CERVEAU_MCP_ALLOW_WRITE=false` included.


**One journal per data folder, not per prefix.** The memories belong to `CERVEAU_DATA_DIR`; a prefix
is only a set of Qdrant collections built from them. Every prefix that uses the same data folder sees
the same memories after `cerveau replay` — that is how a model change works. For a separate brain
(tests, another person), use another `CERVEAU_DATA_DIR`.

**Changing the embedding model**: use a new `CERVEAU_PREFIX`, run `cerveau index` on your folders and
`cerveau replay` for the memories. The old collections stay untouched until you remove them
(`cerveau reset-collections --confirm <old prefix>` with the old settings).

**A wrong first setup** (model not pulled, wrong `CERVEAU_VECTOR_SIZE`): `cerveau init` tries the model
before creating anything, so nothing is sealed. If a prefix was sealed with wrong values anyway,
`cerveau reset-collections --confirm <prefix>` removes it.

## When something goes wrong

- **`remember` says the memory "IS saved in the journal, but Qdrant failed"**: do not retry (that would
  create a duplicate). Once Qdrant is back, run `cerveau replay`.
- **`error: … 401`**: the Qdrant key is wrong or the expected `.env` was not loaded.
- **`status` shows `journal_torn_lines`**: a crash interrupted a write at some point. The unfinished
  bytes are kept in the file and skipped. A crash cannot lose a memory that `remember` had confirmed.
  (A disk error that damages the very end of the last line can look the same: keep backups.)
- **"unreadable record (not an interrupted write)"**: the journal was changed by something other than
  Cerveau (edited, re-encoded, disk error). Cerveau stops instead of guessing: restore the journal from
  your backup.
- **The journal is unreadable and you have no backup**: if Qdrant still holds the memories, rebuild
  the journal from it: rename the damaged file (keep it until you are sure; it still holds every text,
  purged ones included), then run `cerveau adopt-orphans` — every
  memory found in Qdrant is copied into a new journal, with its original date.
- **What is *not* detected**: a journal cut short at the end (an older backup restored, a copy that
  stopped in the middle of its last line) looks like a journal that simply has fewer records. `status`
  then warns that Qdrant holds memories that are not in the journal (`cerveau list --orphans`).
- **"the brain is busy"**: another command holds a lock (60 s wait). Documents and memories have
  separate locks: a long `cerveau index` blocks another `index`, never the agent's `remember`; a
  `replay` blocks `remember`. Searches are never blocked.
- **"did not answer within 300 s"**: Ollama is too slow for one request (CPU only). Raise
  `CERVEAU_OLLAMA_TIMEOUT`.
- **Indexing a file fails with "input length exceeds the context length"**: very dense text (code,
  tables, hexadecimal, CJK) uses more tokens per character. Lower `CERVEAU_CHUNK_MAX_CHARS`. The same
  limit can refuse a very dense `remember`: nothing is written then, split the text.

`search` always returns the `k` closest results, even when nothing is really close: look at the scores.

## Reporting a bug

Cerveau keeps a log on your machine, in `<CERVEAU_DATA_DIR>/logs/`: `cli.log` for the command line,
`mcp.log` for the MCP server (5 MB each, then 3 older files). It never leaves your computer by itself:
nothing is sent anywhere. It holds versions, command and tool names, durations, counts, warnings and
errors with the place in the code where they happened — never a memory, a query, a document or the
Qdrant key; your data folder, settings file's folder and home folder are written `<data_dir>`,
`<env_dir>` and `~`. To report a bug, read the
file, then attach it to a [GitHub issue](https://github.com/romainmaintenat-max/cerveau/issues) with the
command you ran. To turn the log off: `CERVEAU_LOG=off` in `.env`.

## How it works

```
your notes ──► cerveau index ──► chunks ──► Ollama (embedding) ──► Qdrant  <prefix>_docs
agent ──► remember ──► Ollama (embedding) ──► journal.jsonl (fsync) ──► Qdrant  <prefix>_memories
agent ──► search_memory ──► Ollama ──► both collections, merged by cosine score
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # unit tests: in-memory Qdrant, fake embedder, no GPU needed
pytest -m integration     # against a real Qdrant: set CERVEAU_IT_QDRANT_URL and CERVEAU_IT_QDRANT_API_KEY
                          # and a real Ollama (before a release): set CERVEAU_IT_OLLAMA_URL
ruff check . && ruff format --check .
```

## Limits of this version

- Text files only (`.md`, `.markdown`, `.txt`). No PDF, no Office files. Pasted images (`data:` URIs)
  and runs of 1000 or more base64 or hexadecimal characters are replaced by `[binary data omitted]`
  before indexing (text in any language is kept, including Chinese or Japanese written without spaces).
- One writer at a time per data folder (a lock file enforces it). Keep `CERVEAU_DATA_DIR` on a local
  disk: network shares and synced folders (Dropbox, OneDrive…) do not make locks and appends reliable.
- No limit on how many memories an agent can add: review them with `cerveau list --actor mcp`.
- No duplicate detection, no automatic cleanup of memories: you curate with `archive` (or `purge`).
- The seal compares the model *name* (`bge-m3:latest` and `bge-m3` are the same
  name for Ollama and are treated as one). A model re-downloaded with different weights under the same name is not detected.

## License

MIT — see [LICENSE](LICENSE).
