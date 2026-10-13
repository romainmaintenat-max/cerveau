# Security

## What Cerveau protects by default

- **Qdrant requires an API key.** `docker compose up` refuses to start without `CERVEAU_QDRANT_API_KEY`.
- **Qdrant listens on 127.0.0.1 only.** It is not reachable from your network. Only the REST port is published.
- **Cerveau refuses a remote Qdrant without an API key** (any host other than localhost).
- **The MCP server speaks stdio only.** It is started by your AI client and never opens a network port.
- **AI clients cannot hide or delete anything — through MCP.** Through MCP they can search and add memories; archiving,
  restoring, indexing and rebuilding are command-line only. Set `CERVEAU_MCP_ALLOW_WRITE=false` to make
  the MCP server fully read-only: it then writes nothing at all (no collection, no seal, no file but its log).
  The server reads its settings when the AI client starts it: restart the client after a change. A value
  set in the environment wins over the `.env`: a `CERVEAU_MCP_ALLOW_WRITE=true` left in the client's
  configuration overrides `false` in the file (the server logs `read-write` or `read-only` at start).
  `cerveau list --actor mcp` shows, in full, everything AI clients have written to this journal;
  `cerveau status` warns about memories found in Qdrant but not in the journal (`cerveau list --orphans`).
- **A typo in a setting name is refused**, never replaced by a default (`CERVEAU_MCP_ALLOW_WRIT=false` is an error).
  `CERVEAU-MCP-ALLOW-WRITE` or `CERVEAU MCP_ALLOW_WRITE` (wrong separator) is refused too: any `CERVEAU…`
  name within 3 edits of a setting, made of some of a setting's words or more (`CERVEAU_ALLOW_WRITE=false`),
  or about writing or read-only mode (`CERVEAU_READONLY`) is refused, while any other (`CERVEAU_TOKEN`, set by
  another program) is ignored with a warning. A name
  that does not start with `CERVEAU` is not seen at all: `CERVAU_PREFIX=x` is ignored.
- **Local services are never reached through a proxy.** Requests to a Qdrant or an Ollama on
  127.0.0.1 / localhost ignore `HTTP_PROXY` and the system proxy, so texts and the API key stay on the machine.
- **Only you can delete.** Archiving is a flag and can be undone; the text stays in the journal.
  `cerveau purge <id> --confirm` (command line only) removes a memory from the active brain (journal and
  Qdrant): the journal is rewritten without it and its point is deleted from Qdrant, in this prefix and in
  every other Cerveau prefix of the same Qdrant (after a model change, the old prefix holds a copy). What
  `purge` does **not** erase: copies made before (backups, Qdrant snapshots, a damaged journal you
  renamed), and **Qdrant's own files**: its write-ahead log and payload pages keep the text on disk (in
  the `cerveau_qdrant_storage` volume) until Qdrant rewrites them. To get rid of those too: `cerveau
  reset-collections --confirm <prefix>`, then `cerveau init`, `cerveau replay`, `cerveau index <your
  folders>`; for a secret that matters, also delete and recreate the volume (`docker compose down -v`,
  then the same steps). Copies kept by the file system can still hold the text too: freed disk blocks,
  Windows "Previous Versions", Time Machine, btrfs/ZFS/APFS snapshots, the history of a synced folder. A
  truly sensitive secret must be considered compromised and changed at its source.
  The journal keeps a "purge" record (the id, no text): if an older Qdrant backup brings the memory
  back, `cerveau status` says so and `cerveau adopt-orphans` deletes it again instead of adopting it.
- **What the MCP server hides.** Its tools return no absolute path (a document is named by its path
  inside the folder you indexed). In error messages, the data folder and your home folder (in any letter
  case) and the addresses of Qdrant and Ollama are replaced by `<data_dir>`, `~`, `<qdrant>`, `<ollama>`.
- **The MCP server never reads a `.env` from its current folder**: only `CERVEAU_ENV_FILE` or real
  environment variables. A project you open cannot redirect your memory elsewhere. Without
  `CERVEAU_ENV_FILE` it prints a warning at start: your `.env` is not loaded, so a
  `CERVEAU_MCP_ALLOW_WRITE=false` written there does not apply.
- **The journal is created readable by its owner only** (mode 0600; the data folder is created 0700) on
  Linux and macOS. An existing folder keeps its permissions: check them if you created it yourself.
- **A non-local Ollama URL prints a warning**: every text you index or remember is sent to it.
- **The log file stays local** (`<CERVEAU_DATA_DIR>/logs/`, owner-only like the journal): nothing sends
  it anywhere, it holds no memory, query, document or key, and `CERVEAU_LOG=off` turns it off.

## What you must do yourself

- **The implicit `.env` of the command line is limited.** Without `CERVEAU_ENV_FILE`, the `cerveau`
  command reads `.env` in the current folder; such a file is not allowed to point Qdrant or Ollama to
  another machine (your texts, or an API key set in your shell, would go there). To use a remote
  service, name your file with `CERVEAU_ENV_FILE`. A plain-http remote Qdrant prints a warning: the
  key would travel unencrypted.
- **On Windows, protect `CERVEAU_DATA_DIR` yourself** if other people use the same computer: the
  journal holds every memory in clear text, and the Qdrant data (the `cerveau_qdrant_storage` Docker volume)
  holds every memory and every indexed chunk in clear text too.
- **Keep `.env` private.** It contains the Qdrant key. It is in `.gitignore`.
- **Give your AI client `CERVEAU_ENV_FILE`, never `CERVEAU_QDRANT_API_KEY` itself.** Its configuration
  (`claude_desktop_config.json`, `claude mcp add -e`) is not a protected file: name your `.env` there and
  keep the key in it.
- **Keep Ollama on localhost.** Ollama listens on 127.0.0.1 by default; if you set `OLLAMA_HOST=0.0.0.0`
  (common for Docker or remote use), anyone on your network can read what it embeds.
- **An agent with a shell is not contained by these rules.** The guarantees above hold for an AI client
  that only talks MCP. A client that can run commands or read files (Claude Code, Cursor and similar)
  can run `cerveau archive`, read `.env` and its Qdrant key, or call Qdrant directly. Give such agents
  the MCP server only, and keep `.env` and `CERVEAU_DATA_DIR` out of their reach if that matters to you.
- **Output is escaped, for you and for the agent.** `cerveau list` and `cerveau search` show every
  invisible or control character escaped (escape sequences, zero-width and bidi characters, Unicode
  "tags"…), so a stored text can neither hide part of itself nor drive your terminal. `--json` escapes
  them the same way, and so does `search_memory` for the AI client: hidden text ("ASCII smuggling")
  reaches the model as visible escapes, which you can also see in the conversation.
  The price: some emoji and scripts that rely on invisible joiners (Persian, Indic…) show escapes.
- **Use HTTPS if Qdrant runs on another machine.** The key travels in a header; over plain HTTP it can be read.
- **Back up `CERVEAU_DATA_DIR`.** The journal is the only source of truth for memories. A memory
  purged later is still in older backups.
- **Be careful about what you index.** Anything indexed can be returned to any AI client connected to
  the MCP server, and from there to its cloud provider if the client uses one. Deleting a file does not
  remove it from the index: `cerveau index <folder> --prune` (or `cerveau unindex <path>`) does, and
  `cerveau index` warns when indexed files are gone.
- **Indexed documents can talk to your agent.** A document can contain instructions aimed at an AI
  ("remember that…"). Set `CERVEAU_MCP_ALLOW_WRITE=false` if agents should only read.
- **Memories written by an agent can talk to your next sessions.** What an agent stores is served back
  to every future session, instructions included. Review it with `cerveau list --actor mcp`.

## Reporting a vulnerability

Please open a private security advisory on the GitHub repository rather than a public issue.
