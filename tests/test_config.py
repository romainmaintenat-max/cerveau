import os
from pathlib import Path

import pytest

from cerveau.config import ConfigError, Settings, parse_env_file


def test_defaults_are_local_and_need_no_key():
    s = Settings.from_env(env={}, env_file=None)
    assert s.qdrant_url == "http://127.0.0.1:6333"
    assert s.embed_model == "bge-m3"
    assert s.vector_size == 1024


def test_remote_qdrant_without_key_is_refused():
    with pytest.raises(ConfigError, match="API key"):
        Settings.from_env(env={"CERVEAU_QDRANT_URL": "http://10.0.0.5:6333"}, env_file=None)


def test_remote_qdrant_with_key_is_accepted():
    s = Settings.from_env(
        env={"CERVEAU_QDRANT_URL": "https://q.example:6333", "CERVEAU_QDRANT_API_KEY": "k"},
        env_file=None,
    )
    assert s.qdrant_api_key == "k"


def test_env_file_is_read_and_environment_wins(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nCERVEAU_PREFIX=fromfile\nCERVEAU_VECTOR_SIZE='768'\nCERVEAU_EMBED_MODEL=\"m\"\n",
        encoding="utf-8",
    )
    s = Settings.from_env(env={"CERVEAU_PREFIX": "fromenv"}, env_file=env_file)
    assert s.prefix == "fromenv"
    assert s.vector_size == 768
    assert s.embed_model == "m"


def test_bad_values_are_rejected(tmp_path: Path):
    with pytest.raises(ConfigError):
        Settings.from_env(env={"CERVEAU_VECTOR_SIZE": "abc"}, env_file=None)
    with pytest.raises(ConfigError):
        Settings.from_env(env={"CERVEAU_EMBEDDER": "openai"}, env_file=None)
    with pytest.raises(ConfigError):
        Settings.from_env(env={"CERVEAU_PREFIX": "a b"}, env_file=None)
    with pytest.raises(ConfigError):
        Settings.from_env(env={"CERVEAU_MCP_ALLOW_WRITE": "maybe"}, env_file=None)
    bad = tmp_path / "bad.env"
    bad.write_text("CERVEAU_NOEQUALSIGN\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        parse_env_file(bad)


def test_chunk_must_fit_in_text_limit():
    with pytest.raises(ConfigError):
        Settings.from_env(
            env={"CERVEAU_CHUNK_MAX_CHARS": "5000", "CERVEAU_MAX_TEXT_CHARS": "1000"}, env_file=None
        )


def test_foreign_lines_bom_comments_and_export_are_handled(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OTHER_TOOL_CERT=-----BEGIN\nnot even key value\n"
        "export CERVEAU_PREFIX=abc  # my prefix\n"
        "CERVEAU_EMBED_MODEL='m # not a comment'\n",
        encoding="utf-8-sig",
    )
    s = Settings.from_env(env={}, env_file=env_file)
    assert s.prefix == "abc" and s.embed_model == "m # not a comment"


def test_empty_environment_variable_does_not_hide_the_file_value(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("CERVEAU_PREFIX=fromfile\n", encoding="utf-8")
    s = Settings.from_env(env={"CERVEAU_PREFIX": ""}, env_file=env_file)
    assert s.prefix == "fromfile"


def test_missing_explicit_env_file_is_an_error(tmp_path: Path):
    with pytest.raises(ConfigError, match="missing file"):
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(tmp_path / "nope.env")})


def test_relative_data_dir_is_resolved_against_the_env_file(tmp_path: Path, monkeypatch):
    env_file = tmp_path / "conf" / ".env"
    env_file.parent.mkdir()
    env_file.write_text("CERVEAU_DATA_DIR=data\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    s = Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})
    assert s.data_dir == (tmp_path / "conf" / "data").resolve()
    assert s.env_file == env_file.resolve()


def test_relative_data_dir_from_environment_is_refused():
    with pytest.raises(ConfigError, match="absolute"):
        Settings.from_env(env={"CERVEAU_DATA_DIR": "data"}, env_file=None)


def test_implicit_dot_env_in_current_directory(tmp_path: Path, monkeypatch):
    (tmp_path / ".env").write_text("CERVEAU_PREFIX=here\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert Settings.from_env(env={}).prefix == "here"


def test_remote_ollama_warns():
    with pytest.warns(UserWarning, match="every text"):
        Settings.from_env(env={"CERVEAU_OLLAMA_URL": "http://192.168.1.20:11434"}, env_file=None)


def test_api_key_is_not_in_the_repr():
    s = Settings(qdrant_api_key="SECRETKEY")
    assert "SECRETKEY" not in repr(s)


def test_unknown_settings_are_refused(tmp_path: Path):
    with pytest.raises(ConfigError, match="CERVEAU_MCP_ALLOW_WRIT"):
        Settings.from_env(env={"CERVEAU_MCP_ALLOW_WRIT": "false"}, env_file=None)
    env_file = tmp_path / ".env"
    env_file.write_text("CERVEAU_PREFX=typo\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="CERVEAU_PREFX"):
        Settings.from_env(env={}, env_file=env_file)
    Settings.from_env(env={"CERVEAU_IT_QDRANT_URL": "x"}, env_file=None)  # test settings are fine


def test_empty_allow_write_is_an_error_not_the_default(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("CERVEAU_MCP_ALLOW_WRITE=\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="ALLOW_WRITE"):
        Settings.from_env(env={}, env_file=env_file)


def test_env_file_in_utf16_is_read_and_ansi_is_explained(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_bytes("CERVEAU_PREFIX=utf16\r\n".encode("utf-16"))
    assert Settings.from_env(env={}, env_file=env_file).prefix == "utf16"
    env_file.write_bytes("CERVEAU_PREFIX=caf\xe9\n".encode("cp1252"))
    with pytest.raises(ConfigError, match="UTF-8"):
        Settings.from_env(env={}, env_file=env_file)


def test_variables_in_data_dir_are_expanded(tmp_path: Path):
    import os

    var = "%HOME_X%" if os.name == "nt" else "$HOME_X"
    s = Settings.from_env(env={"HOME_X": str(tmp_path), "CERVEAU_DATA_DIR": f"{var}/brain"}, env_file=None)
    assert s.data_dir == tmp_path / "brain"
    missing = "%NOT_DEFINED_XYZ%" if os.name == "nt" else "$NOT_DEFINED_XYZ"
    with pytest.raises(ConfigError, match="undefined"):
        Settings.from_env(env={"CERVEAU_DATA_DIR": f"{missing}/brain"}, env_file=None)


def test_variable_syntax_follows_the_platform():
    from cerveau.config import _expand_vars

    env = {"USERPROFILE": r"C:\Users\Bob", "HOME": "/home/bob"}
    assert _expand_vars(r"%UserProfile%\.cerveau", env, windows=True) == r"C:\Users\Bob\.cerveau"
    assert _expand_vars(r"C:\a$b\notes", env, windows=True) == r"C:\a$b\notes"  # "$" is literal
    assert _expand_vars("%LOCALAPPDATA%", {"LocalAppData": "C:/L"}, windows=True) == "C:/L"
    assert _expand_vars("${HOME}/.cerveau", env, windows=False) == "/home/bob/.cerveau"
    assert _expand_vars("/data/100%/x", env, windows=False) == "/data/100%/x"


def test_lowercase_and_url_mistakes_are_refused(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("cerveau_mcp_allow_write=false\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="capitals"):
        Settings.from_env(env={}, env_file=env_file)
    with pytest.raises(ConfigError, match="unknown"):
        Settings.from_env(env={"cerveau_mcp_allow_write": "false"}, env_file=None)
    for url in ("ftp://127.0.0.1:6333", "http://127.0.0.1:99999", "127.0.0.1:6333"):
        with pytest.raises(ConfigError):
            Settings.from_env(env={"CERVEAU_QDRANT_URL": url}, env_file=None)


@pytest.mark.parametrize("prefix", ["é", "ab²", "x" * 41, ""])
def test_prefix_must_be_short_ascii(prefix):
    with pytest.raises(ConfigError):
        Settings(prefix=prefix).validate()


def test_credentials_in_urls_are_hidden_when_displayed():
    from cerveau.config import redact_url

    assert redact_url("http://user:pw@127.0.0.1:6333/x") == "http://127.0.0.1:6333/x"
    assert redact_url("http://127.0.0.1:6333") == "http://127.0.0.1:6333"


def test_memory_qdrant_is_for_tests_only():
    with pytest.raises(ConfigError, match="test"):
        Settings.from_env(env={"CERVEAU_QDRANT_URL": ":memory:"}, env_file=None)


@pytest.mark.parametrize("url", ["http://evil.example\\@localhost:11434", "http://local host:1"])
def test_ambiguous_urls_are_refused(url):
    with pytest.raises(ConfigError):
        Settings.from_env(env={"CERVEAU_OLLAMA_URL": url}, env_file=None)


def test_dot_env_of_the_current_folder_cannot_send_data_elsewhere(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("CERVEAU_QDRANT_URL=http://10.9.9.9:6333\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="current folder"):
        Settings.from_env(env={"CERVEAU_QDRANT_API_KEY": "my-shell-key"})
    explicit = Settings.from_env(
        env={"CERVEAU_QDRANT_API_KEY": "k", "CERVEAU_ENV_FILE": str(tmp_path / ".env")}
    )
    assert explicit.qdrant_url == "http://10.9.9.9:6333"


def test_plain_http_to_a_remote_qdrant_warns():
    with pytest.warns(UserWarning, match="unencrypted"):
        Settings.from_env(
            env={"CERVEAU_QDRANT_URL": "http://10.0.0.5:6333", "CERVEAU_QDRANT_API_KEY": "k"}, env_file=None
        )


@pytest.mark.parametrize("key", ["abc\u201cdef", "abc def", "abc\u00a0def", "pa$$word"])
def test_suspicious_api_keys_are_refused(key):
    with pytest.raises(ConfigError, match="CERVEAU_QDRANT_API_KEY"):
        Settings.from_env(env={"CERVEAU_QDRANT_API_KEY": key}, env_file=None)


def test_odd_values_are_config_errors_not_tracebacks(tmp_path, monkeypatch):
    import os

    if os.name != "nt":  # Windows resolves ~user without checking that the user exists
        with pytest.raises(ConfigError):
            Settings.from_env(env={"CERVEAU_DATA_DIR": "~nonexistent_user_xyz/x"}, env_file=None)
    (tmp_path / ".env").write_text("CERVEAU_OLLAMA_URL=http://[::1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError):
        Settings.from_env(env={})


def test_compose_port_setting_is_known(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("CERVEAU_QDRANT_PORT=6400\n", encoding="utf-8")
    Settings.from_env(env={}, env_file=env_file)


def test_api_key_with_a_backslash_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="must not contain"):
        Settings(qdrant_api_key="abc\\def", data_dir=tmp_path).validate()


def test_undecodable_dot_env_of_the_current_folder_is_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_bytes(b"OTHER_TOOL=caf\xe9\n")
    with pytest.warns(UserWarning, match="not UTF-8"):
        settings = Settings.from_env(env={"CERVEAU_DATA_DIR": str(tmp_path / "d")})
    assert settings.env_file is None


def test_undecodable_named_env_file_is_still_an_error(tmp_path):
    env_file = tmp_path / "my.env"
    env_file.write_bytes(b"CERVEAU_PREFIX=caf\xe9\n")
    with pytest.raises(ConfigError, match="UTF-8"):
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})


def test_malformed_dot_env_of_the_current_folder_is_still_an_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("CERVEAU_PREFIX\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        Settings.from_env(env={})


def test_env_file_setting_inside_a_dot_env_gets_its_own_message(tmp_path):
    env_file = tmp_path / "my.env"
    env_file.write_text("CERVEAU_ENV_FILE=/elsewhere/.env\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot be set inside a .env file"):
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})


def test_bad_urls_never_show_their_password():
    import pytest as _pytest

    for url in ("http://user:s3cretpw@host:99999", "http://user:s3cretpw@[::1", "ftp://u:s3cretpw@h"):
        with _pytest.raises(ConfigError) as caught:
            Settings.from_env(env={"CERVEAU_OLLAMA_URL": url}, env_file=None)
        assert "s3cretpw" not in str(caught.value)


def test_a_short_api_key_gets_a_warning(tmp_path):
    with pytest.warns(UserWarning, match="short"):
        Settings(qdrant_api_key="changeme", data_dir=tmp_path).validate()


def test_a_setting_with_the_wrong_separator_is_refused_not_ignored(tmp_path):
    for line in ("CERVEAU-MCP-ALLOW-WRITE=false", "CERVEAU MCP_ALLOW_WRITE=false", "Cerveau.prefix=x"):
        env_file = tmp_path / "x.env"
        env_file.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="did you mean"):
            Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})
    with pytest.raises(ConfigError, match="CERVEAU_MCP_ALLOW_WRITE"):
        Settings.from_env(env={"CERVEAU-MCP-ALLOW-WRITE": "false"}, env_file=None)
    env_file.write_text("OTHER-TOOL=1\nCERVEAUTOMATE=1\n", encoding="utf-8")
    with pytest.warns(UserWarning, match="CERVEAUTOMATE is ignored"):  # far from every setting: not ours
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})
    env_file.write_text("OTHER-TOOL=1\n", encoding="utf-8")  # not ours: ignored
    Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})


def test_glued_or_invisible_mistakes_in_a_name_are_refused(tmp_path):
    env_file = tmp_path / "x.env"
    for line in ("CERVEAUMCP_ALLOW_WRITE=false", "\u200bCERVEAU_MCP_ALLOW_WRITE=false"):
        env_file.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="did you mean CERVEAU_MCP_ALLOW_WRITE"):
            Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})
    with pytest.raises(ConfigError, match="did you mean CERVEAU_MCP_ALLOW_WRITE"):
        Settings.from_env(env={"CERVEAUMCP_ALLOW_WRITE": "false"}, env_file=None)


def test_an_empty_environment_variable_leaves_the_file_value(tmp_path):
    env_file = tmp_path / "x.env"
    env_file.write_text("CERVEAU_MCP_ALLOW_WRITE=false\n", encoding="utf-8")
    settings = Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file), "CERVEAU_MCP_ALLOW_WRITE": ""})
    assert settings.mcp_allow_write is False


def test_a_relative_env_file_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("CERVEAU_PREFIX=planted\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="absolute path"):
        Settings.from_env(env={"CERVEAU_ENV_FILE": ".env"}, allow_cwd_env=False)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_an_env_file_readable_by_others_gets_a_warning(tmp_path):
    import warnings

    env_file = tmp_path / "x.env"
    env_file.write_text("CERVEAU_PREFIX=x\n", encoding="utf-8")
    env_file.chmod(0o644)
    with pytest.warns(UserWarning, match="can be read by other users: run `chmod 600"):  # named: ours
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})
    env_file.write_text("CERVEAU_PREFIX=x\nCERVEAU_QDRANT_API_KEY=a-long-enough-test-key\n", encoding="utf-8")
    for mode in (0o644, 0o640, 0o604):
        env_file.chmod(mode)
        with pytest.warns(UserWarning, match="chmod 600"):
            Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})
    env_file.chmod(0o600)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)}).prefix == "x"


@pytest.mark.parametrize(
    "name",
    [
        "CERVEAU_MCP_ALLOW_WRIT",
        "CERVEAU_MCP_ALOW_WRITE",
        "CERVEAU_PREFIXE",
        "CERVEAUMCP_ALLOW_WRITE",
        "CERVEAU-MCP-ALLOW-WRITE",
    ],
)
def test_a_name_close_to_a_setting_is_refused(name, tmp_path):
    with pytest.raises(ConfigError, match="did you mean .*belongs to another program, rename or unset it"):
        Settings.from_env(env={name: "x"}, env_file=None)
    env_file = tmp_path / "x.env"
    env_file.write_text(f"{name}=x\n", encoding="utf-8")
    env_file.chmod(0o600)
    with pytest.raises(ConfigError, match="did you mean .*belongs to another program, rename or unset it"):
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})


@pytest.mark.parametrize(
    ("name", "meant"),
    [
        ("CERVEAU_ALLOW_WRITE", "CERVEAU_MCP_ALLOW_WRITE"),  # words of a setting, some missing
        ("CERVEAU_WRITE", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_MCP_WRITE", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_API_KEY", "CERVEAU_QDRANT_API_KEY"),
        ("CERVEAU_DATA", "CERVEAU_DATA_DIR"),
        ("CERVEAU_PREFIX_OLD", "CERVEAU_PREFIX"),  # words of a setting, and more
        ("CERVEAU_READONLY", "CERVEAU_MCP_ALLOW_WRITE"),  # about writing: never ignored
        ("CERVEAU_READ_ONLY", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_NOWRITE", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU-ALLOW-WRITE", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_LOGS", "CERVEAU_LOG"),
        ("CERVEAU_MCP_WRITABLE", "CERVEAU_MCP_ALLOW_WRITE"),  # no "WRITE" in it
        ("CERVEAU_WRITABLE", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_MCP_WRITING", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_NO_REMEMBER", "CERVEAU_MCP_ALLOW_WRITE"),  # the MCP write tool
        ("CERVEAU_DISABLE_REMEMBER", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_MCP_REMEMBER", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_REMEMBER", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_MCP_RO", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_RO", "CERVEAU_MCP_ALLOW_WRITE"),
        ("CERVEAU_MCP_RW", "CERVEAU_MCP_ALLOW_WRITE"),
    ],
)
def test_a_name_with_the_words_of_a_setting_or_about_writing_is_refused(name, meant, tmp_path):
    expected = f"{name} \\(did you mean {meant}\\?\\).*belongs to another program"
    with pytest.raises(ConfigError, match=expected):
        Settings.from_env(env={name: "false"}, env_file=None)
    env_file = tmp_path / "x.env"
    env_file.write_text(f"{name}=false\n", encoding="utf-8")
    env_file.chmod(0o600)
    with pytest.raises(ConfigError, match=f"{name}.*did you mean {meant}\\?"):
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})


def test_log_setting(tmp_path):
    assert Settings.from_env(env={}, env_file=None).log is True
    assert Settings.from_env(env={"CERVEAU_LOG": "off"}, env_file=None).log is False
    with pytest.raises(ConfigError, match="CERVEAU_LOG: expected a boolean"):
        Settings.from_env(env={"CERVEAU_LOG": "maybe"}, env_file=None)
    env_file = tmp_path / "x.env"
    env_file.write_text("CERVEAU_LOG=off\n", encoding="utf-8")
    env_file.chmod(0o600)
    assert Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)}).log is False
    assert Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file), "CERVEAU_LOG": "on"}).log is True
    env_file.write_text("CERVEAU_LOG=\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="CERVEAU_LOG"):
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})


@pytest.mark.parametrize(
    "name",
    [
        "CERVEAU_TOKEN",
        "CERVEAU_HOME",
        "CERVEAU_DB",
        "CERVEAU_API_URL",
        "CERVEAU-TOKEN",
        "CERVEAU_LGO",
        "CERVEAU_CACHE",
        "CERVEAU_DEBUG",
        "CERVEAU_ROOT",  # "RO" and "RW" are whole words only
        "CERVEAU_ROUTER",
    ],
)
def test_a_name_far_from_every_setting_is_ignored_with_a_warning(name, tmp_path):
    with pytest.warns(UserWarning, match=f"{name} is ignored: it does not look like any Cerveau setting"):
        settings = Settings.from_env(env={name: "x", "CERVEAU_PREFIX": "started"}, env_file=None)
    assert settings.prefix == "started"
    env_file = tmp_path / "x.env"
    env_file.write_text(f"{name}=x\nCERVEAU_PREFIX=fromfile\n", encoding="utf-8")
    env_file.chmod(0o600)
    with pytest.warns(UserWarning, match=f"{name} is ignored"):
        settings = Settings.from_env(env={"CERVEAU_ENV_FILE": str(env_file)})
    assert settings.prefix == "fromfile"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_env_file_permission_warnings(tmp_path, monkeypatch):
    import warnings

    monkeypatch.chdir(tmp_path)
    implicit = tmp_path / ".env"
    implicit.write_text("CERVEAU_PREFIX=x\n", encoding="utf-8")
    implicit.chmod(0o666)  # anyone may write it, key or not
    with pytest.warns(UserWarning, match="can be modified by other users and decides where your texts go"):
        Settings.from_env(env={})
    implicit.chmod(0o644)  # readable, no key, maybe another project's file: nothing said
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Settings.from_env(env={})
    named = tmp_path / "named.env"
    named.write_text("CERVEAU_PREFIX=x\n", encoding="utf-8")
    named.chmod(0o644)  # named by the user: theirs, even without a key
    with pytest.warns(UserWarning, match="can be read by other users"):
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(named)})
    named.write_text("CERVEAU_QDRANT_API_KEY=a-long-enough-test-key\n", encoding="utf-8")
    named.chmod(0o666)  # both at once: a single warning
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Settings.from_env(env={"CERVEAU_ENV_FILE": str(named)})
    assert len([w for w in caught if "chmod 600" in str(w.message)]) == 1
    for path in (implicit, named):
        path.chmod(0o600)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Settings.from_env(env={"CERVEAU_ENV_FILE": str(path)} if path is named else {})
