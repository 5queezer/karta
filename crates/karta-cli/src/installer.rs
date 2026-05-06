use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
};

const START_MARKER: &str = "<!-- karta:start -->";
const END_MARKER: &str = "<!-- karta:end -->";
const PI_EXTENSION_TS: &str = include_str!("../../../.pi/extensions/karta.ts");
const CLAUDE_USER_PROMPT_HOOK_COMMAND: &str = r#"KARTA_HOOK_PAYLOAD="$(cat)" python3 - <<'PY'
import json, os, subprocess
try:
    payload = json.loads(os.environ.get('KARTA_HOOK_PAYLOAD') or '{}')
except Exception:
    payload = {}
prompt = payload.get('prompt') or payload.get('user_prompt') or ''
if not prompt.strip():
    raise SystemExit(0)
top_k = os.environ.get('KARTA_AUTO_CONTEXT_TOP_K', '5')
karta_bin = os.environ.get('KARTA_BIN', 'karta')
cmd = [karta_bin, '--json', 'search', '--query', prompt, '--top-k', top_k]
try:
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
    if result.returncode != 0:
        raise SystemExit(0)
    data = json.loads(result.stdout)
except Exception:
    raise SystemExit(0)
hits = data.get('results') or data.get('data', {}).get('results') or []
lines = []
for hit in hits[:5]:
    note = hit.get('note') or {}
    content = ' '.join((note.get('content') or '').split())
    if content:
        lines.append(f"- {note.get('id', 'unknown')}: {content[:700]}")
if not lines:
    raise SystemExit(0)
context = "Relevant durable Karta memories for this prompt:\n" + "\n".join(lines)
event_name = payload.get('hook_event_name') or payload.get('hookEventName') or 'UserPromptSubmit'
print(json.dumps({"hookSpecificOutput":{"hookEventName": event_name,"additionalContext": context}}))
PY"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Client {
    Claude,
    Codex,
    Gemini,
    Hermes,
    Pi,
}

impl std::str::FromStr for Client {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "claude" | "claude-code" => Ok(Self::Claude),
            "codex" => Ok(Self::Codex),
            "gemini" | "gemini-cli" => Ok(Self::Gemini),
            "hermes" => Ok(Self::Hermes),
            "pi" | "pi-coding-agent" => Ok(Self::Pi),
            other => {
                bail!("unknown client '{other}'. Choose from: claude, codex, gemini, hermes, pi")
            }
        }
    }
}

pub fn parse_client(value: &str) -> std::result::Result<Client, String> {
    value.parse::<Client>().map_err(|error| error.to_string())
}

impl Client {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Claude => "claude",
            Self::Codex => "codex",
            Self::Gemini => "gemini",
            Self::Hermes => "hermes",
            Self::Pi => "pi",
        }
    }

    fn skill_path(self, home: &Path) -> PathBuf {
        match self {
            Self::Claude => home.join(".claude/skills/karta/SKILL.md"),
            Self::Codex => home.join(".agents/skills/karta/SKILL.md"),
            Self::Gemini => home.join(".gemini/skills/karta/SKILL.md"),
            Self::Hermes => home.join(".hermes/skills/karta/SKILL.md"),
            Self::Pi => home.join(".pi/agent/skills/karta/SKILL.md"),
        }
    }
}

pub fn install_client(client: Client, home: &Path, project: &Path) -> Result<Vec<PathBuf>> {
    let mut changed = Vec::new();
    let skill_path = client.skill_path(home);
    write_file(&skill_path, skill_content(client))?;
    changed.push(skill_path.clone());

    match client {
        Client::Claude => {
            let path = home.join(".claude/CLAUDE.md");
            upsert_section(&path, &global_section("CLAUDE.md"))?;
            changed.push(path);

            let hook_path = project.join(".claude/settings.json");
            install_claude_hook(&hook_path)?;
            changed.push(hook_path);
        }
        Client::Codex => {
            let path = project.join("AGENTS.md");
            upsert_section(&path, &project_section("AGENTS.md"))?;
            changed.push(path);

            let config_path = project.join(".codex/config.toml");
            install_codex_config(&config_path)?;
            changed.push(config_path);

            let hook_path = project.join(".codex/hooks.json");
            install_json_hook(&hook_path, "UserPromptSubmit")?;
            changed.push(hook_path);
        }
        Client::Gemini => {
            let path = project.join("GEMINI.md");
            upsert_section(&path, &project_section("GEMINI.md"))?;
            changed.push(path);

            let settings_path = project.join(".gemini/settings.json");
            install_json_hook(&settings_path, "BeforeAgent")?;
            changed.push(settings_path);
        }
        Client::Hermes => {}
        Client::Pi => {
            let extension_path = home.join(".pi/agent/extensions/karta.ts");
            write_file(&extension_path, PI_EXTENSION_TS.to_string())?;
            changed.push(extension_path);
        }
    }

    Ok(changed)
}

pub fn uninstall_client(client: Client, home: &Path, project: &Path) -> Result<Vec<PathBuf>> {
    let mut changed = Vec::new();
    let skill_path = client.skill_path(home);
    if skill_path.exists() {
        fs::remove_file(&skill_path)
            .with_context(|| format!("failed to remove {}", skill_path.display()))?;
        changed.push(skill_path.clone());
        prune_empty_dirs(skill_path.parent());
    }

    match client {
        Client::Claude => {
            let path = home.join(".claude/CLAUDE.md");
            if remove_section(&path)? {
                changed.push(path);
            }

            let hook_path = project.join(".claude/settings.json");
            if uninstall_claude_hook(&hook_path)? {
                changed.push(hook_path);
            }
        }
        Client::Codex => {
            let path = project.join("AGENTS.md");
            if remove_section(&path)? {
                changed.push(path);
            }

            let hook_path = project.join(".codex/hooks.json");
            if uninstall_json_hook(&hook_path, "UserPromptSubmit")? {
                changed.push(hook_path);
            }
        }
        Client::Gemini => {
            let path = project.join("GEMINI.md");
            if remove_section(&path)? {
                changed.push(path);
            }

            let settings_path = project.join(".gemini/settings.json");
            if uninstall_json_hook(&settings_path, "BeforeAgent")? {
                changed.push(settings_path);
            }
        }
        Client::Hermes => {}
        Client::Pi => {
            let extension_path = home.join(".pi/agent/extensions/karta.ts");
            if extension_path.exists() {
                fs::remove_file(&extension_path)
                    .with_context(|| format!("failed to remove {}", extension_path.display()))?;
                changed.push(extension_path.clone());
                prune_empty_dirs(extension_path.parent());
            }
        }
    }

    Ok(changed)
}

pub fn default_home_dir() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("HOME") {
        return Ok(PathBuf::from(home));
    }
    if let Some(home) = std::env::var_os("USERPROFILE") {
        return Ok(PathBuf::from(home));
    }
    bail!("could not determine home directory; set HOME")
}

fn write_file(path: &Path, content: String) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))
}

fn upsert_section(path: &Path, section: &str) -> Result<()> {
    let existing = fs::read_to_string(path).unwrap_or_default();
    let without_old = strip_section(&existing);
    let mut next = without_old.trim_end().to_string();
    if !next.is_empty() {
        next.push_str("\n\n");
    }
    next.push_str(section.trim_end());
    next.push('\n');
    write_file(path, next)
}

fn remove_section(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let existing =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let stripped = strip_section(&existing);
    if stripped == existing {
        return Ok(false);
    }
    let next = stripped.trim().to_string();
    if next.is_empty() {
        fs::remove_file(path).with_context(|| format!("failed to remove {}", path.display()))?;
    } else {
        write_file(path, format!("{}\n", next))?;
    }
    Ok(true)
}

fn install_claude_hook(path: &Path) -> Result<()> {
    install_json_hook(path, "UserPromptSubmit")
}

fn uninstall_claude_hook(path: &Path) -> Result<bool> {
    uninstall_json_hook(path, "UserPromptSubmit")
}

fn install_json_hook(path: &Path, event: &str) -> Result<()> {
    let mut settings = read_json_object(path)?;
    let hooks = settings
        .as_object_mut()
        .expect("read_json_object always returns an object")
        .entry("hooks")
        .or_insert_with(|| json!({}));
    if !hooks.is_object() {
        *hooks = json!({});
    }
    let event_hooks = hooks
        .as_object_mut()
        .expect("hooks was normalized to an object")
        .entry(event)
        .or_insert_with(|| json!([]));
    if !event_hooks.is_array() {
        *event_hooks = json!([]);
    }

    let entries = event_hooks
        .as_array_mut()
        .expect("event hook list was normalized to an array");
    entries.retain(|entry| !entry.to_string().contains("karta-memory"));
    entries.push(json!({
        "name": "karta-memory",
        "hooks": [
            {
                "type": "command",
                "command": CLAUDE_USER_PROMPT_HOOK_COMMAND,
            }
        ]
    }));

    write_json(path, &settings)
}

fn uninstall_json_hook(path: &Path, event: &str) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    let mut settings = read_json_object(path)?;
    let Some(event_hooks) = settings
        .get_mut("hooks")
        .and_then(|hooks| hooks.get_mut(event))
        .and_then(Value::as_array_mut)
    else {
        return Ok(false);
    };

    let original_len = event_hooks.len();
    event_hooks.retain(|entry| !entry.to_string().contains("karta-memory"));
    if event_hooks.len() == original_len {
        return Ok(false);
    }
    write_json(path, &settings)?;
    Ok(true)
}

fn install_codex_config(path: &Path) -> Result<()> {
    let existing = fs::read_to_string(path).unwrap_or_default();
    let mut lines: Vec<String> = existing.lines().map(ToString::to_string).collect();
    let mut features_index = None;
    let mut codex_hooks_index = None;

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed == "[features]" {
            features_index = Some(index);
        } else if trimmed.starts_with("codex_hooks") {
            codex_hooks_index = Some(index);
        }
    }

    if let Some(index) = codex_hooks_index {
        lines[index] = "codex_hooks = true".to_string();
    } else if let Some(index) = features_index {
        lines.insert(index + 1, "codex_hooks = true".to_string());
    } else {
        if !lines.is_empty() && !lines.last().is_some_and(|line| line.trim().is_empty()) {
            lines.push(String::new());
        }
        lines.push("[features]".to_string());
        lines.push("codex_hooks = true".to_string());
    }

    write_file(path, format!("{}\n", lines.join("\n")))
}

fn read_json_object(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let value = serde_json::from_str::<Value>(&text).unwrap_or_else(|_| json!({}));
    if value.is_object() {
        Ok(value)
    } else {
        Ok(json!({}))
    }
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    write_file(path, format!("{}\n", serde_json::to_string_pretty(value)?))
}

fn strip_section(content: &str) -> String {
    let Some(start) = content.find(START_MARKER) else {
        return content.to_string();
    };
    let Some(end_relative) = content[start..].find(END_MARKER) else {
        return content.to_string();
    };
    let end = start + end_relative + END_MARKER.len();
    format!("{}{}", &content[..start], &content[end..])
}

fn prune_empty_dirs(mut dir: Option<&Path>) {
    while let Some(path) = dir {
        if fs::remove_dir(path).is_err() {
            break;
        }
        dir = path.parent();
    }
}

fn skill_content(client: Client) -> String {
    format!(
        r#"---
name: karta
description: Persistent agentic memory for durable project facts, decisions, preferences, bug root causes, and architectural context. Use before answering questions that may depend on prior project history; add concise notes for stable facts worth remembering.
---

# Karta Memory ({client})

Karta is an agentic memory system available through the `karta` CLI.

## When to use
- Search Karta before answering questions that may depend on previous project decisions, maintainer preferences, recurring bugs, or architectural context.
- Add a note only for stable, reusable information: architecture decisions, constraints, preferences, root causes, and important follow-ups.
- Do not store secrets, raw logs, transient scratch work, speculative guesses, or large code blocks.

## Commands
- `karta search --query "..." --top-k 5`
- `karta ask --query "..." --top-k 5`
- `karta add-note --content "..."`
- `karta health`

Prefer `--json` when the client needs machine-readable output.
"#,
        client = client.as_str()
    )
}

fn global_section(_file_name: &str) -> String {
    section("The Karta skill is installed globally for this client.")
}

fn project_section(_file_name: &str) -> String {
    section("This project is configured to use Karta memory.")
}

fn section(intro: &str) -> String {
    format!(
        r#"{START_MARKER}
## karta

{intro}

Rules:
- Before answering questions that may depend on previous project context, run `karta search --query "<question>" --top-k 5` or `karta ask --query "<question>" --top-k 5`.
- Store only stable, reusable memory with `karta add-note --content "<concise fact>"`.
- Do not store secrets, raw logs, transient scratch work, speculative guesses, or large code blocks.
- Prefer `karta --json ...` for automation.

Installed by `karta install`; remove with `karta uninstall`.
{END_MARKER}
"#,
        START_MARKER = START_MARKER,
        END_MARKER = END_MARKER,
        intro = intro,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn read(path: &std::path::Path) -> String {
        fs::read_to_string(path).unwrap()
    }

    #[test]
    fn install_claude_writes_skill_registers_claude_md_and_user_prompt_hook() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let project = tmp.path().join("project");
        fs::create_dir_all(&project).unwrap();

        install_client(Client::Claude, &home, &project).unwrap();

        let skill = home.join(".claude/skills/karta/SKILL.md");
        assert!(skill.exists());
        assert!(read(&skill).contains("karta"));
        let claude_md = home.join(".claude/CLAUDE.md");
        assert!(read(&claude_md).contains("## karta"));
        let settings = read(&project.join(".claude/settings.json"));
        assert!(settings.contains("UserPromptSubmit"));
        assert!(settings.contains("karta"));
    }

    #[test]
    fn install_codex_writes_skill_project_agents_md_and_hooks_idempotently() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let project = tmp.path().join("project");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("AGENTS.md"), "# Existing\n\nKeep me.\n").unwrap();

        install_client(Client::Codex, &home, &project).unwrap();
        install_client(Client::Codex, &home, &project).unwrap();

        assert!(home.join(".agents/skills/karta/SKILL.md").exists());
        let agents = read(&project.join("AGENTS.md"));
        assert!(agents.contains("Keep me."));
        assert_eq!(agents.matches("## karta").count(), 1);
        assert!(read(&project.join(".codex/config.toml")).contains("codex_hooks = true"));
        let hooks = read(&project.join(".codex/hooks.json"));
        assert!(hooks.contains("UserPromptSubmit"));
        assert_eq!(hooks.matches("karta-memory").count(), 1);
    }

    #[test]
    fn install_gemini_writes_skill_project_gemini_md_and_before_agent_hook() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let project = tmp.path().join("project");
        fs::create_dir_all(&project).unwrap();

        install_client(Client::Gemini, &home, &project).unwrap();

        assert!(home.join(".gemini/skills/karta/SKILL.md").exists());
        assert!(read(&project.join("GEMINI.md")).contains("## karta"));
        let settings = read(&project.join(".gemini/settings.json"));
        assert!(settings.contains("BeforeAgent"));
        assert!(settings.contains("karta"));
    }

    #[test]
    fn install_hermes_writes_skill_and_pi_writes_extension_plus_skill() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let project = tmp.path().join("project");
        fs::create_dir_all(&project).unwrap();

        install_client(Client::Hermes, &home, &project).unwrap();
        install_client(Client::Pi, &home, &project).unwrap();

        assert!(home.join(".hermes/skills/karta/SKILL.md").exists());
        assert!(home.join(".pi/agent/skills/karta/SKILL.md").exists());
        let extension = home.join(".pi/agent/extensions/karta.ts");
        assert!(extension.exists());
        assert!(read(&extension).contains("karta_add_note"));
    }

    #[test]
    fn parse_client_names_and_reject_unknown() {
        assert_eq!("claude".parse::<Client>().unwrap(), Client::Claude);
        assert_eq!("codex".parse::<Client>().unwrap(), Client::Codex);
        assert_eq!("gemini".parse::<Client>().unwrap(), Client::Gemini);
        assert_eq!("hermes".parse::<Client>().unwrap(), Client::Hermes);
        assert_eq!("pi".parse::<Client>().unwrap(), Client::Pi);
        assert!("cursor".parse::<Client>().is_err());
    }

    #[test]
    fn uninstall_removes_karta_section_but_preserves_existing_content() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let project = tmp.path().join("project");
        fs::create_dir_all(&project).unwrap();
        fs::write(project.join("AGENTS.md"), "# Existing\n\nKeep me.\n").unwrap();
        install_client(Client::Codex, &home, &project).unwrap();

        uninstall_client(Client::Codex, &home, &project).unwrap();

        assert!(!home.join(".agents/skills/karta/SKILL.md").exists());
        let agents = read(&project.join("AGENTS.md"));
        assert!(agents.contains("Keep me."));
        assert!(!agents.contains("## karta"));
    }

    #[test]
    fn uninstall_claude_and_pi_remove_tightly_coupled_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let project = tmp.path().join("project");
        fs::create_dir_all(&project).unwrap();
        install_client(Client::Claude, &home, &project).unwrap();
        install_client(Client::Pi, &home, &project).unwrap();

        uninstall_client(Client::Claude, &home, &project).unwrap();
        uninstall_client(Client::Pi, &home, &project).unwrap();

        assert!(!home.join(".claude/skills/karta/SKILL.md").exists());
        assert!(!home.join(".pi/agent/skills/karta/SKILL.md").exists());
        assert!(!home.join(".pi/agent/extensions/karta.ts").exists());
        let settings = read(&project.join(".claude/settings.json"));
        assert!(!settings.contains("karta"));
    }
}
