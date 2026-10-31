//! The registry and the settings as the front door sees them: through `Store`, which the program implements (glue
//! depends on foundation and contracts only: CONTRIBUTING.md) - the same calls as nextcuda-models' registry.

use std::path::{Path, PathBuf};

use serde_json::Value;

/// The model registry and the settings file
pub trait Store: Send + Sync {
    /// every entry
    fn models(&self) -> Result<Vec<Value>, String>;
    /// change an entry (the studio's chat modes kept in step): the entry as saved
    fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String>;
    /// the registry's file
    fn registry_file(&self) -> PathBuf;
    /// a setting (the environment over the files)
    fn get(&self, name: &str) -> Option<String>;
    /// whether the environment sets it
    fn in_env(&self, name: &str) -> bool;
    /// the settings file the page edits
    fn file(&self) -> PathBuf;
    /// an entry's prompt guide (Markdown), from the model cards
    fn guide(&self, entry: &Value) -> Option<String>;
    /// the model cards (each as JSON; none: no cards to read) - what an entry registered before a card key existed
    /// lacks (`headroom_gib`)
    fn cards(&self) -> Vec<Value> {
        Vec::new()
    }
    /// a card's files as `nextcuda models pull` lays them out: (path under the models folder, bytes); empty: no such
    /// card (pulls.rs: what a download is and how far it is)
    fn card_files(&self, _id: &str) -> Vec<(String, u64)> {
        Vec::new()
    }
}

impl<T: Store + ?Sized> Store for Box<T> {
    fn models(&self) -> Result<Vec<Value>, String> {
        (**self).models()
    }
    fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String> {
        (**self).update(id, f)
    }
    fn registry_file(&self) -> PathBuf {
        (**self).registry_file()
    }
    fn get(&self, name: &str) -> Option<String> {
        (**self).get(name)
    }
    fn in_env(&self, name: &str) -> bool {
        (**self).in_env(name)
    }
    fn file(&self) -> PathBuf {
        (**self).file()
    }
    fn guide(&self, entry: &Value) -> Option<String> {
        (**self).guide(entry)
    }
    fn cards(&self) -> Vec<Value> {
        (**self).cards()
    }
    fn card_files(&self, id: &str) -> Vec<(String, u64)> {
        (**self).card_files(id)
    }
}

/// An entry's kind: "llm", "image", "video", "audio", "model3d" or "lora" (entries from before kinds are llm)
pub fn kind_of(m: &Value) -> &str {
    m.get("kind").and_then(|v| v.as_str()).unwrap_or("llm")
}

/// What an entry or a card says of its weights' licence: `license`, `commercial` (false: it forbids commercial use),
/// `license_note` (the conditions it puts on it) - the keys it has, for a model's row
pub fn licence(src: &Value) -> serde_json::Map<String, Value> {
    ["license", "commercial", "license_note"].iter().filter_map(|k| Some((k.to_string(), src.get(*k)?.clone()))).collect()
}

/// Whether an entry is pinned: loaded and kept (no idle unload, never stopped for another model's room)
pub fn pinned(m: &Value) -> bool {
    m.get("pinned").and_then(Value::as_bool) == Some(true)
}

pub fn all(s: &dyn Store) -> Result<Vec<Value>, String> {
    s.models()
}

pub fn find(s: &dyn Store, id: &str) -> Result<Option<Value>, String> {
    Ok(s.models()?.into_iter().find(|m| m["id"] == id))
}

pub fn update(s: &dyn Store, id: &str, mut f: impl FnMut(&mut Value)) -> Result<Value, String> {
    s.update(id, &mut f)
}

pub fn registry(s: &dyn Store) -> PathBuf {
    s.registry_file()
}

/// The host paths an entry names: its file, its files by role, settings that are paths (`path`, `path:scale`)
pub fn paths(m: &Value) -> Vec<PathBuf> {
    let mut v = vec![PathBuf::from(m.get("file").and_then(Value::as_str).unwrap_or(""))];
    v.extend(m["files"].as_object().into_iter().flatten().filter_map(|(_, p)| p.as_str()).map(PathBuf::from));
    for val in m.get("env").and_then(Value::as_object).into_iter().flatten().filter_map(|(_, v)| v.as_str()) {
        for part in val.split(',') {
            let p = match part.rsplit_once(':') {
                Some((p, scale)) if scale.parse::<f64>().is_ok() => p,
                _ => part,
            };
            if p.starts_with('/') && Path::new(p).exists() {
                v.push(PathBuf::from(p));
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A registry kept in a JSON file, as models.json is
    struct File(PathBuf, Mutex<()>);

    impl Store for File {
        fn models(&self) -> Result<Vec<Value>, String> {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&self.0).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            Ok(v["models"].as_array().cloned().unwrap_or_default())
        }
        fn update(&self, id: &str, f: &mut dyn FnMut(&mut Value)) -> Result<Value, String> {
            let _g = self.1.lock().unwrap();
            let mut models = self.models()?;
            let m = models.iter_mut().find(|m| m["id"] == id).ok_or("no model")?;
            f(m);
            let out = m.clone();
            std::fs::write(&self.0, serde_json::json!({"models": models}).to_string()).map_err(|e| e.to_string())?;
            Ok(out)
        }
        fn registry_file(&self) -> PathBuf {
            self.0.clone()
        }
        fn get(&self, _: &str) -> Option<String> {
            None
        }
        fn in_env(&self, _: &str) -> bool {
            false
        }
        fn file(&self) -> PathBuf {
            PathBuf::new()
        }
        fn guide(&self, _: &Value) -> Option<String> {
            None
        }
    }

    #[test]
    fn a_pin_is_kept_in_the_registry_file_and_survives_a_restart() {
        let path = std::env::temp_dir().join(format!("nextcuda-pin-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::json!({"models": [{"id": "a", "enabled": true}, {"id": "b", "enabled": true}]}).to_string()).unwrap();
        let st = File(path.clone(), Mutex::new(()));
        update(&st, "a", |e| e["pinned"] = serde_json::json!(true)).unwrap();
        // a new reader of the same file (serve restarted): a is pinned, b is not
        let again = File(path.clone(), Mutex::new(()));
        assert!(pinned(&find(&again, "a").unwrap().unwrap()));
        assert!(!pinned(&find(&again, "b").unwrap().unwrap()));
        // disabling unpins (life::disable writes both)
        update(&again, "a", |e| {
            e["enabled"] = serde_json::json!(false);
            e["pinned"] = serde_json::json!(false);
        }).unwrap();
        assert!(!pinned(&find(&st, "a").unwrap().unwrap()));
        let _ = std::fs::remove_file(path);
    }
}
