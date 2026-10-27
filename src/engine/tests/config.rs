use std::path::PathBuf;

use crate::config::Config;

const CONFIG: &str = include_str!("../../../config/config.toml");

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gomoku-config-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.canonicalize().unwrap()
}

#[test]
fn relative_paths_are_resolved_from_project_root() {
    // プロジェクトの下のどこに置いても、pyproject.toml のあるディレクトリが基準になる
    let root = temp_dir("root");
    std::fs::write(root.join("pyproject.toml"), "").unwrap();
    let nested = root.join("a/b");
    std::fs::create_dir_all(&nested).unwrap();
    std::fs::write(nested.join("config.toml"), CONFIG).unwrap();

    let config = Config::load(&nested.join("config.toml")).unwrap();
    assert_eq!(config.paths.model_dir, root.join("runs/models"));
    assert_eq!(config.latest_model(), root.join("runs/models/latest.onnx"));
    std::fs::remove_dir_all(root).unwrap();
}
