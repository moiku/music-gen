# CLAUDE.md

生成AIで音楽（Instagram・CM 用の BGM）を作るシステム。趣味の個人プロジェクト。

- 方式: ローカルモデルで生成する（外部 API は使わない）。Apple Silicon（Mac Studio M4 Max, 128GB）の MLX を第一候補とする
- 用途: ショート動画・CM の BGM。尺は 15〜60 秒程度、ループ可能なものを想定
- 環境: Python は uv で管理（`uv init` で作成、pip は使わない）
- モデルの重み・生成した音声は git 管理外（`.gitignore` 済み）。大容量は `ssd-archive` スキルで退避
- 方式・モデルの選定は未了。調査結果をここに追記する
