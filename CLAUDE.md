# CLAUDE.md

生成AIで音楽（Instagram・CM 用の BGM）を作るシステム。趣味の個人プロジェクトで、GitHub で公開している（moiku/music-gen）。
このマシン固有の設定（IP、パス、SSD など）は `CLAUDE.local.md` に書く。git の管理外。

- 方式: 音楽はローカルモデルで生成する。Apple Silicon（M4 Max, 128GB）の MLX を第一候補とする。
- 外部 API: 指示されない限り使わない。例外として、日本語の注文を生成の設定に置き換える処理には Claude Sonnet（`claude-sonnet-5-5`、`musicgen/interpret.py`）を使う（2026-11-07 にオーナーが指示）。API キーは `~/.secrets/api_keys.env` から読み、リポジトリには置かない。
- 用途: ショート動画・CM の BGM。尺は 15〜60 秒程度で、ループできるものを想定する。
- 環境: Python は uv で管理する（pip は使わない）。music-gen は単独の uv プロジェクトで、上位のワークスペースには入れない。
- モデルの重み、生成した音声、`vendor/`（公式リポジトリの clone）、`output/` は git の管理外。
- 公開リポジトリなので、commit に秘密情報・自宅環境の情報・所属を入れない。

## 決まったこと（2026-11-07）

- **モデル**: ACE-Step 1.5 XL-turbo（MIT、生成物の商用利用可）を主に使う。
  - Stable Audio 3 medium は、歌なしの tech 系・cinematic 系で強い。ただし Stability Community License なので、個人での登録と年商の制限がある。そのため予備扱い。
  - 経緯と根拠は `survey.md` にある。
- **ACE 本体**: `vendor/ACE-Step-1.5/.venv` で動く。重みは `vendor/ACE-Step-1.5/checkpoints` に置く（実体の場所は `CLAUDE.local.md`）。

## 使い方（ギャラリー）

```bash
uv run python -m musicgen.server --host <待ち受けるIP> --port 8890
```

- **画面でできること**: 新規作成（曲のイメージから AI が設定を自動入力）、修正版を作成、試聴、評価（★）、お気に入り、メモ、書き出し（フル・ショート・ループ・ステム・テンポ確認。M4A と WAV）、バリエーションを作成、長さを変えて作成。
- **ワーカー**: 生成が入ると ACE のワーカーが自動で起動し、10 分使わなければ終了してメモリ（約 33GB）を解放する。
- **保存先**: `output/library/`（`MUSICGEN_LIBRARY` で変更できる）。1 回の生成ごとに、音声と `meta.json`（条件・自動判定・評価）を 1 組で保存する。
- **ACE-Step の癖への対処**: `musicgen/` に組み込み済み。
  - 長めに生成して切る。
  - 格子に対する 3 区間のずれで拍を判定する。
  - ずれていれば伸縮と冒頭のトリムで合わせる。
  - ループはクロスフェードした小節コピーで作る。
- **注意**: ほかの GPU ジョブと同時に走らせると、生成が 3〜4 倍遅くなる（2026-11-07 に実測）。
