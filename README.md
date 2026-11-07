# music-gen

Instagram のストーリーズ・リールや CM に使う、歌なしの BGM を手元の Mac で作るツールです。生成には [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) を使い、ブラウザで作成・試聴・評価・書き出しを行います。

*A local BGM generator for short videos and commercials on Apple Silicon, built on ACE-Step 1.5. It has a browser gallery for creating, rating and exporting tracks. The UI and docs are in Japanese.*

## できること

- **新規作成**: 長さ、BPM、調、終わり方、プロンプトを指定して作ります。
  - 終わり方は「ループ」「きっちり終わる」「おまかせ」から選びます。
  - 長さはプリセットからも選べます（ストーリーズ 15 秒、リール 7・15・30 秒、CM 15・30 秒など）。
- **曲のイメージから自動入力**（任意）: 「カフェの朝、落ち着いたアコースティック」のような日本語から、Claude（Anthropic API）が設定を埋めます。
- **修正版を作成**: 「テンポを落として尺八を目立たせて」のような指示から、元の曲の設定を直して作り直します。
- **自動チェック**: 作った曲を次の点で判定します。
  - 拍が BPM どおりの格子に乗っているか
  - 0 秒から鳴っているか
  - 最後まで鳴っているか
  - モノラルで再生しても音が打ち消し合わないか

  テンポのずれは自動で補正します。
- **書き出し**: −14 LUFS にそろえて、次の形式で書き出します。
  - フル、ショート、ループ（M4A と WAV）
  - ステム（Demucs による分離）
  - テンポ確認用（クリック音入り）
- **ギャラリー**: 評価（★）、お気に入り、メモ、アーカイブ、派生元の記録、絞り込みができます。

## 必要なもの

- Apple Silicon の Mac。生成中に約 33GB のメモリを使い、作成していない時間が 10 分続くと解放します。
- [uv](https://docs.astral.sh/uv/)、git、ffmpeg
- ACE-Step 1.5 の重み。XL-turbo、LM、VAE などで約 30GB あります。
- 曲のイメージからの自動入力を使う場合だけ、Anthropic の API キーが必要です。

## セットアップ

```bash
git clone https://github.com/moiku/music-gen.git && cd music-gen
uv sync

# ACE-Step 1.5 本体（vendor/ は git の管理外）
git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git vendor/ACE-Step-1.5
(cd vendor/ACE-Step-1.5 && uv sync && .venv/bin/python -c "
from acestep import model_downloader as md
md.download_main_model(); md.download_submodel('acestep-v15-xl-turbo')")
```

重みは `vendor/ACE-Step-1.5/checkpoints/` に保存されます。外付けディスクに置きたい場合は、このディレクトリをシンボリックリンクにしてください。

API キーは `~/.secrets/api_keys.env` に `ANTHROPIC_API_KEY=...` と書くか、環境変数で渡します。

## 使い方

```bash
uv run python -m musicgen.server --host 127.0.0.1 --port 8890
```

ブラウザで `http://127.0.0.1:8890/` を開きます。ほかの端末から使う場合は、VPN（Tailscale など）のアドレスで待ち受けてください。このサーバーには認証がないので、インターネットに直接公開しないでください。

作った曲は `output/library/` に保存されます。保存先は `MUSICGEN_LIBRARY` で変えられます。

## 構成

| パス | 内容 |
|---|---|
| `musicgen/server.py` | Web 画面と API（FastAPI） |
| `musicgen/worker.py` | 生成ジョブを処理するワーカー（ACE-Step の venv で動く） |
| `musicgen/analysis.py` | 自動チェック、テンポ補正、長さの切りそろえ |
| `musicgen/finish.py` | 書き出し（ラウドネス、ショート、ループ、ステム） |
| `musicgen/interpret.py` | 曲のイメージから設定を作る（Claude Sonnet） |
| `survey.md` | モデル選定の調査。Suno と Lyria の比較、ライセンス、Instagram の作法、試作の記録 |
| `compare/`, `cm/` | モデルの聴き比べと、CM 曲の試作に使ったスクリプト |

## ライセンス

- このリポジトリのコードは MIT License です。
- ACE-Step 1.5 は MIT License です。公式のモデルカードには、生成した音楽を商用に使ってよいと書かれています。ただし、学習データの出所は開発元の説明によるもので、外部からは確認できません。生成した曲をどう使うかは、使う人の責任で判断してください。
- テンポ補正には [pedalboard](https://github.com/spotify/pedalboard)（GPL-3.0）を使っています。このリポジトリは pedalboard を同梱しておらず、インストールして import するだけです。
