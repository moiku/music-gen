# ローカル音楽生成の調査（2026-11-06）

## 推奨案

> **2026-11-06 更新（調査 4）**: SA3 論文の聴取評価では、歌なしの曲で SA3 medium が ACE-Step 1.5 XL を上回った。ライセンスを全文読んだ結果、個人として使う限り商用条件も満たせる見込み。そこで、**SA3 medium（audio-studio-mcp を流用）を先に試し、ACE-Step 1.5 は予備に回す**方針に変える。詳細は調査 4。以下は当初の推奨。

1. **最初に ACE-Step 1.5 を試す**。コードもウェイトも MIT で、モデルカードに「生成物を商用に使ってよい」と明記されている。出力は 48kHz ステレオ。公開モデルの中では品質が最も Suno に近く、Mac では LM を MLX、DiT を MPS で動かせる。
2. **対抗は Stable Audio 3 Small/Medium**。楽器だけの BGM ならこちらも有力。学習データはライセンス済み（AudioSparx）、MLX 版がある。ただし Community License なので、年商 $1M 以上の組織が使う場合は別ライセンスが必要。
3. MusicGen・MAGNeT（ウェイトが CC-BY-NC）、YuE2、SongGeneration（商用の条件がはっきりしない）は、CM 用途の候補から外す。
4. 「奥行き」は生成モデルだけで決まるものではないと考える（§2）。ステム分離 → リバーブ・定位の調整 → マスタリングという後処理をあわせて組む前提で進める。
5. 継ぎ目なしでループすることを機能として持つモデルはない。BPM を指定して小節の境界で切り、継ぎ目を inpainting（ACE-Step の repaint など）で作り直す方法を検証する。

> この調査は Web 調査のみで、どのモデルも手元ではまだ動かしていない。各社が自分で出したベンチマーク（「Suno v5 並み」など）は、聴き比べで確かめる必要がある。
> ライセンスについては、ACE-Step 1.5（MIT、生成物の商用利用可）と Stable Audio 3（Community License、学習データの内訳）だけ、HF のモデルカードを直接見て確認した。

---

## 調査 1：ローカルで動く音楽生成モデル

### 一覧

| モデル（公開時期） | 商用利用（コード / ウェイト） | 出力 | 最大尺 | Apple Silicon | 品質の評判 | 規模・メモリ | プロンプトの与え方 | 方式 |
|---|---|---|---|---|---|---|---|---|
| **ACE-Step 1.5**（2026-01、XL は 2026-04） | ◎ MIT / MIT | 48kHz ステレオ | 10秒〜10分 | LM は MLX、DiT は MPS（MLX 版 DiT は PR で進行中）。CUDA の 3〜10 倍遅いという報告あり | 開発元の評価では「Suno v4.5 と v5 の間」。AudioBox の CU/PQ は Suno v5 を上回る | DiT 2B（XL は 4B）＋ LM 0.6/1.7/4B。4GB 未満から動く | テキスト、歌詞、参照音声、cover、repaint、BGM 化、LoRA、BPM/キー指定 | Qwen3 系の LM で曲の構成を決め、DiT で生成。1D VAE（25Hz） |
| ACE-Step v1 3.5B（2025-05） | ◎ Apache-2.0 | 未確認 | 約4〜5分 | MPS で動く。M2 Max で RTF 2.27x（27 ステップ） | 1.5 に置き換わった | 3.5B | テキスト、歌詞、repaint、extend、remix | 拡散（DCAE）＋軽量 Transformer |
| **Stable Audio 3 Small / Medium**（2026-05） | ○ Stability Community License（年商 $1M 未満は商用可） | 44.1kHz ステレオ（サンプリングレートはモデルカードに記載なし） | Small 約2分、Medium 約6分 | MLX 版がある（mlx-community など）。M4 MacBook Pro で数秒 | 学習データはライセンス済み。歌は弱い（楽器 BGM 向け） | Small 0.6B、Medium は未確認 | テキスト、inpainting、続きの生成、audio-to-audio、LoRA | 潜在拡散（semantic-acoustic AE ＋ T5Gemma） |
| Stable Audio Open 1.0 | ○ 同上 | 44.1kHz ステレオ | 47秒 | MLX は未確認 | 効果音向き。音楽は弱く、歌は不可 | 約1B | テキスト（英語） | 潜在拡散 DiT ＋ T5-base |
| Stable Audio Open Small | ○ 同上 | 44.1kHz ステレオ | 11秒 | Arm CPU 向けの最適化あり | 短いサンプル向け | 約0.5B | テキスト | 潜在拡散 |
| MusicGen（stereo / melody） | × MIT / **CC-BY-NC 4.0** | 32kHz（ステレオ版あり） | 学習は30秒。continuation で延長 | mlx-examples に musicgen がある | 旧世代。歌は不可 | 300M〜3.3B、16GB 推奨 | テキスト、メロディ（chroma） | 自己回帰、EnCodec 4 コードブック |
| MAGNeT | × CC-BY-NC 4.0 | 32kHz | 10秒 / 30秒 | 未確認 | 旧世代 | 最大 1.5B | テキスト | 非自己回帰のマスク生成、EnCodec |
| YuE2-3B（2026-09） | △ コードは Apache / ウェイトは CC-BY-NC 4.0（README で個人クリエイターには追加の許諾あり） | 48kHz ステレオ | 5分のデモあり | 「experimental」とあるだけ | 自己申告で Suno v5 超え（best-of-8 の値） | 3.6〜4B、GPU 24GB | 歌詞＋スタイル、ABC 譜、cover | AR/NAR の混合＋flow matching |
| HeartMuLa-oss-3B（2026-01） | ◎ Apache-2.0 | 48kHz ステレオ | 既定 4分 | Mac 対応は未確認 | 歌もの寄り。7B（未公開）は Suno 並みと自己申告 | 3B | タグ、歌詞 | LM ＋ HeartCodec（12.5Hz） |
| SongGeneration / LeVo 2（2026-03） | **未確認**（商用不可とする指摘あり） | 未確認 | 4分30秒 | 未確認 | 自己申告で Suno v5 超え | 4B、22〜28GB | テキスト、タグ、10秒の参照音声、歌詞、BGM のみの出力 | LeLM ＋ 拡散 |
| DiffRhythm 2（2025-10） | ○? Apache-2.0（v1 は VAE が SAO 由来なので Stability の条件が及ぶ） | 未確認 | 曲全体 | macOS の手順は一部だけ | 中位 | 未確認 | 歌詞、参照音声（楽器のみの生成は TODO） | ブロック単位の flow matching |
| SongBloom | 未確認（ライセンス表記は "other"） | 未確認 | 長尺版あり | 未確認 | 中位 | 未確認 | 歌詞＋参照音声 | AR ＋ 拡散 |

Riffusion は旧世代で、現在は商用サービスに移っているため対象外とした。

### モデルごとの補足（出典）

**ACE-Step 1.5**
- **ライセンスと学習データ**: コードは MIT（https://github.com/ace-step/ACE-Step-1.5 ）、ウェイトも MIT（https://huggingface.co/ace-step/Ace-Step1.5 ）。学習データは「ライセンスを受けた楽曲、ロイヤリティフリー／パブリックドメインの音楽、MIDI から合成した音声」と書かれており、生成物の商用利用を許可する明記がある（モデルカードで直接確認）。ただし学習データの説明は開発元の自己申告で、外部からは検証できない。
- **仕様と評価**: 48kHz ステレオ、1D VAE（64 次元、25Hz）、DiT は約2B。AudioBox CU 8.09 / PQ 8.35 で、Suno v5（7.87 / 8.29）を上回る。一方、スタイルや歌詞の一致度は Suno のほうが上（https://arxiv.org/html/2602.00744 の Table 1）。
- **Mac での既知の問題**: MLX 版の DiT では LoRA が効かない（issue #559）。Autoscore でメモリリークが起きる（#1081）。LoRA の学習に約 45GB 必要（#282。128GB なら足りる）。
- **MLX 化の動き**:
  - 公式リポジトリに、DiT をネイティブ MLX で動かす PR #439 がある。
  - コミュニティ版: mlx-music（https://pypi.org/project/mlx-music/ ）、https://huggingface.co/thepatch/acestep-1.5-mlx 、XL Turbo の 8bit 版 https://huggingface.co/ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit 。
  - Mac での動かし方の解説記事: https://lilting.ch/articles/ace-step-ui-local-music-generation
- **XL**: ウェイトは bf16 で約 9GB、メモリは 20GB 以上が目安。

**Stable Audio 3**
- **公開状況**: 2026-05 公開。Small SFX / Small / Medium はウェイト公開、Large は非公開（https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models ）。
- **ライセンス**: Community License。年商 $1M 以上の組織は Enterprise License が必要（https://stability.ai/license ）。T5Gemma の部分には Gemma Terms of Use もかかる。HF でのダウンロードには同意が必要。
- **学習データ**: 約128万件。AudioSparx からライセンスを受けたものが 806,284 件、Freesound の CC-0 / CC-BY / CC-Sampling+ が 472,618 件（https://huggingface.co/stabilityai/stable-audio-3-small-music で直接確認）。
- **仕様**: Small は 0.6B。論文は https://arxiv.org/abs/2605.17991 。
- **MLX 版**: https://huggingface.co/mlx-community/stable-audio-3-small-music 、Medium の 8bit 版 https://huggingface.co/aufklarer/Stable-Audio-3-DiT-Medium-MLX-8bit 、nodetool-mlx（audio-to-audio と時間範囲を指定した inpainting に対応）。
- **注意**: 公式に MLX ランタイムがあるかどうかは、モデルカードからは確認できなかった。
- **CM 用途での注意**: 依頼元の企業の名義で生成する場合は、その企業の年商が条件にかかるかを確認する必要がある。

**MusicGen / MAGNeT**
- ウェイトは CC-BY-NC 4.0 で、CM には使えない（https://huggingface.co/facebook/musicgen-stereo-large 、https://huggingface.co/facebook/magnet-medium-30secs ）。
- MLX 版は https://github.com/ml-explore/mlx-examples/tree/main/musicgen 。

**YuE2**
- ウェイトは CC-BY-NC 4.0（https://huggingface.co/m-a-p/YuE2-3B ）。
- GitHub の README（https://github.com/multimodal-art-projection/YuE ）には「個人・クリエイター・ミュージシャンは生成物で収益化してよい」とある。ただし HF の LICENSE ファイルにはこの文言がない。受託の CM が対象に入るかもはっきりしない。

**HeartMuLa**
- 2026-01-20 にコードとウェイトの両方を Apache-2.0 に変更した（https://github.com/HeartMuLa/heartlib ）。
- 歌もの寄りで、楽器のみの品質と Mac での動作は未確認。

**SongGeneration / LeVo 2**
- ライセンス表記は NOASSERTION。「研究・教育用途に限る」とする指摘がある（https://github.com/vllm-project/vllm-omni/issues/3390 ）。LICENSE ファイル本体は未確認。

**DiffRhythm**
- v2 は Apache-2.0 だが、楽器のみの生成は TODO のまま（https://huggingface.co/ASLP-lab/DiffRhythm2/blob/main/README.md ）。
- v1 は VAE が Stable Audio Open 由来という注記がある（https://huggingface.co/ASLP-lab/DiffRhythm-base ）。

### ループについて

継ぎ目なしでループできることを明記したモデルは見つからなかった。以下は未検証の案。
- BPM を指定して生成し、小節の境界で切る。
- 曲の末尾と先頭をつなげ、つなぎ目の前後を ACE-Step の repaint や SA3 の時間範囲 inpainting で作り直す。
- 足りない場合は、短いクロスフェードで補う。

---

## 調査 2：Suno と Gemini（Lyria）の「奥行き」の差

前提: どちらも内部はほとんど公開されていない。以下では **事実（出典付き）** と **推測** を分けて書く。

### 2.1 出力形式とニューラルコーデック

**事実**
- **Lyria 2（Vertex AI、`lyria-002`）**: 48kHz の WAV、楽器のみ。1回あたり約 30 秒（別の公式ページには最大 32.8 秒とあり、記述が一致しない）。SynthID の透かしが入る。https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/lyria/lyria-002
- **Lyria 3 Clip / 3.5（Gemini API・AI Studio）**: 既定は MP3。WAV を選べるのは 3.5 だけ。ドキュメントには「44.1 kHz high-fidelity stereo」とあるが、MP3 のビットレートは書かれていない。https://aistudio.google.com/docs/music-generation
- **Gemini アプリの Lyria 3**: 30 秒のクリップ。https://blog.google/intl/en-ie/products/lyria-3/ 、https://gemini.google/mc/overview/music-generation/
- **Lyria RealTime**: 48kHz ステレオの 16bit PCM を返す。https://ai.google.dev/gemini-api/docs/models/lyria-realtime-exp
- **SpectroStream**: SoundStream の後継。48kHz ステレオを 4〜16kbps に圧縮する。マルチチャンネルは delayed-fusion という方式で扱う。https://arxiv.org/abs/2508.05207
- **Magenta RT / Lyria RT のトークン**: SpectroStream の RVQ は 64 段、16kbps。Magenta RT が生成するのは**上位 16 段（4kbps）だけ**。Lyria RT は残りの 48 段を予測するモデルを足しており、その効果を論文は「a slight bump in audio fidelity」と書いている。https://arxiv.org/html/2508.04651v3
- **Suno**: MP3 は全プランで、WAV は Pro/Premier でダウンロードできる。サンプリングレートとビットレートは公式に書かれていない。https://help.suno.com/en/articles/13926081
- **参考（他のコーデック）**:
  - MusicGen は 32kHz モノラルの EnCodec を使う。ステレオ版は左右を別々に符号化し、コードブックを交互に並べる。https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md
  - Stable Audio Open の VAE は 44.1kHz ステレオを連続潜在表現（21.5Hz、64ch）にする。https://arxiv.org/abs/2407.14358
  - 左右を独立に扱うコーデックは空間の再現が悪いという報告がある（音声が対象）。https://arxiv.org/abs/2309.07432

**推測**
- RVQ では、細部を担う下位の段に、残響の尾、空気感、高域、左右の差（サイド成分）が入りやすい。生成する段を減らすと、これらが先に痩せ、「主旋律だけが前に出る」質感になりうる。
- ただし Lyria 3 / 3.5 は連続潜在の拡散モデル（§2.2）なので、この説明が直接当てはまるのは Lyria RT 系だけ。
- Gemini アプリが返す MP3 では、ジョイントステレオの圧縮でサイド成分の高域がさらに削られている可能性がある。ビットレートが分からないので、どの程度かは判断できない。

### 2.2 生成の構造

**事実**
- **Lyria 3 / 3.5**: モデルカードに「utilizes latent diffusion, applied to temporal audio latents」とある。拡散方式で、ステム別に生成しているかどうかは書かれていない。https://deepmind.google/models/model-cards/lyria-3/
- **Lyria RT / Magenta RT**: コーデックのトークンを予測する言語モデル（エンコーダ・デコーダ型 Transformer）。2 秒ずつ生成し、直前 10 秒を参照する。ステムの on/off、brightness、density などの操作ができる。
- **Google の系譜**: MusicLM は SoundStream（24kHz）のトークンを使う階層型の自己回帰モデルだった。https://arxiv.org/abs/2301.11325
- **Suno の方式**: CEO はインタビューで、音を離散トークンにして Transformer で次のトークンを予測する方式だと述べている。https://latent.space/p/suno 。v4.5 以降の構造は公開されていない。
- **Suno Studio**（2025-09）: マルチトラック編集、ステムの再生成（最大 12 ステム）、MIDI 書き出しができる。https://en.wikipedia.org/wiki/Suno_(platform)

**推測**
- Suno が内部でステムごとに生成しているという公式情報はない。ただし後述の独立分析では、ステムを作り直してからミックスすると左右の幅が広がっていた。ステム単位で生成する仕組みがあることを間接的に示している可能性がある。

### 2.3 学習データとミックスの傾向

**事実**
- **Suno の学習データ**: 2024-08 の答弁書で、「数千万の録音」で学習し、原告であるレーベルの楽曲も「presumably」含まれると認めた。https://techcrunch.com/2024/08/01/ai-music-startup-suno-response-riaa-lawsuit
- **Suno と Warner**: 2025-11 に和解し、2026 年にライセンスを受けたモデルへ移行すると報じられた。https://www.capitalfm.co.ke/business/2025/11/warner-settles-lawsuit-with-ai-music-firm-and-launches-joint-venture/
- **Magenta RT の学習データ**: 「約19万時間、主に楽器のみのストック音楽」。https://arxiv.org/html/2508.04651v3
- **Lyria 3 の学習データ**: モデルカードには出所が書かれていない。
- **Suno v5 のステレオを測った独立分析**（第三者、査読なし、2026-03）: https://suno-stereo-analysis.pages.dev/en
  - Suno v5 の 3,571 曲を M/S 解析した。
  - サイド成分とミッド成分の比の中央値は約 −10.8dB。比較した一般の DAW 制作曲（2 例）より **5〜12dB 狭い**。
  - サイド成分の高域も減衰している。
  - ステムを作り直してからミックスすると、約 8dB 広がる。
- **Suno v5.5**: 「左右の幅がさらに縮んだ」という第三者の記事があるが、未確認。https://masterforge-app.pages.dev/blog/suno-v5-vs-v5-5/

**推測**
- 上の測定が正しければ、Suno の「奥行き」は**左右の広がりではなく、前後方向の手がかり**から来ている可能性が高い。前後方向の手がかりとは、センターにも乗る残響の量、楽器ごとの帯域の住み分け、音量差とダイナミクス、トランジェントの処理などを指す。
- 商用の楽曲を数千万曲学習したことで、市販のミックスやマスタリングの作法がモデルに取り込まれたと考えるのが自然である。
- Google 側で学習データが分かっているのは Magenta RT だけで、「主に楽器のみのストック BGM」である。ストック BGM はナレーションの下に敷く前提で作られ、残響が控えめでドライ、主旋律が前に出るものが多い。Lyria 3 も同じ傾向のデータで学習していれば、聴いた印象とよく合う。ただし Lyria 3 のデータの中身は公開されていない。

### 2.4 後処理（マスタリング・アップミックス）

**事実**
- Suno の「Remaster」は、既存の曲を新しいモデルで生成し直す機能（https://help.suno.com/en/articles/3591617 ）。生成後に決まったマスタリングをかけているという公式の記述はない。
- Google が公表している後処理は、SynthID の埋め込みと安全フィルタだけ。

**推測**
- 独立分析では、同じプロンプトでも左右の幅が中央値で 1.77dB ばらつく。決まった処理をかけているなら幅はそろうはずなので、固定のマスタリングチェーンよりも、生成そのもののばらつきと考えるほうが自然。どちらの会社についても、明示的な後処理の証拠は見つからなかった。

### 2.5 ローカルで同じような奥行きを出す手段

**ツール（ライセンスは GitHub で確認）**

| 用途 | ツール | ライセンス |
|---|---|---|
| ステム分離 | Demucs（htdemucs）https://github.com/adefossez/demucs | MIT |
| ステム分離（RoFormer 系） | python-audio-separator https://github.com/nomadkaraoke/python-audio-separator | MIT |
| ステム分離（RoFormer 系の学習・推論） | Music-Source-Separation-Training https://github.com/ZFTurbo/Music-Source-Separation-Training | MIT（ウェイトはモデルごとに確認） |
| リバーブ、畳み込みリバーブ、EQ、コンプ、リミッター、VST3 の読み込み | pedalboard（Spotify）https://github.com/spotify/pedalboard | GPL-3.0 |
| 参照曲に合わせる自動マスタリング | Matchering https://github.com/sergree/matchering | GPL-3.0 |
| ラウドネスの測定と正規化 | pyloudnorm https://github.com/csteinmetz1/pyloudnorm | MIT |
| 帯域拡張（高域の補完） | AudioSR https://github.com/haoheliu/versatile_audio_super_resolution | MIT |
| 帯域拡張（高速版） | FlashSR https://arxiv.org/abs/2501.10807 | 未確認 |

GPL のツールは、自分で使うだけなら問題ない。生成物（音声）に GPL はかからない。

**処理チェーンの案（推測。§2.3 の「前後方向の奥行き」の仮説にもとづく）**

1. **ステム分離**: RoFormer か htdemucs で、vocals / drums / bass / other に分ける。
2. **ステムごとの空間づくり**（pedalboard の Reverb、または Convolution ＋ホールの IR）:
   - 主旋律: 少し下げ、プレディレイ付きのリバーブで奥に置く。
   - パッドや other: パンを振り、Haas 効果（10〜25ms）か M/S 処理でサイド成分を足す。
   - ベースとキック: モノラルのまま残す。
3. **バス処理**: 全体に短いリバーブを薄くかけ、同じ空間にまとめる。
4. **（任意）帯域拡張**: 高域が足りなければ、ミックスの前に各ステムへ AudioSR をかける。
5. **マスタリング**: Matchering で参照曲（市販曲や Suno の出力）に合わせる。続けて pyloudnorm で約 −14 LUFS にそろえ、リミッターで −1 dBTP に抑える。Instagram の公式のラウドネス目標は見つからなかったので、この値は第三者の資料での通例。
6. **確認**: モノラルにして聴いても崩れないか、位相の相関を見る。スマホのスピーカーは実質モノラルなので、ここが崩れると左右を広げた効果が逆効果になる。

**検証の進め方（実装フェーズで行う）**
- 同じプロンプトで生成したものを、後処理なしと後処理ありで比べる。
- Suno の出力と M/S 比、残響の長さ（RT60 の推定）を並べて比べ、奥行きの原因が左右の幅なのか前後の手がかりなのかを確かめる。

---

## 未確認・要確認の項目
- ~~Stable Audio 3 Medium のパラメータ数。SA3 の公式の MLX ランタイムの有無。~~ → 調査 3 で確認済み（Medium は 1.4B、公式 MLX ランタイムあり）。
- SongGeneration / LeVo 2、SongBloom の LICENSE ファイル本体。
- YuE2 の追加許諾が、受託の CM 制作にも当てはまるか。
- HeartMuLa の Mac での動作と、楽器のみの曲の品質。
- Suno の出力のサンプリングレートとビットレート。Gemini アプリの MP3 のビットレート。
- 独立分析（Suno のステレオ幅）は査読されていない。結論に使う前に、手元で同じ解析をして確かめる。

---

## 調査 3：候補 2 モデルで曲を作る既存のスキル・システム（GitHub 中心、2026-11-06）

候補は ACE-Step 1.5 と Stable Audio 3（SA3）に絞った。★（スター数）・最終 push 日・ライセンスは `gh api` で取得した値。どれも手元では動かしていない。Apple Silicon への対応は README に書かれている内容をそのまま載せた。

### 結論

- **生成の土台は、どちらのモデルも公式のものを使えば足りる**。
  - ACE-Step 1.5: 公式リポジトリの REST API サーバを使う。macOS 用の起動スクリプトが、MLX で動く DiT と VAE（PR #439 で main に取り込み済み）を自動で選ぶ。
  - SA3: 公式リポジトリの `optimized/mlx` を使う。PyTorch なしの MLX だけで動き、CLI、Gradio、inpaint、LoRA 学習までそろっている。
  - 第三者が MLX に移植した版は、追いかける必要がない。
- **構想にいちばん近い既製品は audio-studio-mcp**（MIT、★595）。SA3 の公式 MLX 実装を取り込んだ MCP サーバ兼 Web UI で、次の流れを一通り持つ。
  - 生成（候補 1〜4 本、BPM 指定）
  - 編集（inpaint／続きの生成）
  - ループ化
  - 2 パスの loudnorm による LUFS 合わせ
  - 書き出し
  
  これを fork して ACE-Step のバックエンドを足すのが最短の道筋。
- **ローカルのモデルで「生成 → 選別 → ステム分離 → ミックス → マスタリング → 尺合わせ」を最後まで通す、成熟した既製品は見つからなかった**。次の 4 つは自作になる。
  - 拍に合わせたループ化
  - CM の尺（15／30／60 秒）への合わせ込み
  - 候補の自動選別
  - 奥行きを出す後処理（§2.5）

### A. ACE-Step 1.5 の周辺

| ツール | ★ / 最終 push | ライセンス | 内容 | Apple Silicon | 評価 |
|---|---|---|---|---|---|
| **公式 ace-step/ACE-Step-1.5** https://github.com/ace-step/ACE-Step-1.5 | 14.4k / 2026-10 | MIT | Gradio UI、REST API（非同期タスク）、CLI、Python API、LoRA 学習 | ◎ `start_api_server_macos.sh` が MLX バックエンドを自動で選ぶ | **土台にする** |
| 公式に同梱された Claude Code スキル（`.claude/skills/`） | 本体に同梱 | MIT | `acestep`（API を呼ぶ：generate / cover / repaint / health）、`acestep-songwriting`（caption、構造タグ、BPM・キー・尺の決め方）、`acestep-docs` など | API 経由なので環境を問わない | **流用する**。歌もの寄りなので、BGM 向けの指示を書き足す |
| ServeurpersoCom/acestep.cpp | 456 / 2026-11 | MIT | GGML による C++ 実装。サーバとブラウザ UI。全タスク、LoRA、量子化に対応 | ○ Metal | Python なしで軽く動かす選択肢。公式と同じ出力になるかは未確認 |
| Majiks-Studio/majiks-music-studio | 16 / 2026-10 | Apache-2.0 | macOS 向け DAW（Swift）。ACE-Step（MLX）で生成し、小節にそろえたステムに分けて、ミックスとマスタリングまでできる。MCP あり | ◎ | 手で詰める作業をするときの選択肢。組み込むには重い |
| AgriciDaniel/claude-music | 90 / 2026-08 | MIT | Claude Code スキル。generate / repaint / export（SNS 別のラウドネス）/ analyze / enhance | △ CUDA 前提 | コマンド構成を参考にする |
| fspecii/ace-step-ui | 5.2k / 2026-06 | **LICENSE ファイルなし** | Suno 風の UI。Demucs によるステム抽出あり | △ MPS | 手作業向け。ライセンスが不明なので、CM 用のパイプラインには入れない |
| ComfyUI（ACE-Step 1.5 にネイティブ対応）、ace-step/ACE-Step-ComfyUI | 85 / 2026-03 | MIT | 公式のワークフロー | MPS（未確認） | GUI で試作するときだけ使う |
| koda-dernet/Side-Step | 163 / 2026-09 | **CC BY-NC-SA 4.0** | LoRA の学習ツール | — | **非商用ライセンスなので使わない** |
| NON906/ACE-Step-MCP、xDarkzx/SongForge-MCP | 0〜13 | Apache-2.0 | MCP サーバ | 未確認 | 未成熟 |
| ace-step/awesome-ace-step | 166 / 2026-05 | CC0 | 公式のリンク集 | — | 情報源として使う |

**BGM・ループに関わる仕様（公式）**

- **歌なしにする**: `instrumental=True` を指定する。
- **曲の条件**: `bpm`（30〜300）、`key_scale`、`audio_duration`（10〜600 秒）、`batch_size`（最大 8）はパラメータで指定する。caption には書かないのが公式の推奨。
- **ループ専用の機能はない**。継ぎ目を再生成するには、repaint（`chunk_mask_mode="explicit"`）で自作する必要がある。
- **延長とステム抽出**: `complete` と `extract` の task_type で行う。ただし base モデルでしか使えない。

### B. Stable Audio 3 の周辺

| ツール | ★ / 最終 push | ライセンス | 内容 | Apple Silicon | 評価 |
|---|---|---|---|---|---|
| **公式 Stability-AI/stable-audio-3** の `optimized/mlx` https://github.com/Stability-AI/stable-audio-3 | 847 / 2026-11 | コードは MIT、重みは Community License | `./sa3` CLI、`./sa3-gradio`。small-music、medium、sfx に対応。inpaint（`--inpaint-range`）、audio-to-audio、LoRA の重ね掛けと学習 | ◎ PyTorch 不要。M4 Pro で medium の 30 秒を約 5 秒、約 3.9GB で生成 | **土台にする** |
| 公式の HF 重み（small-music 433M・最長 120 秒、medium 1.4B・最長 380 秒） | — | Community License（ダウンロードに同意が必要） | MLX 用は optimized 版 | ○ | 使う |
| **kanaworksai-hue/audio-studio-mcp** https://github.com/kanaworksai-hue/audio-studio-mcp | 595 / 2026-10-24 | MIT | MCP と Web UI。generate／edit（variation・inpaint・continue）／process（トリム → ループ化 → 2 パス loudnorm → フェード → WAV/OGG/FLAC）。生成の条件を JSON の sidecar に残す | ◎ 公式 MLX 実装をコミット固定で取り込んでいる | **骨格として流用する**。ループ化は等電力クロスフェードだけで、拍は検出していない。ステム分離とマスタリングもない |
| dada-bots/underfit | 108 / 2026-10 | MIT | LoRA 学習のダッシュボード（Mac では公式の MLX トレーナーを使う） | ○ | 音のトーンを統一したくなったら使う |
| mlx-community/stable-audio-3-small-music | — | — | mlx-audio で変換したもの。今の mlx-audio に SA3 の実装が見当たらない | — | **使わない** |
| AIEGOBOT/agent-audio、avalonreset/legends-stable-audio-3 | 5〜51 / 2026-11 | MIT / Apache | MCP やエージェント用の CLI | 未確認 | プレビュー段階。参考にとどめる |
| MAz-Codes/Fragmenta | 68 | **AGPL-3.0** | デスクトップアプリ | ○ | 組み込みには使わない |

### C. モデルに依存しない部品（後処理・ループ・スキル）

| 用途 | ツール | ★ | ライセンス | 評価 |
|---|---|---|---|---|
| ループ点の探索 | arkrow/PyMusicLooper | 415 | MIT | **第一候補**（拍とスペクトルの類似度で探す。CLI あり） |
| 拍・ダウンビートの検出 | CPJKU/beat_this | 467 | MIT | 小節の頭で切るのに使う（MPS で動くかは未確認） |
| ステム分離 | ssmall256/demucs-mlx、andrade0/demucs-mlx | 21〜45 | MIT | MLX 版の HTDemucs。M4 Max で実時間の約 30 倍速という記載がある |
| ステム分離（品質重視） | nomadkaraoke/python-audio-separator | 1.5k | MIT | RoFormer 系を使う |
| エフェクト・マスタリング | spotify/pedalboard | 6.5k | GPL-3.0 | チェーンの本命 |
| 参照曲に合わせるマスタリング | sergree/matchering | 2.7k | GPL-3.0 | CLI として外から呼ぶ |
| ラウドネス | csteinmetz1/pyloudnorm（＋ ffmpeg の loudnorm） | 790 | MIT | 必須 |
| スキル設計・マスタリングの知識 | bitwize-music-studio/claude-ai-music-skills | 631 | CC0 | Suno 前提だが、mastering-engineer スキルとプラットフォーム別の LUFS 表を流用できる |
| モデルを横断する Suno 代替 | HamSanders/soundbox | 141 | 表記が一致しない | SA3、ACE-Step、Demucs、MCP を扱う。「Mac では未検証」と明記されている。参考にとどめる |

GPL-3.0（pedalboard、Matchering）は、自分で使う限り問題ない。生成した音声にも GPL はかからない。

### 構成案（実装フェーズへの引き継ぎ）

```
Claude Code スキル（SKILL.md：BGM の要件 → プロンプト・BPM・キー・尺を決める）
  └─ 自作のオーケストレータ（audio-studio-mcp を fork）
       ├─ backend: SA3 公式 MLX（取り込み済み）
       ├─ backend: ACE-Step 1.5 公式 API サーバ（新しく追加）
       ├─ 候補の生成（batch）→ 自動選別（拍の安定度・継ぎ目の誤差・ラウドネス）
       ├─ ループ化：beat_this で小節の頭を検出 → N 小節で切る → 継ぎ目を inpaint／repaint で再生成し、仕上げにクロスフェード
       ├─ 奥行きを出す処理：demucs-mlx → ステムごとのリバーブ・定位（pedalboard）→ バス処理
       └─ マスタリング・書き出し：pedalboard ／ Matchering → pyloudnorm → 用途別のプリセット
```

**書き出しのラウドネス目標**

| 用途 | 目標 | 根拠 |
|---|---|---|
| Instagram | −14 LUFS、TP −1 dB | 第三者の資料での通例 |
| 日本の TV CM | −24 LKFS | ARIB TR-B32 |

audio-studio-mcp の表にある −23 は EBU の基準なので、日本の TV CM にはそのまま使わない。

### 未確認（調査 3）

- ComfyUI、soundbox、beat_this が Mac で実際に動くか。
- acestep.cpp の出力が公式と同じになるか。
- SA3 の Community License で、受託の CM 制作のときに誰の年商で判定するか。原文で確認する。

---

## 調査 4：Stable Audio 3 の品質とライセンス（全文を読んだ結果、2026-11-06）

### 4.1 品質：SA3 と ACE-Step 1.5 の比較

**事実**（SA3 論文の Table 3。https://arxiv.org/html/2605.17991v1 ）
- 条件: 歌なしの曲を 120 秒生成する。プロンプトは 424 個。聴取評価の参加者は 14 名。評価は 5 段階で、OVL は全体の品質、REL はプロンプトとの一致度、MUS は音楽性。

| モデル | FAD ↓ | CLAP ↑ | OVL ↑ | REL ↑ | MUS ↑ |
|---|---|---|---|---|---|
| ACE-Step 1.5 XL-turbo | 0.193 | 0.321 | 3.35 | 3.30 | 3.15 |
| SA3 small-music | 0.145 | 0.393 | 3.20 | 3.60 | 3.15 |
| **SA3 medium** | **0.107** | 0.390 | **4.20** | 4.25 | 4.15 |
| SA3 large（公開されていない） | 0.101 | 0.393 | 3.95 | 3.80 | 4.30 |

- SA3 は歌なしの曲に特化している。論文は、歌詞の扱いや指示による編集は対象外だと書いている。
- small の品質は ACE-Step 1.5 と同じくらいで、**medium ははっきり上**だった。

**注意**
- この評価は Stability 自身が行ったもので、聴き手は 14 名と少ない。
- ACE-Step の論文は、自分たちの評価で「Suno v5 並み」と主張している（調査 1）。
- 結論を出すには、同じプロンプトで生成して自分の耳で聴き比べる必要がある。

**判断**: 歌なしの BGM なら、**SA3 medium が第一候補**。歌が必要な場合や、ライセンスの条件（下記）を満たせない場合に ACE-Step 1.5 を使う。

### 4.2 SA3 のライセンス：全文を読んで分かったこと

**対象にした本文**
- モデルの重み: Stability AI Community License Agreement（Last Updated: July 5, 2024）。HF の `stabilityai/stable-audio-3-medium/LICENSE.md` を全文読んだ。small-music と optimized も同じライセンス。
- コード: https://github.com/Stability-AI/stable-audio-3 で、MIT（2026 Stability AI）。
- 関係するその他の規約: Gemma Terms of Use（https://ai.google.dev/gemma/terms 。T5Gemma 部分に適用）と、Stability の AUP（https://stability.ai/use-policy ）。

**条文の要点（原文にもとづく）**

| 論点 | 条文の内容 |
|---|---|
| 非商用 | 趣味・評価・試験は「Non-Commercial Purpose」にあたり、無償で使える。登録は不要（§2） |
| 商用の定義 | 「commercial advantage or monetary compensation to You or others」を主な目的とする利用。自分の製品やサービスを作る・配布することや、業務上の利用を含む（§3） |
| **商用時の登録** | 商用に使う場合は https://stability.ai/community-license での**登録が必須**（§3） |
| **年商の上限** | You または Affiliate（議決権の 50% 超で支配関係にある組織）の年間収益が、単独でも合計でも **US$1,000,000 を超えた時点で、ライセンスは自動的に終了する**。その収益が SA3 から生じたかどうかは問わない。超えた場合は Enterprise ライセンスを申請する（Stability の裁量で許可される）（§3） |
| **「You」の範囲** | 会社や組織のために使う場合、「You」には**その組織が含まれる**（§1） |
| 生成物の権利 | Stability との関係では、生成物は利用者のものになる。ただし法律が許す範囲で（§4c-iii）。生成物は Derivative Work（派生物）にあたらない（§5） |
| 生成物の使い道の制限 | 生成物を、ほかの**基盤となる生成モデル**の作成や改良に使ってはいけない。SA3 自体の LoRA は例外（§4b） |
| 表示義務 | 「モデルや派生物」または「それを使った製品・サービス」を第三者に提供する場合は、本文の添付、Notice ファイル、「**Powered by Stability AI**」の目立つ表示が必要（§4a） |
| 利用ポリシー（AUP） | 実在の人物になりすますことの禁止。AI の生成物を「人間が作った」と偽ることの禁止。他人の知的財産権の侵害の禁止など。AUP は更新されることがある |
| 終了 | 違反した場合、またはライセンサーを権利侵害で訴えた場合は終了し、モデルを削除する義務を負う。準拠法は米国・カリフォルニア州法 |
| Gemma の規約 | 収益の上限はない。Google は生成物に権利を主張しない。モデルを再配布しない限り、追加の義務はほぼない |

### 4.3 このプロジェクトに当てはめると（解釈。法的な助言ではない）

1. **個人として使う形に限定する**。
   - 勤務先など、年間収益が $1M を超える組織の業務として使うと、「You」にその組織が含まれ、**Community License の対象外になる**。
   - 組織の費用で用意した端末や、組織名義のアカウントで使うことも避ける。
2. **個人の年間収益が $1M 未満なら、商用で使える**。ただし、CM や収益化した Instagram に使う前に、**商用として登録しておく**（無償）。
3. **受託で CM の BGM を作る場合**
   - 依頼元に音声ファイル（生成物）だけを渡すのであれば、ライセンスを受けているのは自分なので、判定に使う年商は自分の年商だと読める。
   - 依頼元の企業が自分でこのシステムを動かす場合は、その企業の年商で判定される。
4. **「Powered by Stability AI」の表示**
   - 受託制作は「SA3 を使ったサービス」にあたる可能性がある。
   - 安全のため、制作者のサイトやプロフィールに表示しておく（手間はほとんどかからない）。
   - 自分の Instagram 投稿だけなら、表示義務は不要と読めるが、断定はできない。
5. **プロンプトに実在のアーティスト名を入れない**。AUP のなりすまし禁止に触れるおそれがあり、CM では権利上のリスクにもなる。
6. **日本法では、AI の生成物には著作権が生じない場合がある**。「生成物は利用者のもの」という条項は、Stability との間の取り決めにすぎない。CM の依頼元に「権利の保証」を求められた場合は、契約で扱いを決めておく。

**ACE-Step 1.5（MIT）との比較**: ACE-Step 1.5 は、登録、年商の上限、表示義務がいずれもない。**年商の大きい組織の業務として使う場合や、そうした法人に渡して使ってもらう場合は、ACE-Step 1.5 しか選べない**。

### 4.4 工数への影響

SA3 だけで済めば、audio-studio-mcp（SA3 の公式 MLX 実装を同梱）をほぼそのまま土台にできる。ACE-Step のバックエンドを追加する作業がなくなる。自作が必要なのは次の 3 つ。
- 拍に合わせたループ化
- CM の尺合わせ
- 奥行きを出す後処理と、用途別の書き出し

ACE-Step は、のちに「2 つ目のバックエンド」として追加できるよう、インターフェースだけ揃えておけば足りる。

### 4.5 ACE-Step 1.5 のモデルの種類と、ACE-Step 側の評価（追記）

**モデルの種類**（README と HF のモデルカードより。https://github.com/ace-step/ACE-Step-1.5 ）

| 種類 | DiT の規模 | ステップ数 | CFG | 使えるタスク | README の評価（品質／多様性） |
|---|---|---|---|---|---|
| base / xl-base | 2B / 4B | 50 | あり | すべて（repaint、extract、lego、complete を含む） | 中・高／高 |
| sft / xl-sft | 2B / 4B | 50 | あり | repaint まで | 高・非常に高／中 |
| turbo / xl-turbo | 2B / 4B | **8**（蒸留） | なし | repaint まで | 非常に高／中 |

- **turbo と sft の違い**: turbo は sft を蒸留して 8 ステップで生成できるようにした速度重視版で、CFG（プロンプトへの従い具合を強める仕組み）を使えない。種類ごとの品質を数値で比べた資料は公開されていない。
- **XL と 2B の違い**: XL は DiT を 4B に大きくしたもの。「2B より音が豊か」という定性的な説明しかない。
- **24GB 以上ある場合の推奨**: README は「XL-sft と 4B の LM」の組み合わせを推奨している。

**ACE-Step 側の評価**（技術報告 Table 1。https://arxiv.org/html/2602.00744 ）
- **評価したモデル**: 「1.7B の LM と、蒸留した DiT（約 2B）」の組み合わせ。XL は含まれていない。
- **評価の条件**: Gemini 3 Pro で作ったプロンプトと歌詞を使っており、**歌ものの評価**である。歌なしの曲だけの評価はない。
- **結果**: Suno v5 と並べると次のとおり（AudioBox の CE・CU・PC・PQ と、SongEval の Coherence・Musicality は 10 段階または 5 段階のスコア）。

| モデル | CE | CU | PC | PQ | Coherence | Musicality | スタイル一致 | 歌詞一致 |
|---|---|---|---|---|---|---|---|---|
| ACE-Step 1.5 | 7.42 | 8.09 | 6.47 | 8.35 | 4.72 | 4.67 | 39.1 | 26.3 |
| Suno v5 | 7.69 | 7.87 | 6.51 | 8.29 | 4.72 | 4.62 | 46.8 | 34.2 |
| MiniMax 2.0 | 7.71 | 7.95 | 6.42 | 8.38 | 4.61 | 4.51 | 43.1 | 29.5 |

- **聴取評価**: ブラインドの A/B テストで「Suno v4.5 と v5 の間」。数値は公表されていない。

**SA3 側の評価との違い**

| | SA3 論文 | ACE-Step 技術報告 |
|---|---|---|
| ACE-Step のモデル | XL-turbo（8 ステップ） | 2B の蒸留版 |
| 対象 | 歌なしの曲 120 秒 | 歌もの |
| 比較相手 | SA3 | 商用サービス（Suno、Udio、MiniMax など） |

- 2 つの評価は条件がずれていて、互いの主張は矛盾しない。
- 歌なしの曲で両方を直接比べた資料は SA3 論文だけで、そこでは SA3 medium が上だった。
- **xl-sft（50 ステップ、CFG あり）は、どちらの評価にも入っていない**。聴き比べでは、SA3 medium・ACE xl-turbo・ACE xl-sft の 3 つを比べる。

---

## 調査 5：手元での試走（2026-11-06、M4 Max 128GB）

- **条件**: 歌なしの BGM 用プロンプト 6 種 × seed 2 種、各 30 秒。プロンプトは `compare/prompts.json`。
  - SA3 には BPM とキーを指定するパラメータがないので、プロンプトの文中に書いた。
  - ACE には `bpm`・`keyscale`・`instrumental=True` を渡した。
- **認証**: Hugging Face の認証は不要だった。SA3 の MLX 用の重みは `stabilityai/stable-audio-3-optimized` にあり、ダウンロードに同意が要らない。
- **重みの置き場所**: SSD に置いた。SA3 は `HF_HOME`、ACE は `vendor/ACE-Step-1.5/checkpoints` から SSD へのシンボリックリンク。

| モデル | 1 本あたりの時間（30 秒の曲） | 出力 | side/mid 平均 | L/R 相関 | 12kHz 以上のエネルギー |
|---|---|---|---|---|---|
| SA3 medium（MLX） | **約 8.7 秒**（CLI を毎回起動し、モデルの読み込みも含む） | 44.1kHz ステレオ | −9.4 dB | 0.64 | −36.5 dB |
| ACE xl-turbo（MLX、LM 1.7B） | 約 40 秒（読み込み 64 秒は別） | 48kHz ステレオ | −7.8 dB | 0.66 | −34.2 dB |
| ACE xl-sft（MLX、50 ステップ） | 約 113 秒（読み込み 58 秒は別） | 48kHz ステレオ | −6.5 dB | 0.58 | −34.0 dB |

**表から読み取れること**
- **速度**: SA3 は ACE turbo の約 5 倍、ACE sft の約 13 倍速い。
- **左右の広がり**: ACE の 2 つのほうが SA3 より広い。side/mid 比は ACE が −6.5〜−7.8 dB、SA3 が −9.4 dB。
- **SA3 の cinematic（2 本とも）**: L/R 相関がマイナス（−0.15、−0.20）で、side 成分が mid 成分を上回っていた。逆相の成分が多いので、モノラルで再生すると音が打ち消し合うおそれがある。スマホのスピーカーで流すときは要注意。

**聴き比べページ**: `output/compare/blind/index.html`
- 全クリップの音量を −14 LUFS にそろえた（ピークリミッタで、ピークは −1 dBFS 以下）。
- ピークが突出していたクリップは、そろえきれず −16 LUFS 前後まで残った。
- 判断はこのページでの聴き比べで行う。

### 5.1 聴き比べの結果（2026-11-07、評価者 1 名、ブラインド、各セル n=2）

評価の生データは `output/compare/ratings_2026-11-07.csv`。

| ジャンル | SA3 medium | ACE xl-turbo | ACE xl-sft |
|---|---|---|---|
| cinematic | 4.5 | 4.5 | 3.0 |
| corporate | 3.0 | 4.5 | 4.5 |
| folk | 3.5 | 3.5 | 2.5 |
| lofi | 3.0 | 4.0 | 3.5 |
| tech | **5.0** | 4.5 | 4.5 |
| tropical | 4.5 | **5.0** | 3.0 |
| **全体** | 3.92 | **4.33** | 3.50 |

**分かったこと**
- **ACE xl-turbo が最も安定していた**。全体の平均が最高で、どのジャンルでも 3.5 を下回らなかった。
- **SA3 はジャンルによって差が大きかった**。
  - 良かったジャンル: tech（5.0）、cinematic、tropical。シンセの質感や空間の広がりが主役になる曲。メモには「厚み」「広がり」とあった。
  - 弱かったジャンル: corporate と lofi（3.0）。メモには「音が軽い」「低音が軽い」とあった。
- **ACE xl-sft は予想に反して最下位だった**。メモの「尖り」はほとんどが sft のクリップについてのものだった。
  - 測定すると、sft は 150Hz 以下の低域が他の 2 つより約 5dB 少なく、「音が軽い」という評価と一致した。
  - CFG（既定値 7.0）が強すぎる可能性がある。

**音響の指標と点数の関係**（ジャンルごとの平均を引いてから、Spearman の順位相関を計算）

| 指標 | 相関係数 |
|---|---|
| 8〜16kHz のエネルギー | −0.32 |
| 2.5〜6kHz のエネルギー | −0.16 |
| side/mid 比 | +0.14 |

- 高域が多いクリップほど点が低い、という弱い傾向がある。「尖り」というメモと方向は合う。
- 左右の広がりと点数には、ほとんど関係が見られなかった。

**注意点**
- 評価者は 1 人で、各セルは 2 本だけである。0.5 点の差は「1 本で 1 点違った」ことにすぎない。
- ジャンルによる差は傾向として読むにとどめる。

---

## 調査 6：Egatari の CM 曲の試作で分かったこと（2026-11-07、ACE-Step 1.5 XL-turbo）

注文書は `egatari/handoff/cm-music-brief-2026-11-06.md`。条件は 96 BPM、4/4 拍子、15.0 秒、6 小節、0 秒から鳴り始め、ぴたっと終わる、の 6 点。

**仕組み**（`cm/` 以下）
- `gen_candidates.py`: 候補を一括で生成する。
- `score.py`, `grid_check.py`: 注文書の条件を満たしているかを自動で測る。
- `finish.py`: 仕上げを行う。テンポの補正、拍の位置合わせ、−14 LUFS、6 秒版、ステム、クリック付きの確認用音源を作る。
- `make_loop_xfade.py`: 繰り返し再生版を作る。

**メモリ**: PyTorch 側の DiT を解放すると、プロセスのメモリ使用量は 43GB から 33GB に減った（`cm/ace_engine.py`）。MLX 側の DiT を bf16 に変換する方法では減らなかった。

**ACE-Step の癖（今回の実測）**

| 癖 | 実測 | 対処 |
|---|---|---|
| 指定した長さより早く曲が終わる | 15 秒を指定すると約 12.4 秒で鳴り止む（5 小節目の終わり） | 長めに生成して切る |
| 長さを伸ばすとテンポが変わることがある | 17.5 秒を指定すると 97 BPM になった | Rubber Band で 96 BPM に伸縮する |
| 拍の位置が格子からずれる | 12 本中 6 本が約半拍（280 ms）ずれていた。そろっていたものも 20〜35 ms 遅れていた | 半拍ずれは採点で落とす。小さな遅れは冒頭を詰めて合わせる |
| テンポの推定が外れることがある | 12 本中 1 本が 129 BPM | 採点で落とす |
| repaint ではループを作れなかった | 次に来る「1 小節目」の直前で音を止めてしまう（3 回試して 3 回とも） | 小節をコピーし、クロスフェードでつなぐ方法に切り替えた |

- `librosa` によるテンポの推定値（刻みが粗い）と、拍への直線あてはめは、どちらも不安定だった。拍の判定は、96 BPM の格子に対する 5 秒ごとのずれ（`grid_check.py`）で行うのが確実だった。

**本命の候補 plain_s1005 の実測値**
- 格子とのずれ: −4 / −2 / +2 ms（区間 0–5 / 5–10 / 10–15 s）
- 鳴り終わり: 14.0 秒。余韻も 15 秒の中に収まる。
- ループ: 継ぎ目の音量差は 3.6 dB。
- 歌: vocals ステムは −64 dBFS で、歌は入っていない。

**注文書を満たせなかった点**
- ステムは Demucs の drums / bass / other に分けた。注文書の「決め・リズム・その他」には分けられない。決めの音はリズムと同じ drums ステムに入っている。
- 6 秒版は 1 小節目と 6 小節目をつないだもので、5.0 秒になる。
- 3 拍目の「決め」の強さを数値で見ると、plain_s1005 は 2.2、和楽器入りの wa_s2000 は 2.7（伸縮前は 7.0）。

**利用条件の記録（Egatari の attachments 用）**
- 生成に使ったモデル: ACE-Step 1.5 XL-turbo と 5Hz LM 1.7B。コードとウェイトはどちらも MIT License（https://github.com/ace-step/ACE-Step-1.5 ）。
- モデルカードには「生成した音楽は商用目的に使える」と明記されている（https://huggingface.co/ace-step/Ace-Step1.5 ）。
- 後処理に使ったツール: Demucs（MIT）、pedalboard（GPL-3.0。ツールとして使っただけで、配布はしない）、pyloudnorm（MIT）。
- 注意: 学習データの出所は開発元の自己申告であり、外部からは検証できない。

---

## 調査 7：Instagram の Stories と Reels で BGM を使うときの作法（2026-11-07）

**公式資料で確認できたこと**
- **Stories の長さ**: 1 枚あたり 3〜60 秒（Meta 開発者ドキュメント「IG User Media」 https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/media ）。
- **Reels の長さ**: アプリから投稿できるのは最長 3 分（2025-01-21 公式 https://creators.instagram.com/blog/the-latest-with-instagram ）。API 経由なら最長 15 分。
- **音声の仕様**: AAC、最大 48kHz、モノラルかステレオ、128kbps（上記の開発者ドキュメント）。

**第三者の資料にもとづく実務上の目安**
- **1 枚の尺**: 実際に使われるのは 5〜15 秒が中心。視聴者はタップで次へ進む。最初の 1〜3 秒で曲の性格が伝わるようにする。
- **ループ**: Stories は 1 枚を再生し終えると次のカードへ進み、ループしない。Reels は自動でループ再生されるので、終わりから頭へ自然につながる曲が有利。
- **ビジネスアカウントの音楽**: 使える曲は、Meta が無料で提供する Sound Collection にほぼ限られる。自前の BGM を動画に焼き込んで投稿する意味は大きい。
- **音量の目安**: 公式の値はない。通例は −14 LUFS、トゥルーピーク −1 dBTP。
- **スマホのスピーカー**: 実質モノラルで鳴る。モノラルにしても音が崩れないことと、中域でメロディが聞き取れることが重要。

**生成器が既定で出すものへの反映**

| 用途 | 主力 | 補助 |
|---|---|---|
| Stories | 15 秒ループ | 7〜8 秒、15 秒の「締めあり」、60 秒 |
| Reels | 7・15・30 秒のループ | — |

- 調べた限り、日本独自の慣行はない。15 秒は TV CM の標準尺とも重なる。
- モノラル互換の確認（左右の相関がマイナスになっていないか）を、自動採点の項目に加える。
