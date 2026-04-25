# Parking Lot

Hint from user: you may want to take a look at https://huggingface.co/pyannote/speaker-diarization for diarization if Sortformer isn't enough/not working

- Run Sortformer + TDT in parallel (rayon or threads) — biggest win if both fit on GPU
- Reduce Sortformer chunk_len to trade accuracy for speed
- Use CUDA for both models (sortformer + TDT)
- Run diarization and TDT transcription concurrently using threads
- Skip identify mode overhead — only extract snippets after diarization, no TDT needed in identify mode
- Use intra_threads tuning on CPU path
- TDT: reduce chunk size to minimize padding overhead on short audio
