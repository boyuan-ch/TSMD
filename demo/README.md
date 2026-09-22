# TSMD Robustness Demo

A zero-build static site that synchronizes official YouTube embeds with
frame-level video-summarization predictions. It contains two selectable demos:

- **Original demo:** ground truth, TripleSumm, and TSMD-Temporal under a shared
	synchronized 50% inference-drop mask.
- **Robustness examples:** four MoSu videos comparing TripleSumm and TSMD-Mix
	under clean and independent 50% modality drop.

Clean-inference curves are dashed and can be hidden with the page toggle. The
timeline's bottom rows show the visual, audio, and text drop masks.

Videos remain hosted by YouTube and their original uploaders. This directory
contains no downloaded video, audio, subtitles, model checkpoints, or HDF5
feature files.

## Run locally

From the repository root:

```bash
python demo/serve.py
```

Open `http://localhost:8765`. The page must be served over HTTP because it
fetches the manifest and per-video JSON files. To use another port, run
`python demo/serve.py --port 9000`.

The page does not contact YouTube until the viewer accepts the privacy notice. It then uses the YouTube IFrame Player API with `youtube-nocookie.com`, native controls, no autoplay, and no API key or user login.

## Controls

- Use the circular button or the YouTube player's native control to play and pause.
- Select one of the playback rates reported as available by YouTube.
- Drag the transport slider to seek.
- Click or drag the lower chart to seek while updating the video frame, bars, revealed curves, cursor, and transport time together.
- Switch between the original and robustness demos with the top tabs.
- Select any of the four ranked robustness examples.
- Toggle the dashed clean-inference curves, meters, legend entries, and metrics.

The YouTube player's `getCurrentTime()` value is the only playback clock. During playback and all seek interactions, the bars and chart are derived from that value.

## Data

`data/manifest.json` defines the demo tabs, video order, YouTube IDs, and paths
to the five self-contained per-video JSON files. Robustness curves are
five-second centered rolling means sampled once per second; modality masks are
unsmoothed binary arrays. The browser linearly interpolates adjacent score
samples for smooth live meters and chart cursors.

## Files

- `index.html`: page structure and accessible controls
- `styles.css`: responsive three-region layout
- `app.js`: YouTube player synchronization, interpolation, controls, and Canvas chart
- `serve.py`: dependency-free local static server
- `data/manifest.json`: demo and video registry
- `data/original/ir9p.json`: original synchronized-drop example
- `data/robustness/*.json`: four independent-drop examples and their metrics
- `privacy.html`: disclosure shown before loading YouTube
- `terms.html`: source ownership and research-output notice

## Public deployment

The repository's `.github/workflows/pages.yml` publishes this directory as the
GitHub Pages site root. Keep all site links relative so the demo also works
under a project path.

If the source video is private, deleted, geographically restricted, age-restricted, or has embedding disabled, the page keeps the score visualization available and directs the viewer to the canonical YouTube URL.

See the [YouTube API Services Terms](https://developers.google.com/youtube/terms/api-services-terms-of-service), [Developer Policies](https://developers.google.com/youtube/terms/developer-policies), and [Required Minimum Functionality](https://developers.google.com/youtube/terms/required-minimum-functionality). These engineering controls reduce redistribution and privacy risks but are not legal advice.
