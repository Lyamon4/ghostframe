# ghostframe

**Open-source frame generation. No paid DLLs, no vendor lock-in.**

ghostframe creates the frames in between the ones a game renders: motion-compensated interpolation on the GPU, written from scratch in Rust and WGSL compute shaders. It runs on Vulkan, DirectX 12 and Metal through [wgpu](https://wgpu.rs).

The goal: what Lossless Scaling's frame generation does, but free and open. Today the Linux port of that feature ([lsfg-vk](https://github.com/PancakeTAS/lsfg-vk)) still needs the DLL from the paid Windows app, and AMD's open FSR 3 frame generation only works in games that integrate it. ghostframe needs nothing but the frames themselves.

![blend vs ghostframe vs the real frame](assets/compare.png)

*A dropped frame regenerated from its two neighbours. Left: a plain cross-fade. Middle: ghostframe. Right: the frame the game actually rendered.*

## Measured, not eyeballed

`ghostframe bench` renders game-like scenes, drops every other frame of a 60 fps sequence, regenerates it, and compares it with the real one (PSNR, higher is better):

| scene | motion | repeat frame | cross-fade | **ghostframe** |
|---|---|---|---|---|
| pan | camera pan, 6 px/frame | 22.1 dB | 25.3 dB | **38.0 dB** |
| sprites | 8 objects up to 14 px/frame, occlusions | 25.1 dB | 27.8 dB | **35.1 dB** |
| fast | camera 22 px/frame + objects 30 px/frame | 16.5 dB | 18.9 dB | **26.4 dB** |
| zoom | zoom in 1.5%/frame | 24.2 dB | 27.4 dB | **38.2 dB** |
| **mean** | | 22.0 dB | 24.8 dB | **34.4 dB** |

*960×540, RTX 4060 (Vulkan). "Repeat frame" is what you see without frame generation; "cross-fade" is interpolation without motion.*

Speed on an RTX 4060, per generated frame:

| resolution | time | generated frames/s |
|---|---|---|
| 1280×720 | 4.4 ms | 229 |
| 1920×1080 | 6.5 ms | 153 |
| 2560×1440 | 12.5 ms | 80 |

Comfortably fast enough to take a 1080p game from 30 to 60 fps or from 60 to 120 fps.

## Try it

```bash
git clone https://github.com/Lyamon4/ghostframe && cd ghostframe
cargo run --release -- demo            # live window: a game at 30 fps vs ghostframe at 60 fps
cargo run --release -- bench --save out # the quality table above, plus example frames
cargo run --release -- speed            # GPU time per frame at 1080p
cargo run --release -- interp frames/ out/ --factor 2   # double the frame rate of a PNG sequence
```

To use your own footage: `ffmpeg -i clip.mp4 frames/%05d.png`, run `interp`, then `ffmpeg -framerate 60 -i out/%06d.png clip60.mp4`.

## How it works

Everything runs in compute shaders ([`src/shaders`](src/shaders)):

1. **Luma pyramid.** Each frame is reduced to brightness at 1/1, 1/2, 1/4, 1/8 and 1/16 scale.
2. **Bilateral motion search on the in-between grid.** For each 8×8 block of the frame being created, find the motion `v` such that the old frame at `p − v/2` matches the new frame at `p + v/2`. Searching on the output grid gives every output block exactly one vector: no holes and no overlaps, unlike pushing a flow field forward. It goes coarse to fine:
   - an exhaustive ±6 px search at 1/16 scale (±96 px at full resolution), so fast motion is caught;
   - at each finer level, seeds from the coarser level plus candidates from the 8 neighbouring blocks (3-D recursive search, the technique behind TV motion smoothing), with a smoothness penalty;
   - refinement down to a quarter pixel.

   From 360p up, the search stops at half resolution. It is 4× cheaper and measured no worse.
3. **Vector median filter.** Isolated wrong vectors are replaced by the most central vector of their 3×3 neighbourhood.
4. **Per-pixel composition.** Each output pixel uses a smooth (bilinear) vector field. Where neighbouring blocks disagree, at object edges, it picks the vector whose two motion-compensated samples agree best on a small luma cross, then blends the two frames by time `t`. Any `t` works, so 3× and 4× frame generation fall out naturally.

On GPUs that can filter 32-bit float textures (all desktop GPUs), block matching uses hardware bilinear sampling. Elsewhere a manual path is compiled in.

```
src/
  engine.rs   GPU pipeline: textures, bind groups, dispatch order
  shaders/    pyramid.wgsl, motion.wgsl, compose.wgsl
  scene.rs    procedural game-like test scenes with exact ground truth, PSNR
  demo.rs     side-by-side live window (winit + wgpu)
  main.rs     CLI: bench, speed, interp, demo, info
```

## Status and roadmap

ghostframe is a working interpolation engine with an honest benchmark, not yet a drop-in game tool.

- [x] GPU motion estimation and frame composition, any `t`
- [x] Ground-truth benchmark, speed test, PNG-sequence interpolation, live demo
- [ ] **Linux Vulkan layer** (`ENABLE_GHOSTFRAME=1 %command%`): hook `vkQueuePresentKHR`, run the same shaders (WGSL → SPIR-V via naga) on the game's device, present the extra frames. Steam Deck is the main target.
- [ ] **Windows window capture**, Lossless Scaling style: capture a game window and present at 2× in a borderless overlay.
- [ ] **Better occlusions.** The weak spot is visible in the "fast" row: where an object uncovers background, both frames don't contain the same pixels. Using a third frame to decide which side is visible is the next step.
- [ ] Optional small learned refinement pass for edges.

## License

MIT. Everything here is original. No code, shaders or binaries from Lossless Scaling or any other proprietary tool.
