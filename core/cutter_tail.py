def _build_glow_filter(src_w: int, src_h: int, w: int, h: int, fps: int, duration: float) -> tuple:
    """
    Build an FFmpeg filter chain that creates a glow/pad effect.

    The source video is scaled to fit inside the canvas, centered,
    surrounded by a soft colored glow, with dark fill elsewhere.
    Returns (filter_string, is_complex).
    """
    if src_w / src_h > w / h:
        fg_w = w
        fg_h = int(src_h * w / src_w)
    else:
        fg_h = h
        fg_w = int(src_w * h / src_h)
    fg_w -= fg_w % 2
    fg_h -= fg_h % 2

    ox = (w - fg_w) // 2
    oy = (h - fg_h) // 2

    # Glow settings
    glow_pad = 6
    glow_alpha = "0.4"
    glow_color = "0x8844FF"

    outer_pad = 20
    outer_color = "0x4422AA"
    outer_alpha = "0.2"

    filter_chain = (
        f"[0:v]scale={fg_w}:{fg_h}:flags=lanczos[fg];"
        f"color=c=black@0.85:s={w}x{h}:d={duration}[base];"
        f"[base]drawbox=x={ox - outer_pad}:y={oy - outer_pad}:"
        f"w={fg_w + outer_pad * 2}:h={fg_h + outer_pad * 2}:"
        f"color={outer_color}@{outer_alpha}:t=fill[glow1];"
        f"[glow1]drawbox=x={ox - glow_pad}:y={oy - glow_pad}:"
        f"w={fg_w + glow_pad * 2}:h={fg_h + glow_pad * 2}:"
        f"color={glow_color}@{glow_alpha}:t=fill[glow2];"
        f"[glow2][fg]overlay={ox}:{oy}:format=auto,"
        f"fps={fps},format=yuv420p"
    )

    return filter_chain, True


def _add_subtitle_filter(cmd: list, ass_file: str, job_id: str, clip_id: int) -> list:
    """Add ASS subtitle burning via filter_complex_script to avoid path escaping."""
    import tempfile
    fonts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
    vf_script = os.path.join(tempfile.gettempdir(), f"vf_sub_{job_id}_{clip_id}.txt")
    with open(vf_script, "w", encoding="utf-8") as vfh:
        ass_p = os.path.abspath(ass_file).replace("\\", "/").replace(":", "\\:")
        fonts_p = os.path.abspath(fonts_dir).replace("\\", "/").replace(":", "\\:")
        vfh.write(f"subtitles=filename='{ass_p}':fontsdir='{fonts_p}'\n")
    cmd += ["-vf", f"subtitles=filename='{ass_p}':fontsdir='{fonts_p}'"]
    return cmd


def cut_clip(
    source_path: str,
    job_id: str,
    clip_id: int,
    start: float,
    end: float,
    platform: str = "tiktok",
    captions: Optional[list[dict]] = None,
    caption_style: str = "default",
    brand_template: Optional[str] = None,
    face_track: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
    crop_mode: str = "blur_bg",
) -> str:
    """
    Extract a clip from *source_path*, reformat for *platform*, optionally
    add animated captions and face-aware cropping.
    """
    if platform not in PLATFORM_PRESETS:
        raise ValueError(
            f"Unknown platform '{platform}'. "
            f"Choose from: {list(PLATFORM_PRESETS.keys())}"
        )

    preset = PLATFORM_PRESETS[platform]
    w, h = preset["width"], preset["height"]
    w = w - (w % 2)
    h = h - (h % 2)
    fps = preset["fps"]
    duration = end - start

    if duration > preset["max_duration"]:
        end = start + preset["max_duration"]
        duration = preset["max_duration"]
        logger.info("Clip truncated to %ss (platform limit)", preset["max_duration"])

    if duration < 3:
        raise ValueError("Clip too short (minimum 3 seconds)")

    brand = None
    if brand_template:
        if isinstance(brand_template, dict):
            bt_dict = brand_template
        else:
            bt_dict = get_brand_template(brand_template)
        brand = BrandTemplate(**bt_dict)
        if not caption_style or caption_style == "default":
            caption_style = bt_dict.get("caption_style", caption_style)

    clip_dir = os.path.join(CLIPS_DIR, job_id)
    os.makedirs(clip_dir, exist_ok=True)
    output_filename = f"clip_{clip_id:02d}_{platform}.mp4"
    output_path = os.path.join(clip_dir, output_filename)

    try:
        src_info = _probe_video(source_path)
        src_w = int(src_info["streams"][0].get("width", 1920))
        src_h = int(src_info["streams"][0].get("height", 1080))
    except Exception as exc:
        logger.warning("Could not probe source video (%s), assuming 1920x1080", exc)
        src_w, src_h = 1920, 1080

    source_has_audio = _has_audio(source_path)

    ffmpeg_env = os.environ.copy()
    ffmpeg_env.pop("FONTCONFIG_FILE", None)
    ffmpeg_env.pop("FONTCONFIG_PATH", None)
    ffmpeg_env.pop("FONTCONFIG_SYSROOT", None)

    # Build video filter chain
    vf_string = None
    is_complex = False

    if crop_mode == "blur_bg":
        logger.info("cut_clip: using glow_bg effect (FFmpeg drawbox)")
        vf_string, is_complex = _build_glow_filter(src_w, src_h, w, h, fps, duration)

    elif crop_mode == "face_track":
        face_data = _detect_face_region(source_path, start, duration)
        if face_data:
            face_cx = face_data["cx"]
            scale_f = f"scale=-2:{h}:flags=lanczos"
            x_off = max(0, min(face_cx - w // 2, src_w - w))
            crop_f = f"crop={w}:{h}:{x_off}:0"
            vf_string = f"{scale_f},{crop_f},fps={fps},format=yuv420p"
            logger.info("Face track: face at (%d), crop x_offset=%d", face_cx, x_off)
        else:
            logger.info("Face track: no face detected, falling back to center_crop")
            crop_mode = "center_crop"

    if crop_mode == "center_crop" or vf_string is None:
        scale_f = f"scale=-2:{h}:flags=lanczos"
        crop_f = f"crop={w}:{h}:(in_w-out_w)/2:0"
        vf_string = f"{scale_f},{crop_f},fps={fps},format=yuv420p"

    # Captions
    ass_file = None
    if captions:
        try:
            from core.ass_subtitles import generate_ass_subtitles
            ass_file = generate_ass_subtitles(
                words=captions, width=w, height=h, template_name=caption_style,
            )
            logger.info("ASS subtitles generated: %s", ass_file)
        except Exception as exc:
            logger.warning("ASS subtitle generation failed (%s), skipping captions", exc)

    # Build subtitle filter script
    vf_script = None
    if ass_file and os.path.isfile(ass_file):
        import tempfile as _tf
        ass_abs = os.path.abspath(ass_file)
        fonts_abs = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts"))
        vf_script = os.path.join(_tf.gettempdir(), f"vf_sub_{job_id}_{clip_id}.txt")
        with open(vf_script, "w", encoding="utf-8") as vfh:
            ass_p = ass_abs.replace("\\", "/").replace(":", "\\:")
            fonts_p = fonts_abs.replace("\\", "/").replace(":", "\\:")
            vfh.write(f"subtitles=filename='{ass_p}':fontsdir='{fonts_p}'\n")
        logger.info("ASS subtitles filter script: %s", vf_script)

    logger.info("cut_clip vf: %s", vf_string[:200])

    # Build FFmpeg command
    import tempfile as _tf
    cmd = [FFMPEG, "-y", "-ss", str(start), "-i", source_path, "-t", str(duration)]

    if is_complex:
        # Write complex filter to temp file, append subtitles if needed
        vf_file = os.path.join(_tf.gettempdir(), f"vf_{job_id}_{clip_id}.txt")
        with open(vf_file, "w", encoding="utf-8") as vfh:
            vfh.write(vf_string)
            if vf_script:
                # Append subtitle filter after the glow chain output
                vfh.write(f";[vf]subtitles=filename='{ass_abs.replace(chr(92), '/').replace(chr(58), chr(92)+chr(58))}':fontsdir='{fonts_abs.replace(chr(92), '/').replace(chr(58), chr(92)+chr(58))}'[vfinal]\n")
                # Rename output pad
                vfh.write("[vfinal]\n")
        cmd += ["-filter_complex_script", vf_file]
    else:
        cmd += ["-vf", vf_string]
        if vf_script:
            cmd += ["-filter_complex_script", vf_script]

    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-b:v", preset["video_bitrate"],
    ]
    if source_has_audio:
        cmd += ["-c:a", "aac", "-b:a", preset["audio_bitrate"]]
    else:
        cmd += ["-an"]
    cmd += [
        "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-threads", "0",
        output_path,
    ]

    logger.info("cut_clip: %s", " ".join(cmd[:10]) + " ...")

    if progress_callback:
        progress_callback(0.0)

    logger.info("cut_clip: source exists=%s dir exists=%s", os.path.exists(source_path), os.path.isdir(clip_dir))

    try:
        if is_complex:
            log_path = os.path.join(clip_dir, f"ffmpeg_{clip_id:02d}.log")
            log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
            proc = subprocess.Popen(
                cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=ffmpeg_env,
            )
            ret = proc.wait(timeout=600)
            log_fh.close()
            if ret != 0:
                try:
                    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                        tail = f.readlines()[-30:]
                except Exception:
                    tail = ["(could not read log)"]
                raise RuntimeError(
                    f"FFmpeg cut failed (rc={ret}) for {output_path}:\n{''.join(tail)}"
                )
        else:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=ffmpeg_env,
            )
            total_us = int(duration * 1_000_000)
            try:
                stdout_data, stderr_data = proc.communicate(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError("FFmpeg cut timed out after 600s")

            if stderr_data:
                logger.debug("FFmpeg stderr:\n%s", stderr_data[-2000:])

            if progress_callback and stderr_data:
                for line in stderr_data.splitlines():
                    line = line.strip()
                    if line.startswith("out_time_us="):
                        try:
                            cur_us = int(line.split("=", 1)[1])
                            pct = min(100.0, cur_us / total_us * 100)
                            progress_callback(round(pct, 1))
                        except (ValueError, ZeroDivisionError):
                            pass

            if proc.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg cut failed (rc={proc.returncode}) for {output_path}:\n"
                    f"{stderr_data[-3000:] if stderr_data else 'no stderr'}"
                )

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"FFmpeg error: {exc}")
    finally:
        for tmp_f in [vf_file if is_complex else None, vf_script, ass_file]:
            if tmp_f and os.path.isfile(tmp_f):
                try:
                    os.remove(tmp_f)
                except OSError:
                    pass

    if progress_callback:
        progress_callback(100.0)

    logger.info("cut_clip: output saved to %s", output_path)
    return output_path
