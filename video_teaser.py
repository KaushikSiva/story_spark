import os


def _get_template(env_key: str, default_text: str) -> str:
    return os.environ.get(env_key, default_text)


def build_video_prompt_base(duration: int) -> str:
    default = (
        "Create an irresistible {duration}-second movie teaser trailer. "
        "Hook viewers in the first 2 seconds with a striking image or motion. "
        "High production value with dramatic lighting and professional cinematography. "
        "Dynamic camera moves: push-ins, whip pans, aerial reveals, match cuts. "
        "No on-screen text overlays or credits; pure visual narrative. "
        "Maintain consistent characters and setting throughout. "
        "Sound: cinematic trailer music with a clear motif, rhythmic pulses, risers, braams, and stingers; tasteful sound design cues; purposeful moments of near-silence before impacts. "
        "Build tension and intrigue; escalate stakes; end on a memorable hard button and audio sting."
    )
    tmpl = _get_template("VIDEO_PROMPT_BASE", default)
    return tmpl.format(duration=duration)


def build_video_prompt_safe_suffix() -> str:
    default = (
        " Content constraints: no depictions of violence, weapons, criminal activity, or harm; keep it suspenseful and family-safe. "
        "Use implication and atmosphere instead of explicit conflict; focus on wonder, stakes, and character emotion."
    )
    return _get_template("VIDEO_PROMPT_SAFE_SUFFIX", default)

