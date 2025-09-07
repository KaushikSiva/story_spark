import os
from typing import Optional, List


def _get_template(env_key: str, default_text: str) -> str:
    return os.environ.get(env_key, default_text)


def build_images_prompt_from_synopsis(synopsis: str, n: int, *, character_target: Optional[int]) -> str:
    n = max(1, n)
    ct = int(character_target) if character_target is not None else None
    # Default template includes character reference guidance
    default = (
        "Create a beautifully entertaining {n} part story with {n} narrative beats inspired by the following movie synopsis. "
        "Tell the story purely through imagery with no words or text on the images. "
        "Keep characters, setting, and styling consistent across all parts. {ref_note}\n\n"
        "Synopsis:\n{synopsis}\n\n"
        "Output: Create a single, coherent image that visually stitches together all {n} parts into one cinematic collage."
    )
    tmpl = _get_template("STORY_IMAGES_PROMPT_TEMPLATE", default)
    ref_note = (
        f"You may be given up to {ct + 4 if ct is not None else 8} candidate human reference photos from a 'people' folder. "
        f"If any match the story's characters, use whichever {ct if ct is not None else 'few'} are most suitable. "
        f"If characters are non-human or no references fit, ignore them and invent consistent characters. "
    )
    return tmpl.format(n=n, synopsis=synopsis.strip(), ref_note=ref_note)


def build_poster_prompt(title: str, synopsis: str, director: str = "Bruno", *, character_target: Optional[int] = None) -> str:
    # Default poster template emphasizes identity matching
    default = (
        "Design a cinematic FIRST LOOK MOVIE POSTER for the film titled '{title}'. "
        "Director credit: Directed by {director}. "
        "Use the following synopsis to guide the imagery, characters, tone, and setting. "
        "CRITICAL: Match the same principal characters as shown in the provided reference photos and frames; keep face identity, hairstyle, and costume colors consistent. "
        "If candidate human reference photos are provided, choose whichever best match up to {character_target} characters; "
        "if characters are non-human or references do not fit, ignore them and invent consistent characters. "
        "Include tasteful, legible on-poster text: the film title '{title}', the credit 'Directed by {director}', and 2–4 thematically appropriate character names you invent (avoid real IP). "
        "Style: high fidelity, cohesive layout, strong typography, filmic color grading, premium key-art quality. "
        "Avoid watermarks or logos.\n\n"
        "Synopsis:\n{synopsis}"
    )
    tmpl = _get_template("POSTER_PROMPT_TEMPLATE", default)
    ct = character_target if character_target is not None else 4
    return tmpl.format(title=title, director=director, synopsis=synopsis.strip(), character_target=ct)


# Utility hooks that app can import to avoid duplicating logic if needed later
def load_reference_image_parts(*args, **kwargs):  # pragma: no cover
    from app3 import load_reference_image_parts as _impl
    return _impl(*args, **kwargs)


def file_to_genai_part(*args, **kwargs):  # pragma: no cover
    from app3 import file_to_genai_part as _impl
    return _impl(*args, **kwargs)

