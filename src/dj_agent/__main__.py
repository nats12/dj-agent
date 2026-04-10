"""Launch the DJ Agent GUI.

Usage::

    python -m dj_agent          # Launch mastering GUI
    python -m dj_agent --share  # Launch with public Gradio link
"""

import sys

from .gui import launch

if __name__ == "__main__":
    share = "--share" in sys.argv
    launch(share=share)
