"""Design tokens and the application stylesheet.

The brief for this interface is white, airy, quiet and precise.  That is not
achieved by picking nicer colours for default Qt widgets; it comes from a small
number of decisions applied without exception:

*   one neutral ramp and exactly one accent, used sparingly,
*   a 4-pixel spacing scale, so every gap is a multiple of the same unit,
*   generous padding and large touch targets,
*   borders only where they separate two things that would otherwise collide,
*   no gradients, no drop shadows on flat surfaces, no chrome for its own sake.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    # Surfaces, lightest first.
    canvas: str = "#FFFFFF"
    surface: str = "#FBFBFC"
    surface_sunken: str = "#F5F6F8"
    surface_hover: str = "#F0F1F3"
    overlay: str = "#FFFFFF"

    # Lines.
    border: str = "#E6E8EC"
    border_strong: str = "#D3D7DE"

    # Ink.
    text: str = "#17191C"
    text_secondary: str = "#5C636E"
    text_tertiary: str = "#8B929D"
    text_inverse: str = "#FFFFFF"

    # One accent: a deep, desaturated teal. Distinctive without shouting,
    # and clearly separable from the warm overlay colours used on the image.
    accent: str = "#0F6E63"
    accent_hover: str = "#0C5A51"
    accent_pressed: str = "#094740"
    accent_wash: str = "#E9F3F1"
    accent_border: str = "#BFDBD6"

    # Status.
    warning: str = "#A25B00"
    warning_wash: str = "#FDF3E4"
    danger: str = "#B3261E"
    danger_wash: str = "#FCECEA"
    success: str = "#1B6E3C"

    # Overlay colours drawn on top of greyscale microscopy. Chosen to stay
    # legible against both black and white pixels.
    track_colors: tuple[str, ...] = (
        "#FF6B35", "#22B8CF", "#F7B32B", "#C44DFF",
        "#3DDC84", "#FF4D88", "#4D79FF", "#E8D41F",
    )
    mask_outline: str = "#FF6B35"
    centroid: str = "#FFFFFF"
    axis_guide: str = "#22B8CF"


PALETTE = Palette()

# --------------------------------------------------------------------------
# Scales
# --------------------------------------------------------------------------

SPACE = {"xxs": 2, "xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "2xl": 32, "3xl": 48, "4xl": 64}
RADIUS = {"sm": 6, "md": 10, "lg": 14, "pill": 999}

# "Text" is the small-size optical size of the Windows 11 system face. The
# "Display" cut is drawn for headlines and reads loose and airy at 13px.
FONT_FAMILY = '"Segoe UI Variable Text", "Segoe UI", -apple-system, system-ui, sans-serif'
FONT_DISPLAY = '"Segoe UI Variable Display", "Segoe UI Semibold", "Segoe UI", sans-serif'
FONT_MONO = '"Cascadia Mono", "Consolas", ui-monospace, monospace'

TYPE = {
    "display": 28,
    "title": 20,
    "subtitle": 15,
    "body": 13,
    "caption": 11,
}


def stylesheet(p: Palette = PALETTE) -> str:
    """The whole application stylesheet, generated from the tokens above."""
    return f"""
* {{
    font-family: {FONT_FAMILY};
    font-size: {TYPE['body']}px;
    color: {p.text};
}}

QWidget#Root, QMainWindow, QStackedWidget {{
    background: {p.canvas};
}}

/* Scroll areas must not introduce a grey plate behind white content. */
QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QWidget#SidePanel, QWidget#Inspector {{
    background: {p.canvas};
}}

QToolTip {{
    background: {p.text};
    color: {p.text_inverse};
    border: none;
    border-radius: {RADIUS['sm']}px;
    padding: 6px 9px;
    font-size: {TYPE['caption']}px;
}}

/* ---------------------------------------------------------------- text */
QLabel[role="display"] {{
    font-family: {FONT_DISPLAY};
    font-size: {TYPE['display']}px;
    font-weight: 600;
    color: {p.text};
}}
QLabel[role="title"] {{
    font-family: {FONT_DISPLAY};
    font-size: {TYPE['title']}px;
    font-weight: 600;
}}
QLabel[role="subtitle"] {{
    font-size: {TYPE['subtitle']}px;
    font-weight: 600;
}}
QLabel[role="secondary"] {{
    color: {p.text_secondary};
}}
QLabel[role="tertiary"] {{
    color: {p.text_tertiary};
    font-size: {TYPE['caption']}px;
}}
QLabel[role="metric"] {{
    font-family: {FONT_DISPLAY};
    font-size: {TYPE['title']}px;
    font-weight: 600;
    color: {p.text};
}}
QLabel[role="mono"] {{
    font-family: {FONT_MONO};
    font-size: {TYPE['caption']}px;
    color: {p.text_secondary};
}}

/* -------------------------------------------------------------- buttons */
QPushButton {{
    background: {p.canvas};
    border: 1px solid {p.border_strong};
    border-radius: {RADIUS['md']}px;
    padding: 9px 18px;
    color: {p.text};
    font-weight: 500;
    min-height: 20px;
}}
QPushButton:hover  {{ background: {p.surface_hover}; border-color: {p.text_tertiary}; }}
QPushButton:pressed{{ background: {p.surface_sunken}; }}
QPushButton:disabled {{ color: {p.text_tertiary}; border-color: {p.border}; background: {p.surface}; }}

QPushButton[variant="primary"] {{
    background: {p.accent};
    border: 1px solid {p.accent};
    color: {p.text_inverse};
    padding: 11px 26px;
    font-weight: 600;
}}
QPushButton[variant="primary"]:hover   {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton[variant="primary"]:pressed {{ background: {p.accent_pressed}; }}
QPushButton[variant="primary"]:disabled {{
    background: {p.surface_sunken}; border-color: {p.border}; color: {p.text_tertiary};
}}

QPushButton[variant="ghost"] {{
    background: transparent;
    border: 1px solid transparent;
    padding: 7px 12px;
    color: {p.text_secondary};
}}
QPushButton[variant="ghost"]:hover {{ background: {p.surface_hover}; color: {p.text}; }}

QPushButton[variant="danger"] {{
    background: {p.canvas};
    border: 1px solid {p.danger};
    color: {p.danger};
}}
QPushButton[variant="danger"]:hover {{ background: {p.danger_wash}; }}

/* Small square toggles used for image layers. */
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {RADIUS['sm']}px;
    padding: 6px 10px;
    color: {p.text_secondary};
}}
QToolButton:hover {{ background: {p.surface_hover}; color: {p.text}; }}
QToolButton:checked {{
    background: {p.accent_wash};
    border-color: {p.accent_border};
    color: {p.accent};
    font-weight: 600;
}}

/* --------------------------------------------------------------- inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {p.canvas};
    border: 1px solid {p.border_strong};
    border-radius: {RADIUS['sm']}px;
    padding: 7px 10px;
    selection-background-color: {p.accent_wash};
    selection-color: {p.text};
    min-height: 18px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {p.accent};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background: {p.surface_sunken}; color: {p.text_tertiary};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {p.canvas};
    border: 1px solid {p.border_strong};
    border-radius: {RADIUS['sm']}px;
    padding: 4px;
    outline: none;
    selection-background-color: {p.accent_wash};
    selection-color: {p.text};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}

QCheckBox {{ spacing: 8px; color: {p.text}; }}
QCheckBox::indicator {{
    width: 17px; height: 17px;
    border: 1px solid {p.border_strong};
    border-radius: 5px;
    background: {p.canvas};
}}
QCheckBox::indicator:hover {{ border-color: {p.text_tertiary}; }}
QCheckBox::indicator:checked {{
    background: {p.accent};
    border-color: {p.accent};
    image: url(:/corridor/check.svg);
}}

/* --------------------------------------------------------------- panels */
QFrame[role="card"] {{
    background: {p.canvas};
    border: 1px solid {p.border};
    border-radius: {RADIUS['lg']}px;
}}
QFrame[role="card"]:hover {{ border-color: {p.border_strong}; }}

QFrame[role="panel"] {{
    background: {p.surface};
    border: none;
    border-left: 1px solid {p.border};
}}
QFrame[role="toolbar"] {{
    background: {p.canvas};
    border: none;
    border-bottom: 1px solid {p.border};
}}
QFrame[role="statusbar"] {{
    background: {p.canvas};
    border: none;
    border-top: 1px solid {p.border};
}}
QFrame[role="divider"] {{ background: {p.border}; border: none; max-height: 1px; }}
QFrame[role="vdivider"] {{ background: {p.border}; border: none; max-width: 1px; }}

/* --------------------------------------------------------------- lists */
QListWidget, QTreeWidget, QTableWidget {{
    background: {p.canvas};
    border: 1px solid {p.border};
    border-radius: {RADIUS['md']}px;
    outline: none;
    padding: 4px;
}}
QListWidget::item, QTreeWidget::item {{
    padding: 8px 10px;
    border-radius: {RADIUS['sm']}px;
    color: {p.text};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{ background: {p.surface_hover}; }}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {p.accent_wash};
    color: {p.text};
}}
QHeaderView::section {{
    background: {p.surface};
    border: none;
    border-bottom: 1px solid {p.border};
    padding: 7px 10px;
    color: {p.text_secondary};
    font-weight: 600;
    font-size: {TYPE['caption']}px;
}}

/* ------------------------------------------------------------- progress */
QProgressBar {{
    background: {p.surface_sunken};
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 3px; }}

/* --------------------------------------------------------------- slider */
QSlider::groove:horizontal {{
    background: {p.surface_sunken};
    height: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {p.canvas};
    border: 2px solid {p.accent};
    width: 13px; height: 13px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ border-color: {p.accent_hover}; }}
QSlider::handle:horizontal:disabled {{ border-color: {p.border_strong}; }}

/* ------------------------------------------------------------ scrollbar */
QScrollBar:vertical {{
    background: transparent; width: 11px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {p.border_strong}; border-radius: 5px; min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.text_tertiary}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {p.border_strong}; border-radius: 5px; min-width: 32px;
}}
QScrollBar::handle:horizontal:hover {{ background: {p.text_tertiary}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}

/* ------------------------------------------------------------ groupings */
QGroupBox {{
    border: none;
    margin-top: {SPACE['lg']}px;
    padding-top: {SPACE['sm']}px;
    font-weight: 600;
    color: {p.text_secondary};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 0px;
    padding: 0 0 6px 0;
    font-size: {TYPE['caption']}px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {p.text_tertiary};
}}

QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent;
    padding: 9px 4px;
    margin-right: {SPACE['xl']}px;
    border: none;
    border-bottom: 2px solid transparent;
    color: {p.text_secondary};
}}
QTabBar::tab:selected {{ color: {p.text}; border-bottom-color: {p.accent}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {p.text}; }}

QMenu {{
    background: {p.canvas};
    border: 1px solid {p.border_strong};
    border-radius: {RADIUS['md']}px;
    padding: 6px;
}}
QMenu::item {{ padding: 8px 26px 8px 12px; border-radius: {RADIUS['sm']}px; }}
QMenu::item:selected {{ background: {p.surface_hover}; }}
QMenu::separator {{ height: 1px; background: {p.border}; margin: 5px 8px; }}

QSplitter::handle {{ background: {p.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
"""


def track_color(track_id: int, p: Palette = PALETTE) -> str:
    """A stable colour per track identity."""
    return p.track_colors[(int(track_id) - 1) % len(p.track_colors)]
