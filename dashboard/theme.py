"""Shared visual language for every dashboard page."""
import plotly.graph_objects as go
import streamlit as st

PRIMARY = "#2563EB"
SECONDARY = "#6B7280"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
ERROR = "#EF4444"
TERTIARY = "#06B6D4"
QUATERNARY = "#8B5CF6"
BACKGROUND = "#FFFFFF"
CARD_SURFACE = "#F8FAFC"
CARD_BORDER = "#E5E7EB"
TEXT_MUTED = "#6B7280"
TEXT_PRIMARY = "#111827"

CATEGORICAL = [PRIMARY, SECONDARY, TERTIARY, QUATERNARY, "#3D6E94"]
DIVERGING = [ERROR, "#F3F4F6", PRIMARY]

# Fixed hue assignment for the 4 models compared in Stage 4 — reused everywhere
# the dashboard overlays or groups all four together, so a model's color never
# shifts between charts.
MODEL_COLORS = {
    "Baseline (production RF, 11 features)": WARNING,
    "Logistic Regression (8 features)": PRIMARY,
    "Random Forest (8 features)": TERTIARY,
    "Gradient Boosting (8 features)": QUATERNARY,
}
MODEL_SHORT_NAMES = {
    "Baseline (production RF, 11 features)": "Baseline RF",
    "Logistic Regression (8 features)": "Logistic Regression",
    "Random Forest (8 features)": "Random Forest",
    "Gradient Boosting (8 features)": "Gradient Boosting",
}

CSS = f"""
<style>
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}
    div[data-testid="stMetric"] {{
        background-color: {CARD_SURFACE};
        border: 1px solid {CARD_BORDER};
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
        transition: box-shadow 0.15s ease, border-color 0.15s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 10px rgba(17, 24, 39, 0.07);
        border-color: {PRIMARY}40;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED};
    }}
    div[data-testid="stMetricValue"] {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        font-size: 1.4rem !important;
        line-height: 1.7rem !important;
        color: {TEXT_PRIMARY} !important;
    }}
    div[data-testid="stMetricValue"] > div {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }}

    /* Bordered containers tagged key="card-..." across every page — one rule
       styles every card without touching Streamlit's internal DOM structure. */
    div[class*="st-key-card-"] {{
        background-color: {CARD_SURFACE} !important;
        border: 1px solid {CARD_BORDER} !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04), 0 1px 3px rgba(17, 24, 39, 0.05) !important;
        transition: box-shadow 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
    }}
    div[class*="st-key-card-"]:hover {{
        box-shadow: 0 6px 16px rgba(17, 24, 39, 0.08), 0 2px 4px rgba(17, 24, 39, 0.06) !important;
        border-color: {PRIMARY}40 !important;
        transform: translateY(-1px);
    }}

    .card {{
        background-color: {CARD_SURFACE};
        border: 1px solid {CARD_BORDER};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }}
    h1, h2, h3 {{
        color: {TEXT_PRIMARY};
    }}
    .takeaway {{
        background-color: #EFF6FF;
        border-left: 3px solid {PRIMARY};
        padding: 0.7rem 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
    }}
    .flow-step {{
        background-color: {CARD_SURFACE};
        border: 1px solid {PRIMARY};
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
        font-weight: 500;
        color: {TEXT_PRIMARY};
        max-width: 420px;
        margin: 0 auto;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }}
    .flow-step-h {{
        background-color: {CARD_SURFACE};
        border: 1px solid {PRIMARY};
        border-radius: 8px;
        padding: 0.45rem 0.5rem;
        text-align: center;
        font-weight: 500;
        font-size: 0.85rem;
        color: {TEXT_PRIMARY};
        overflow-wrap: normal;
        word-break: keep-all;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
        min-height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-sizing: border-box;
    }}
    .flow-arrow {{
        text-align: center;
        color: {TEXT_MUTED};
        font-size: 1.2rem;
        line-height: 1.6rem;
    }}
    .flow-arrow-h {{
        text-align: center;
        color: {TEXT_MUTED};
        font-size: 1.1rem;
        padding-top: 0.6rem;
    }}

    /* Buttons and nav links — styling only, no behavior change */
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {{
        border-radius: 8px;
        transition: box-shadow 0.15s ease, transform 0.15s ease;
    }}
    div[data-testid="stButton"] button:hover, div[data-testid="stFormSubmitButton"] button:hover {{
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.18);
        transform: translateY(-1px);
    }}
    div[data-testid="stFormSubmitButton"] button[kind="primary"] {{
        background-color: {PRIMARY};
        border-color: {PRIMARY};
    }}
    a[data-testid="stPageLink-NavLink"] {{
        border-radius: 8px;
        transition: background-color 0.15s ease;
    }}

    /* Hide the anchor-link icon Streamlit adds next to headings on hover */
    [data-testid="stHeaderActionElements"] {{
        display: none !important;
    }}

    /* Pages with many tabs (e.g. Model Performance's 9) overflow the default
       single-row, horizontal-scroll tab bar with no visible scroll cue — the
       last tab or two end up clipped off-screen. Wrap onto a second row
       instead so every tab stays visible and clickable. */
    div[data-baseweb="tab-list"] {{
        flex-wrap: wrap !important;
        overflow-x: visible !important;
        row-gap: 0.25rem;
    }}
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply one consistent Plotly layout across every chart in the app."""
    fig.update_layout(
        height=height,
        font=dict(family="sans-serif", size=13, color=TEXT_PRIMARY),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=CARD_BORDER)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    return fig


def _slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in text).strip("-")


def metric_card(label: str, value: str, height: int = 100, compact: bool = False) -> None:
    """Equal-height metric card — use inside a column for a snapshot grid."""
    label_size = "0.72rem" if compact else "0.85rem"
    value_size = "1.05rem" if compact else "1.35rem"
    value_line_height = "1.2rem" if compact else "1.5rem"
    with st.container(border=True, height=height, key=f"card-metric-{_slugify(label)}"):
        st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:{label_size};'>{label}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:{value_size}; font-weight:700; color:{TEXT_PRIMARY}; "
            f"line-height:{value_line_height};'>{value}</div>",
            unsafe_allow_html=True,
        )


def highlight_card(icon: str, title: str, text: str, height: int = 165, show_icon: bool = True) -> None:
    """Equal-height highlight card with a Material icon — one sentence only."""
    with st.container(border=True, height=height, key=f"card-highlight-{_slugify(title)}"):
        st.markdown(f"**:material/{icon}: {title}**" if show_icon else f"**{title}**")
        st.caption(text)


def probability_bar(value: float, color: str, height: int = 26) -> None:
    """A horizontal fill bar proportional to a 0-1 probability, in a theme color."""
    pct = max(0.0, min(1.0, value)) * 100
    st.markdown(
        f"<div style='background-color:{CARD_BORDER}; border-radius:6px; height:{height}px; "
        f"overflow:hidden;'>"
        f"<div style='background-color:{color}; width:{pct}%; height:100%; "
        f"border-radius:6px 0 0 6px;'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_flow(steps: list[str], vertical: bool = True) -> None:
    """A clean step-by-step pipeline diagram — no paragraphs."""
    if vertical:
        for i, step in enumerate(steps):
            st.markdown(f"<div class='flow-step'>{step}</div>", unsafe_allow_html=True)
            if i < len(steps) - 1:
                st.markdown("<div class='flow-arrow'>&#8595;</div>", unsafe_allow_html=True)
    else:
        weights = []
        for i in range(len(steps)):
            weights.append(6)
            if i < len(steps) - 1:
                weights.append(1)
        cols = st.columns(weights)
        for i, step in enumerate(steps):
            with cols[i * 2]:
                st.markdown(f"<div class='flow-step-h'>{step}</div>", unsafe_allow_html=True)
            if i < len(steps) - 1:
                with cols[i * 2 + 1]:
                    st.markdown("<div class='flow-arrow-h'>&#8594;</div>", unsafe_allow_html=True)


def takeaway(text: str) -> None:
    st.markdown(f'<div class="takeaway"><strong>Takeaway.</strong> {text}</div>', unsafe_allow_html=True)


def page_header(title: str, intro: str) -> None:
    st.title(title)
    st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:1.05rem;'>{intro}</p>", unsafe_allow_html=True)
    st.divider()


def what_this_means(observation: str, impact: str, decision: str | None = None) -> None:
    """Collapsed-by-default explanation block, used the same way on every chart."""
    with st.expander("What This Means"):
        st.markdown(f"**Observation**  \n{observation}")
        st.markdown(f"**Impact**  \n{impact}")
        if decision:
            st.markdown(f"**Decision**  \n{decision}")
