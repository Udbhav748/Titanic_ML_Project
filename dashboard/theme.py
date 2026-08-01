"""Shared visual language for every dashboard page."""
import plotly.graph_objects as go
import streamlit as st

PRIMARY = "#1F4E79"
SECONDARY = "#5B6572"
SUCCESS = "#2E7D32"
WARNING = "#F9A825"
ERROR = "#C62828"
TERTIARY = "#1BAF7A"
QUATERNARY = "#6B4FBB"
BACKGROUND = "#FAFAFA"
CARD_BORDER = "#E2E5E9"
TEXT_MUTED = "#5B6572"

CATEGORICAL = [PRIMARY, SECONDARY, "#7A9CBF", "#A6ADB4", "#3D6E94"]
DIVERGING = [ERROR, "#F0EFEC", PRIMARY]

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
        background-color: white;
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 1rem 1.1rem;
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
    }}
    div[data-testid="stMetricValue"] > div {{
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }}
    .card {{
        background-color: white;
        border: 1px solid {CARD_BORDER};
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
    }}
    h1, h2, h3 {{
        color: #14213D;
    }}
    .takeaway {{
        background-color: #F0F4F8;
        border-left: 3px solid {PRIMARY};
        padding: 0.7rem 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }}
    .flow-step {{
        background-color: white;
        border: 1px solid {PRIMARY};
        border-radius: 6px;
        padding: 0.5rem 1rem;
        text-align: center;
        font-weight: 500;
        color: #14213D;
        max-width: 420px;
        margin: 0 auto;
    }}
    .flow-step-h {{
        background-color: white;
        border: 1px solid {PRIMARY};
        border-radius: 6px;
        padding: 0.45rem 0.5rem;
        text-align: center;
        font-weight: 500;
        font-size: 0.85rem;
        color: #14213D;
        overflow-wrap: normal;
        word-break: keep-all;
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
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply one consistent Plotly layout across every chart in the app."""
    fig.update_layout(
        height=height,
        font=dict(family="sans-serif", size=13, color="#14213D"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=40),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=CARD_BORDER)
    fig.update_yaxes(showgrid=True, gridcolor="#F0F0F0", zeroline=False)
    return fig


def metric_card(label: str, value: str, height: int = 100) -> None:
    """Equal-height metric card — use inside a column for a snapshot grid."""
    with st.container(border=True, height=height):
        st.markdown(f"<div style='color:{TEXT_MUTED}; font-size:0.85rem;'>{label}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:1.35rem; font-weight:700; color:#14213D; line-height:1.5rem;'>{value}</div>",
            unsafe_allow_html=True,
        )


def highlight_card(icon: str, title: str, text: str, height: int = 165) -> None:
    """Equal-height highlight card with a Material icon — one sentence only."""
    with st.container(border=True, height=height):
        st.markdown(f"**:material/{icon}: {title}**")
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
