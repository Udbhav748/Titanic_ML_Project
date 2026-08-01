"""
Generate the three flow diagrams used in README.md — architecture,
ML pipeline, and deployment. One shared style so they read as one system.

Run: python docs/generate_diagrams.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DOCS_DIR = Path(__file__).resolve().parent

PRIMARY = "#1F4E79"
TEXT = "#14213D"
TEXT_MUTED = "#5B6572"
BORDER = "#1F4E79"
BACKGROUND = "#FFFFFF"

BOX_WIDTH = 6.4
BOX_HEIGHT = 0.62
GAP = 0.42
FIGURE_PAD = 0.5


def draw_flow_diagram(steps: list[str], title: str, output_path: Path, subtitle: str = "") -> None:
    n = len(steps)
    fig_height = n * (BOX_HEIGHT + GAP) + FIGURE_PAD + (0.6 if subtitle else 0.3)
    fig, ax = plt.subplots(figsize=(7.5, fig_height))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    ax.set_xlim(0, BOX_WIDTH + 1)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    y_title = fig_height - 0.15
    ax.text(
        (BOX_WIDTH + 1) / 2, y_title, title,
        ha="center", va="top", fontsize=15, fontweight="bold", color=TEXT,
    )
    top_offset = 0.55
    if subtitle:
        ax.text(
            (BOX_WIDTH + 1) / 2, y_title - 0.42, subtitle,
            ha="center", va="top", fontsize=9.5, color=TEXT_MUTED,
        )
        top_offset = 0.95

    x0 = 0.5
    y = fig_height - top_offset - BOX_HEIGHT

    for i, step in enumerate(steps):
        box = FancyBboxPatch(
            (x0, y), BOX_WIDTH, BOX_HEIGHT,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6, edgecolor=BORDER, facecolor="white",
        )
        ax.add_patch(box)
        ax.text(
            x0 + BOX_WIDTH / 2, y + BOX_HEIGHT / 2, step,
            ha="center", va="center", fontsize=10.5, color=TEXT, wrap=True,
        )

        if i < n - 1:
            arrow_top = y
            arrow_bottom = y - GAP
            ax.annotate(
                "", xy=(x0 + BOX_WIDTH / 2, arrow_bottom + 0.06),
                xytext=(x0 + BOX_WIDTH / 2, arrow_top - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=TEXT_MUTED, lw=1.4, shrinkA=0, shrinkB=0),
            )
        y -= (BOX_HEIGHT + GAP)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output_path}")


def main() -> None:
    draw_flow_diagram(
        steps=[
            "Raw Request (JSON)",
            "Pydantic Validation (PassengerInput)",
            "Load Preprocessing Artifact (artifacts/preprocessing.pkl)",
            "Transform Features (TitanicPreprocessor)",
            "Load Model (model_v2.pkl)",
            "Predict Probability",
            "Generate Response (survived, probability, model_version)",
            "Return JSON",
        ],
        title="System Architecture",
        subtitle="Prediction request flow",
        output_path=DOCS_DIR / "architecture-diagram.png",
    )

    draw_flow_diagram(
        steps=[
            "Dataset",
            "Cleaning",
            "Feature Engineering",
            "Feature Selection",
            "Training",
            "Evaluation",
            "Deployment",
        ],
        title="Machine Learning Pipeline",
        subtitle="Stages 1 through 6",
        output_path=DOCS_DIR / "pipeline-diagram.png",
    )

    draw_flow_diagram(
        steps=[
            "Original Features (11)",
            "Statistical Audit (Stage 1 — chi-square / point-biserial)",
            "Mutual Information (Stage 2)",
            "Multicollinearity — VIF (Stage 2)",
            "Engineering Review (Stage 3 — cost, risk, interpretability)",
            "Final Production Features (8)",
        ],
        title="Feature Selection Journey",
        subtitle="Stages 1-3",
        output_path=DOCS_DIR / "feature-selection-journey.png",
    )

    draw_flow_diagram(
        steps=[
            "git pull on EC2 instance",
            "docker build -t titanic-api:v2 .",
            "Pre-cutover health check on a spare port",
            "Stop old container, start new container",
            "Verify /health reports expected model_version",
        ],
        title="Deployment Pipeline",
        subtitle="Docker + AWS EC2",
        output_path=DOCS_DIR / "deployment-diagram.png",
    )


if __name__ == "__main__":
    main()
