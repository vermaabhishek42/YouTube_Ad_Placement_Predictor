# app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="YouTube Ad-Placement Predictor",
    page_icon="🎯",
    layout="wide",
)

# =========================
# Helpers
# =========================
def safe_divide(a: float, b: float) -> float:
    return float(a) / float(b) if b and b != 0 else 0.0


def duration_bucket(seconds: float) -> str:
    if seconds < 60:
        return "Short (<1 min)"
    elif seconds < 300:
        return "Medium (1–5 min)"
    elif seconds < 900:
        return "Long (5–15 min)"
    else:
        return "Very Long (15+ min)"


def get_strategy_text(tier: str, seconds: float, er: float) -> str:
    """
    Business interpretation text (kept simple + academically defendable).
    """
    tier_l = tier.lower()

    if "premium" in tier_l or "mid" in tier_l:
        return "Use **mid-roll ads** (best when the video is long enough and engagement is strong)."
    if "high" in tier_l or "pre" in tier_l:
        return "Use **pre-roll / bumper ads** (best for reach and short-form inventory)."
    return "Use **standard placement** (good default when engagement/inventory is moderate)."


@st.cache_resource
def load_model_and_features():
    """
    Loads the trained model and the exact feature order used during training.
    """
    model_path = "youtube_ad_tier_model.pkl"
    features_path = "model_features.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure youtube_ad_tier_model.pkl is in the same folder as app.py."
        )

    model = joblib.load(model_path)

    # Feature list: prefer your saved feature order
    if os.path.exists(features_path):
        features = joblib.load(features_path)
        # sometimes saved as numpy array
        features = list(features)
    else:
        # fallback: try sklearn feature names
        features = list(getattr(model, "feature_names_in_", []))
        if not features:
            raise FileNotFoundError(
                f"Feature list not found: {features_path} and model has no feature_names_in_. "
                "Please keep model_features.pkl in the project root."
            )

    return model, features


def build_feature_row(
    features: list,
    views: float,
    likes: float,
    comments: float,
    seconds: float,
) -> pd.DataFrame:
    """
    Build exactly the feature row the model expects.
    Supports multiple naming conventions safely.
    """
    # Derived signals (business-friendly + stable)
    engagement_rate = safe_divide(likes + comments, views) * 100.0  # %
    like_view_ratio = safe_divide(likes, views) * 100.0            # %
    comment_view_ratio = safe_divide(comments, views) * 100.0      # %

    # Provide many aliases; we will fill only those requested by `features`
    candidates = {
        # common raw metrics
        "view_count": views,
        "views": views,
        "total_views": views,

        "like_count": likes,
        "likes": likes,
        "total_likes": likes,

        "comment_count": comments,
        "comments": comments,
        "total_comments": comments,

        "duration_seconds": seconds,
        "duration": seconds,

        # derived metrics
        "engagement_rate": engagement_rate,
        "engagement_pct": engagement_rate,
        "like_view_ratio": like_view_ratio,
        "likes_per_view": safe_divide(likes, views),
        "comment_view_ratio": comment_view_ratio,
        "comments_per_view": safe_divide(comments, views),
    }

    row = {f: candidates.get(f, 0.0) for f in features}
    return pd.DataFrame([row])


def pretty_tier_name(tier: str) -> str:
    # Keep the tier label readable
    return tier.strip()


# =========================
# Load model
# =========================
try:
    model, FEATURES = load_model_and_features()
except Exception as e:
    st.error(str(e))
    st.stop()

# Class labels (for probability table)
CLASS_NAMES = list(getattr(model, "classes_", []))
CLASS_NAMES = [pretty_tier_name(str(c)) for c in CLASS_NAMES]

# =========================
# UI Header
# =========================
st.markdown(
    """
    <div style="padding: 8px 0 2px 0;">
      <h1 style="margin-bottom: 0;">🎯 YouTube Ad-Placement Predictor</h1>
      <p style="opacity: 0.85; margin-top: 4px;">
        Decision-support tool using a trained ML classifier to recommend an ad placement tier
        based on engagement + video inventory (duration).
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Sidebar inputs
# =========================
st.sidebar.header("📥 Input Video Metrics")

views = st.sidebar.number_input("Total Views", min_value=0, value=1000000, step=1000)
likes = st.sidebar.number_input("Total Likes", min_value=0, value=50000, step=100)
comments = st.sidebar.number_input("Total Comments", min_value=0, value=2000, step=10)
seconds = st.sidebar.slider("Duration (seconds)", min_value=1, max_value=3600, value=480, step=1)

# Derived metrics
engagement_rate = safe_divide(likes + comments, views) * 100.0
likes_ratio = safe_divide(likes, views) * 100.0
bucket = duration_bucket(seconds)

# =========================
# Main Layout
# =========================
left, right = st.columns([1.1, 1.4], gap="large")

with left:
    st.subheader("Video Profile Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Engagement Rate", f"{engagement_rate:.2f}%")
    c2.metric("Duration", f"{seconds:.0f}s")
    c3.metric("Likes/Views", f"{likes_ratio:.2f}%")

    st.write("**Duration bucket:**", bucket)

    with st.expander("What counts as engagement here?"):
        st.write(
            "- Engagement Rate = (likes + comments) / views × 100\n"
            "- This is a simplified proxy for audience quality.\n"
            "- In real ad planning, you also consider demographics, CPM/CPA goals, and brand safety."
        )

with right:
    st.subheader("Recommended Placement Strategy")

    # Button is optional; by default always compute.
    run = st.button("Generate Ad Strategy", type="primary")

    if run or True:
        # Build feature row in correct order
        X_new = build_feature_row(FEATURES, views, likes, comments, seconds)

        # Predict
        pred = model.predict(X_new)[0]
        pred_name = pretty_tier_name(str(pred))

        # Probabilities (if available)
        probs = None
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_new)[0]
            confidence = float(np.max(probs)) * 100.0

        # Display
        st.markdown(f"### 🏷️ TIER: **{pred_name}**")

        strategy = get_strategy_text(pred_name, seconds, engagement_rate)
        st.info(f"💡 **Strategy:** {strategy}")

        if confidence is not None:
            st.metric("Model Confidence", f"{confidence:.2f}%")

        # Probability table
        if probs is not None and CLASS_NAMES:
            st.markdown("### Probability by tier")
            prob_df = pd.DataFrame(
                {"tier": CLASS_NAMES, "probability": probs}
            ).sort_values("probability", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        # Explanation block (business + academic)
        st.markdown("### Why this recommendation?")
        st.write(f"- **Engagement rate:** {engagement_rate:.2f}%")
        st.write(f"- **Duration bucket:** {bucket}")
        st.write("")
        st.write("**Interpretation for business owners:**")
        st.write(
            "- Higher engagement often indicates stronger audience quality → better ad performance potential.\n"
            "- Longer videos create more **mid-roll inventory**; short videos fit **pre-roll/bumper** formats.\n"
            "- This output is **decision-support**, not an absolute rule."
        )

# Footer disclaimer (important for academic credibility)
st.caption(
    "Decision-support only. Real ad choices also depend on audience demographics, CPM/CPA goals, content category, "
    "brand safety requirements, and campaign objectives."
)
