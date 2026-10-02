
import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Profile Matching System",
    page_icon="🤝",
    layout="wide"
)


# ============================================
# LOAD DATA
# ============================================

users = pd.read_csv("/content/users.csv")

feedback_file = "/content/feedback.csv"

try:
    feedback = pd.read_csv(feedback_file)
except:
    feedback = pd.DataFrame(
        columns=[
            "user_id",
            "matched_user_id",
            "action",
            "timestamp"
        ]
    )


# ============================================
# TEXT CLEANING
# ============================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================
# COMBINED PROFILE
# ============================================

users["combined_text"] = (
    users["professional_summary"].fillna("") + " " +
    users["about_me"].fillna("") + " " +
    users["intrests"].fillna("") + " " +
    users["profession"].fillna("")
)

users["clean_text"] = users[
    "combined_text"
].apply(clean_text)


# ============================================
# TF-IDF
# ============================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(
    users["clean_text"]
)

text_similarity = cosine_similarity(
    tfidf_matrix
)


# ============================================
# MBTI
# ============================================

mbti_groups = {

    "NT": [
        "INTJ",
        "INTP",
        "ENTJ",
        "ENTP"
    ],

    "NF": [
        "INFJ",
        "INFP",
        "ENFJ",
        "ENFP"
    ],

    "SJ": [
        "ISTJ",
        "ISFJ",
        "ESTJ",
        "ESFJ"
    ],

    "SP": [
        "ISTP",
        "ISFP",
        "ESTP",
        "ESFP"
    ]
}


def get_mbti_group(mbti):

    mbti = str(mbti).upper()

    for group, types in mbti_groups.items():

        if mbti in types:
            return group

    return None


def mbti_compatibility(
    mbti1,
    mbti2
):

    mbti1 = str(mbti1).upper()
    mbti2 = str(mbti2).upper()

    if mbti1 == mbti2:
        return 1.0

    if get_mbti_group(mbti1) == get_mbti_group(mbti2):
        return 0.7

    return 0.4


# ============================================
# LOCATION
# ============================================

def location_compatibility(
    location1,
    location2
):

    location1 = str(
        location1
    ).strip().lower()

    location2 = str(
        location2
    ).strip().lower()

    if location1 == location2:
        return 1.0

    return 0.5


# ============================================
# ADAPTIVE WEIGHTS
# ============================================

BASE_WEIGHTS = np.array(
    [0.50, 0.30, 0.20],
    dtype=float
)

FEATURES = [
    "text_similarity",
    "mbti_score",
    "location_score"
]


def normalize_weights(
    weights,
    fallback=None
):

    weights = np.abs(
        np.asarray(
            weights,
            dtype=float
        )
    )

    total = weights.sum()

    if total == 0:

        if fallback is not None:
            return np.asarray(
                fallback,
                dtype=float
            )

        return BASE_WEIGHTS.copy()

    return weights / total


def create_feedback_features(
    feedback_df
):

    rows = []

    for _, row in feedback_df.iterrows():

        try:

            user_matches = users.index[
                users["user_id"] ==
                row["user_id"]
            ]

            matched_matches = users.index[
                users["user_id"] ==
                row["matched_user_id"]
            ]

            if (
                len(user_matches) == 0
                or
                len(matched_matches) == 0
            ):
                continue

            user_index = user_matches[0]

            matched_index = matched_matches[0]

            rows.append({

                "user_id":
                    row["user_id"],

                "matched_user_id":
                    row["matched_user_id"],

                "text_similarity":
                    float(
                        text_similarity[
                            user_index,
                            matched_index
                        ]
                    ),

                "mbti_score":
                    float(
                        mbti_compatibility(
                            users.iloc[
                                user_index
                            ]["mbti"],

                            users.iloc[
                                matched_index
                            ]["mbti"]
                        )
                    ),

                "location_score":
                    float(
                        location_compatibility(
                            users.iloc[
                                user_index
                            ]["location"],

                            users.iloc[
                                matched_index
                            ]["location"]
                        )
                    ),

                "accepted":
                    int(row["action"])
            })

        except Exception:

            continue

    return pd.DataFrame(rows)


def learn_dynamic_weights(
    feedback_df
):

    feature_data = (
        create_feedback_features(
            feedback_df
        )
    )

    global_weights = (
        BASE_WEIGHTS.copy()
    )

    personalized_weights = {}

    # --------------------------------------------------------
    # Global learning
    # --------------------------------------------------------

    if (
        len(feature_data) >= 4
        and
        feature_data[
            "accepted"
        ].nunique() >= 2
    ):

        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        model.fit(
            feature_data[FEATURES],
            feature_data["accepted"]
        )

        global_weights = normalize_weights(
            model.coef_[0],
            BASE_WEIGHTS
        )

    # --------------------------------------------------------
    # User-specific learning
    # --------------------------------------------------------

    for user_id, group in (
        feature_data.groupby("user_id")
    ):

        if len(group) < 4:
            continue

        if group["accepted"].nunique() < 2:
            continue

        try:

            user_model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )

            user_model.fit(
                group[FEATURES],
                group["accepted"]
            )

            personalized_weights[
                str(user_id)
            ] = normalize_weights(
                user_model.coef_[0],
                global_weights
            )

        except Exception:

            continue

    return (
        global_weights,
        personalized_weights,
        feature_data
    )


# ------------------------------------------------------------
# Train when the Streamlit app starts
# ------------------------------------------------------------

(
    dynamic_global_weights,
    dynamic_user_weights,
    dynamic_feedback_features
) = learn_dynamic_weights(
    feedback
)

learned_weights = (
    dynamic_global_weights.copy()
)


# ------------------------------------------------------------
# ADAPTIVE SCORE
# ------------------------------------------------------------

def calculate_match_score(
    user_index,
    matched_index
):

    text_score = float(
        text_similarity[
            user_index,
            matched_index
        ]
    )

    mbti_score = float(
        mbti_compatibility(
            users.iloc[
                user_index
            ]["mbti"],

            users.iloc[
                matched_index
            ]["mbti"]
        )
    )

    location_score = float(
        location_compatibility(
            users.iloc[
                user_index
            ]["location"],

            users.iloc[
                matched_index
            ]["location"]
        )
    )

    selected_user_id = str(
        users.iloc[
            user_index
        ]["user_id"]
    )

    weights = dynamic_user_weights.get(
        selected_user_id,
        dynamic_global_weights
    )

    total_score = (

        weights[0] *
        text_score

        +

        weights[1] *
        mbti_score

        +

        weights[2] *
        location_score
    )

    return round(
        total_score * 100,
        2
    )


# ============================================
# ADAPTIVE SCORE
# ============================================

def calculate_match_score(
    user_index,
    matched_index
):

    text_score = text_similarity[
        user_index,
        matched_index
    ]

    mbti_score = mbti_compatibility(
        users.iloc[user_index]["mbti"],
        users.iloc[matched_index]["mbti"]
    )

    location_score = location_compatibility(
        users.iloc[user_index]["location"],
        users.iloc[matched_index]["location"]
    )

    total_score = (

        learned_weights[0] *
        text_score

        +

        learned_weights[1] *
        mbti_score

        +

        learned_weights[2] *
        location_score

    )

    return round(
        total_score * 100,
        2
    )


# ============================================
# TOP MATCHES
# ============================================

def get_top_matches(
    user_id,
    top_n=5
):

    user_index = users.index[
        users["user_id"] == user_id
    ][0]

    matches = []

    for matched_index in range(
        len(users)
    ):

        if matched_index == user_index:
            continue

        score = calculate_match_score(
            user_index,
            matched_index
        )

        matches.append({

            "user_id":
                users.iloc[
                    matched_index
                ]["user_id"],

            "name":
                users.iloc[
                    matched_index
                ]["name"],

            "profession":
                users.iloc[
                    matched_index
                ]["profession"],

            "location":
                users.iloc[
                    matched_index
                ]["location"],

            "mbti":
                users.iloc[
                    matched_index
                ]["mbti"],

            "compatibility":
                score
        })

    result = pd.DataFrame(
        matches
    )

    result = result.sort_values(
        "compatibility",
        ascending=False
    )

    return result.head(top_n)


# ============================================
# SAVE FEEDBACK + RETRAIN
# ============================================

from datetime import datetime


def save_feedback(
    user_id,
    matched_user_id,
    action
):

    global feedback
    global dynamic_global_weights
    global dynamic_user_weights
    global dynamic_feedback_features
    global learned_weights

    new_feedback = pd.DataFrame([
        {
            "user_id": user_id,

            "matched_user_id":
                matched_user_id,

            "action":
                int(action),

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
        }
    ])

    # Save permanently to feedback.csv
    header_needed = not os.path.exists(
        feedback_file
    )

    new_feedback.to_csv(
        feedback_file,
        mode="a",
        header=header_needed,
        index=False
    )

    # Update in-memory feedback
    feedback = pd.concat(
        [
            feedback,
            new_feedback
        ],
        ignore_index=True
    )

    # Retrain weights immediately
    (
        dynamic_global_weights,
        dynamic_user_weights,
        dynamic_feedback_features
    ) = learn_dynamic_weights(
        feedback
    )

    learned_weights = (
        dynamic_global_weights.copy()
    )

    st.session_state.feedback_message = (
        f"Feedback saved for "
        f"{matched_user_id}. "
        f"Matching weights updated."
    )

    # Refresh the app so the new Top 5
    # uses the newly learned weights
    st.rerun()

# ============================================
# USER INTERFACE
# ============================================

st.title(
    "🤝 AI Profile Matching System"
)

st.write(
    "Find compatible profiles using "
    "text similarity, MBTI and location."
)

if "feedback_message" in st.session_state:

    st.success(
        st.session_state.feedback_message
    )

    del st.session_state.feedback_message


# ============================================
# USER SELECTION
# ============================================

selected_user = st.selectbox(
    "Select User",
    users["user_id"].tolist()
)


# ============================================
# USER PROFILE
# ============================================

selected_row = users[
    users["user_id"] == selected_user
].iloc[0]


st.subheader("Selected Profile")

col1, col2, col3 = st.columns(3)

with col1:

    st.write(
        "**Name:**",
        selected_row["name"]
    )

    st.write(
        "**Profession:**",
        selected_row["profession"]
    )


with col2:

    st.write(
        "**Location:**",
        selected_row["location"]
    )

    st.write(
        "**MBTI:**",
        selected_row["mbti"]
    )


with col3:

    st.write(
        "**User ID:**",
        selected_row["user_id"]
    )


# ============================================
# FIND MATCHES
# ============================================

# Store matches so they remain visible after
# clicking Accept or Reject
if "matches" not in st.session_state:
    st.session_state.matches = None


# Find matches button
if st.button(
    "🔍 Find Top 5 Matches",
    type="primary"
):

    st.session_state.matches = get_top_matches(
        selected_user,
        5
    )


# Display matches
if st.session_state.matches is not None:

    matches = st.session_state.matches

    st.subheader(
        "🏆 Top 5 Matches"
    )

    for i, row in matches.iterrows():

        st.markdown(
            f"### {row['name']}"
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.write(
                f"**Compatibility:** "
                f"{row['compatibility']}%"
            )

        with col2:

            st.write(
                f"**Profession:** "
                f"{row['profession']}"
            )

        with col3:

            st.write(
                f"**Location:** "
                f"{row['location']}"
            )

        with col4:

            st.write(
                f"**MBTI:** "
                f"{row['mbti']}"
            )

        accept, reject = st.columns(2)

        with accept:

            if st.button(
                "👍 Accept",
                key=f"accept_{selected_user}_{row['user_id']}"
            ):

                save_feedback(
                    selected_user,
                    row["user_id"],
                    1
                )

                st.success(
                    f"Accepted {row['name']} ✅"
                )

        with reject:

            if st.button(
                "👎 Reject",
                key=f"reject_{selected_user}_{row['user_id']}"
            ):

                save_feedback(
                    selected_user,
                    row["user_id"],
                    0
                )

                st.warning(
                    f"Rejected {row['name']} ❌"
                )

        st.divider()

# ============================================
# CURRENT WEIGHTS
# ============================================

display_weights = (
    dynamic_user_weights.get(
        str(selected_user),
        dynamic_global_weights
    )
)

st.sidebar.header(
    "Learned Matching Weights"
)

st.sidebar.write(
    f"Text Similarity: "
    f"{display_weights[0] * 100:.2f}%"
)

st.sidebar.write(
    f"MBTI Compatibility: "
    f"{display_weights[1] * 100:.2f}%"
)

st.sidebar.write(
    f"Location Compatibility: "
    f"{display_weights[2] * 100:.2f}%"
)
