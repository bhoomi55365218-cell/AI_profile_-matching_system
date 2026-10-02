
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


