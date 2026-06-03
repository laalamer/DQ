import streamlit as st
import pandas as pd

from src.data_loader import load_data
from src.profiler import build_dataset_profile
from src.pattern_miner import discover_patterns
from src.rule_discovery import discover_rules
from src.violation_detector import detect_violations


st.set_page_config(
    page_title="AI Data Quality Chatbot",
    page_icon="📊",
    layout="wide"
)


st.title("AI Data Quality Chatbot")
st.write("Upload a CSV or Excel file to discover data quality rules automatically.")


uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx"]
)


if uploaded_file is not None:

    df = load_data(uploaded_file)

    st.success("File uploaded successfully.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # -------------------------------
    # Profile
    # -------------------------------

    if st.button("1. Generate Dataset Profile"):

        profile = build_dataset_profile(df)

        st.session_state["profile"] = profile

        st.subheader("Dataset Profile")
        st.json(profile)

    # -------------------------------
    # Patterns
    # -------------------------------

    if st.button("2. Discover Data Patterns"):

        patterns = discover_patterns(df)

        st.session_state["patterns"] = patterns

        st.subheader("Detected Patterns")

        if patterns:
            st.json(patterns)
        else:
            st.info("No strong patterns were detected.")

    # -------------------------------
    # Rules using LLaMA
    # -------------------------------

    if st.button("3. Discover Data Quality Rules with LLaMA"):

        with st.spinner("LLaMA is evaluating detected patterns..."):

            result = discover_rules(df)

            st.session_state["discovery_result"] = result
            st.session_state["rules"] = result["rules"]

        st.subheader("Approved Data Quality Rules")

        if result["rules"]:
            st.json(result["rules"])
        else:
            st.warning("No approved rules were returned by LLaMA.")

        with st.expander("View Raw LLaMA Response"):
            st.write(result["llm_response"])

    # -------------------------------
    # Violations
    # -------------------------------

    if st.button("4. Detect Violations"):

        rules = st.session_state.get("rules")

        if not rules:
            st.warning("Please discover rules first.")
        else:
            violations = detect_violations(df, rules)

            st.session_state["violations"] = violations

            st.subheader("Violation Results")

            total_rows = len(df)
            total_violations = sum(
                item["violation_count"] for item in violations
            )

            dq_score = max(
                0,
                100 - ((total_violations / max(total_rows, 1)) * 100)
            )

            st.metric("Data Quality Score", f"{dq_score:.2f}%")
            st.metric("Total Violations", total_violations)

            for item in violations:

                st.markdown("---")
                st.markdown(f"### {item['rule_id']} - {item['rule_name']}")
                st.write(item["description"])
                st.write("Violation Count:", item["violation_count"])

                if item["violation_count"] > 0:
                    st.dataframe(
                        item["violating_rows"],
                        use_container_width=True
                    )
                else:
                    st.success("No violations found for this rule.")

else:
    st.info("Please upload a CSV or Excel file to start.")