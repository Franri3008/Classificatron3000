import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import ast
import time
import openai

import sys
sys.path.append("C:/Users/HP/Python/Jupyter Notebook/CustomLibs/LLM/")
import LanguageModels
import importlib
from openai import OpenAI
importlib.reload(LanguageModels)
from LanguageModels import CallLLM, BatchUploader, BatchChecker, BatchRetriever

import os
os.system("pip install --upgrade scikit-learn")


def Classificatron3000(
    df,
    labels: dict,
    emb_similarity,
    rel_scores,
    top=10,
    context_prompt_1=None,
    question_prompt_1=None,
    context_prompt_2=None,
    question_prompt_2=None
):
    """
    Extended version of CEPS Data Science Team Classification Tool.

    New Features:
      1) Shows classification status in Streamlit (progress indicator).
      2) Adds reversed ranking (label->feature).
      3) Adds 'av' = average rank in the forward direction.
      4) Adds 'av_r' = average rank in the reversed direction.

    Parameters:
        df: DataFrame with columns:
            - "id": unique identifier
            - "description": text to classify
        labels: dict { label_name: label_description }
        emb_similarity: None (compute embeddings) OR
            single DataFrame (similarity matrix) OR
            tuple (features_embeddings_df, labels_embeddings_df)
        rel_scores: DataFrame with ['from','to','relatedness']
        top: how many labels per feature to keep in forward ranking
        context_prompt_1, question_prompt_1: first classification step
        context_prompt_2, question_prompt_2: second classification step
    """

    # 1) EMBEDDINGS
    if emb_similarity is None:
        openai.api_key = st.secrets["openai"]["api_key"]

        def get_embedding(text, model="text-embedding-3-large"):
            response = openai.embeddings.create(input=[text], model=model)
            return response.data[0].embedding

        # Compute embeddings for main DataFrame
        embeddings = df["description"].apply(get_embedding)
        embedding_dim = len(embeddings.iloc[0])
        columns = [f"V{i+1}" for i in range(embedding_dim)]
        embeddings_df = pd.DataFrame(embeddings.tolist(), columns=columns)
        emb_similarity_df = pd.concat([df, embeddings_df], axis=1)
        emb_similarity_df.drop(columns=["description"], inplace=True)

        # Compute embeddings for label descriptions
        label_rows = []
        for lbl, desc in labels.items():
            vec = get_embedding(desc)
            row = {"Label": lbl}
            row.update({f"V{i+1}": v for i, v in enumerate(vec)})
            label_rows.append(row)
        labels_df = pd.DataFrame(label_rows)

        emb_similarity = (emb_similarity_df, labels_df)

    # 2) SIMILARITY MATRIX
    if isinstance(emb_similarity, tuple):
        feat_df, lbl_df = emb_similarity
        feat_names = feat_df.iloc[:, 0]         # e.g., "id"
        feat_vectors = feat_df.iloc[:, 2:].values

        label_names = lbl_df.iloc[:, 0]         # e.g., "Label"
        label_vectors = lbl_df.iloc[:, 2:].values

        sim_matrix = cosine_similarity(feat_vectors, label_vectors)
        sim_df = pd.DataFrame(sim_matrix, index=feat_names, columns=label_names)

        # pivot => Feature, Label, Similarity
        sim_list = (
            sim_df.reset_index()
            .melt(id_vars=sim_df.index.name or "index")
            .rename(columns={"index": "Feature", "variable": "Label", "value": "Similarity"})
        )
        similarities = sim_list.pivot("Feature", "Label", "Similarity").reset_index()
        # Reorder label columns in the same order as your labels dict
        similarities = similarities[["Feature"] + list(labels.keys())]
    else:
        # It's a single DataFrame with [Feature, label1, label2, ...]
        similarities = emb_similarity

    # 3) FORWARD RANKING
    # Melt so each row is (Feature, Label, Similarity)
    df_output = similarities.melt(
        id_vars=["Feature"],
        var_name="Label",
        value_name="Similarity"
    )
    # Ensure label order is consistent
    df_output["Label"] = pd.Categorical(df_output["Label"], categories=list(labels.keys()), ordered=True)
    df_output.sort_values(by=["Feature", "Label"], inplace=True)
    df_output.reset_index(drop=True, inplace=True)

    # Rank by Similarity => Rank1
    df_output["Rank1"] = df_output.groupby("Feature")["Similarity"].rank(
        ascending=False, method="dense"
    ).astype(int)

    # Build relatedness NxN
    from_vals = rel_scores["from"].unique()
    to_vals = rel_scores["to"].unique()
    all_feats = np.union1d(from_vals, to_vals)
    feat_idx_map = {f: i for i, f in enumerate(all_feats)}
    n = len(all_feats)
    rel_mat = np.zeros((n, n))
    for _, row in rel_scores.iterrows():
        i = feat_idx_map[row["from"]]
        j = feat_idx_map[row["to"]]
        rel_mat[i, j] = row["relatedness"]

    rel_df = pd.DataFrame(rel_mat, index=all_feats, columns=all_feats)

    # Align similarity with all_feats
    sim_pivot = similarities.set_index("Feature").reindex(all_feats).fillna(0)
    A = rel_df.values
    B = sim_pivot.values
    result_mat = A.dot(B)

    res_df = pd.DataFrame(result_mat, index=all_feats, columns=sim_pivot.columns).reset_index()
    res_df.rename(columns={"index": "Feature"}, inplace=True)

    aux = res_df.melt(
        id_vars=["Feature"],
        var_name="Label",
        value_name="Relatedness"
    )
    aux["Label"] = pd.Categorical(aux["Label"], categories=list(labels.keys()), ordered=True)
    aux.sort_values(by=["Feature", "Label"], inplace=True)
    aux.reset_index(drop=True, inplace=True)

    # Merge => Rank2
    df_output = df_output.merge(
        aux[["Feature", "Label", "Relatedness"]],
        on=["Feature", "Label"],
        how="left"
    )
    df_output["Relatedness"].fillna(0, inplace=True)
    df_output["Rank2"] = df_output.groupby("Feature")["Relatedness"].rank(
        ascending=False, method="dense"
    ).astype(int)

    # Combined => Rank3
    df_output["CombinedRank"] = (df_output["Rank1"] + df_output["Rank2"]) / 2
    df_output["Rank3"] = df_output.groupby("Feature")["CombinedRank"].rank(method="first").astype(int)
    df_output.drop(columns=["CombinedRank"], inplace=True)

    # Keep top = N
    df_top = df_output[df_output["Rank3"] <= top].copy()

    # Merge text description
    df_top = df_top.merge(df[["id", "description"]], left_on="Feature", right_on="id", how="left")

    # Build label strings => "Label (LabelDesc)"
    feature_dict = {f: [] for f in df_top["Feature"].unique()}
    for i, row in df_top.iterrows():
        feature_dict[row["Feature"]].append(f"{row['Label']} ({labels[row['Label']]})")

    # 4) FIRST CLASSIFICATION PROMPT
    # Use user-provided or fallback
    if not context_prompt_1:
        context_prompt_1 = f"""
        Your task is to rank labels relevance to a specific feature based on how heavily the label draws on technical knowledge from this specific feature.
        The available labels are:

        <\\labels\\>

        Ranking Rules:
        - Use a scale of 1-{top} where 1 = most relevant label of the list, and {top} = least relevant label of the list.
        - Base rankings on:
          * Direct application
          * Technical overlap with the feature
          * Relevance
          * Label's reliance on the feature's core principles
          * Integration in core products
          * Significance of the label for the feature
        - Ties are NOT allowed

        Output Format if you had 3 labels (label names must be between quotes):

            {{"Label1": rank, "Label2": rank, "Label3": rank}}

        Don't add anything else to your response.
        """.strip()
    if not question_prompt_1:
        question_prompt_1 = "Given this context, provide your answer in the instructed format for this Feature:"

    # Build PART 1 batch
    send_df = df_top[["Feature", "description"]].drop_duplicates().reset_index(drop=True)
    send_df["id"] = send_df.index
    send_df["Labels"] = send_df["Feature"].map(feature_dict).apply(lambda x: "; ".join(x))

    st.write("**Starting GPT Classification (Step 1)**")
    batch1 = BatchUploader(
        send_df,
        id_col="id",
        inf_col="description",
        role=context_prompt_1,
        question=question_prompt_1,
        label_col="Labels",
        model="gpt-4o",
        max_tokens=2000,
        temp=0.0,
        description="Classification PART 1",
        path="C:/Users/HP/downloads/Batches"
    )

    # Show classification progress in Streamlit
    status_placeholder = st.empty()
    while True:
        time.sleep(5)
        check = BatchChecker(batch1.id)
        status_placeholder.write(f"**Step 1 Batch Status**: {check.status}")
        if check.status in ["completed", "failed"]:
            break

    # Retrieve => RankGPT
    send_df = BatchRetriever(send_df, "id", "C:/Users/HP/downloads/output.txt")
    df_output["RankGPT"] = None
    for _, row_ in send_df.iterrows():
        feat = row_["Feature"]
        try:
            resp_dict = ast.literal_eval(row_["response"])
        except (ValueError, SyntaxError):
            resp_dict = {}
        mask = df_output["Feature"] == feat
        for idx in df_output[mask].index:
            lab = df_output.at[idx, "Label"]
            if lab in resp_dict:
                df_output.at[idx, "RankGPT"] = resp_dict[lab]

    # 5) SECOND CLASSIFICATION PROMPT
    if not context_prompt_2:
        context_prompt_2 = """
        Your task is to determine whether a Label is relevant to a specific Feature by thinking in how heavily the Label draws on technical knowledge from this specific Feature. 
        Available Labels:

        <\\labels\\>

        Rules:
        - Answer 0 if the Label is not relevant, 1 if it is.
        - Base your answers on:
          * Direct application
          * Technical overlap with the feature
          * Relevance
          * Label's reliance on the feature's core principles
          * Integration in core products
          * Significance of the label for the feature
        -You'll gain a point for every Label that you get correctly classified, but will lose 10 points for every Label incorrectly classified as 1.
        -Aim to get the highest amount of points.

        Output Format if you had 3 labels (label names must be between quotes):

            {"Label1": response, "Label2": response, "Label3": response}

        Don't add anything else to your response.
        """.strip()
    if not question_prompt_2:
        question_prompt_2 = "Given this context, provide your answer in the instructed format for this Feature:"

    send_df = df_top[["Feature", "description"]].drop_duplicates().reset_index(drop=True)
    send_df["id"] = send_df.index
    send_df["Labels"] = send_df["Feature"].map(feature_dict).apply(lambda x: "; ".join(x))

    st.write("**Starting GPT Classification (Step 2)**")
    batch2 = BatchUploader(
        send_df,
        id_col="id",
        inf_col="description",
        role=context_prompt_2,
        question=question_prompt_2,
        label_col="Labels",
        model="gpt-4o",
        max_tokens=2000,
        temp=0.0,
        description="Classification PART 2",
        path="C:/Users/HP/downloads/Batches"
    )

    # Show classification progress
    while True:
        time.sleep(5)
        check = BatchChecker(batch2.id)
        status_placeholder.write(f"**Step 2 Batch Status**: {check.status}")
        if check.status in ["completed", "failed"]:
            break

    # Retrieve => FinalCheck
    send_df = BatchRetriever(send_df, "id", "C:/Users/HP/downloads/output.txt")
    df_output["FinalCheck"] = None
    for _, row_ in send_df.iterrows():
        feat = row_["Feature"]
        try:
            resp_dict = ast.literal_eval(row_["response"])
        except (ValueError, SyntaxError):
            resp_dict = {}
        mask = df_output["Feature"] == feat
        for idx in df_output[mask].index:
            lab = df_output.at[idx, "Label"]
            if lab in resp_dict:
                df_output.at[idx, "FinalCheck"] = resp_dict[lab]

    # 6) ADD 'av' = average forward rank
    df_output["RankGPT"] = pd.to_numeric(df_output["RankGPT"], errors="coerce")
    df_output["av"] = df_output[["Rank1", "Rank2", "Rank3", "RankGPT"]].mean(axis=1, numeric_only=True)

    # 7) REVERSED RANK => label->feature
    # Copy df_output => rename columns
    df_rev = df_output.copy()
    df_rev.rename(
        columns={
            "Feature": "Label_rev",
            "Label": "Feature_rev",
            "Rank1": "Rank1_r",
            "Rank2": "Rank2_r",
            "Rank3": "Rank3_r",
            "RankGPT": "RankGPT_r",
            "Similarity": "Similarity_r",
            "Relatedness": "Relatedness_r",
        },
        inplace=True
    )
    # Re-rank grouping by "Feature_rev" (the new "group")
    df_rev["Rank1_r"] = df_rev.groupby("Feature_rev")["Similarity_r"]\
        .rank(ascending=False, method="dense").astype(int)
    df_rev["Rank2_r"] = df_rev.groupby("Feature_rev")["Relatedness_r"]\
        .rank(ascending=False, method="dense").astype(int)
    df_rev["Rank3_r"] = ((df_rev["Rank1_r"] + df_rev["Rank2_r"]) / 2).astype(int)
    # For GPT => ascending smaller = more relevant
    df_rev["RankGPT_r"] = df_rev.groupby("Feature_rev")["RankGPT_r"]\
        .rank(ascending=True, method="dense")

    # average => av_r
    df_rev["av_r"] = df_rev[["Rank1_r", "Rank2_r", "Rank3_r", "RankGPT_r"]].mean(axis=1)

    # Merge av_r back to df_output
    merged = pd.merge(
        df_output,
        df_rev[["Label_rev","Feature_rev","av_r"]],
        left_on=["Feature","Label"],
        right_on=["Label_rev","Feature_rev"],
        how="left"
    ).drop(["Label_rev","Feature_rev"], axis=1)

    df_output = merged  # update

    # 8) SAVE FINAL FILES
    try:
        df_output.to_csv("C:/Users/HP/downloads/Classification.csv", index=False)
    except:
        df_output.to_csv("C:/Users/HP/downloads/Classification.csv", index=False)

    final = df_output[df_output["FinalCheck"] == 1]
    final.to_csv("C:/Users/HP/downloads/ClassificationFiltered.csv", index=False)

    # Crosswalk => "id" -> top label among final
    dict_result = {}
    for fid in df["id"].unique():
        sub = final[final["Feature"] == fid].sort_values(by=["RankGPT"], ascending=True)
        try:
            dict_result[fid] = sub.head(1).iloc[0]["Label"]
        except:
            dict_result[fid] = "None"

    cross = pd.DataFrame(data=list(dict_result.items()), columns=["Features","Labels"])
    cross.to_csv("C:/Users/HP/downloads/Crosswalk.csv", index=False)


def main():
    st.title("Classificatron3000 v2.0 (Reversed Rank, av & av_r)")

    # 1) Main DataFrame
    st.subheader("1) Main DataFrame CSV")
    st.markdown(
        "**Required Columns:**\n"
        "- `id` (unique identifier for each row)\n"
        "- `description` (text content that will be classified)"
    )
    uploaded_df = st.file_uploader("Upload the main DataFrame CSV", type=["csv"])

    # 2) Relatedness Scores
    st.subheader("2) Relatedness Scores CSV")
    st.markdown(
        "**Required Columns:**\n"
        "- `from` (source feature)\n"
        "- `to` (target feature)\n"
        "- `relatedness` (numeric score for how related the two features are)"
    )
    uploaded_rel = st.file_uploader("Upload the relatedness scores CSV", type=["csv"])

    # 3) Labels Dictionary
    st.subheader("3) Labels Dictionary (JSON format)")
    st.markdown(
        "**Required Format:** A JSON object where...\n"
        "- **key**: label name\n"
        "- **value**: label description\n\n"
        "Example: `{\"LabelA\": \"DescA\", \"LabelB\": \"DescB\"}`"
    )
    labels_text = st.text_area(
        "Labels dictionary:",
        value='{"LabelA": "DescriptionA", "LabelB": "DescriptionB"}'
    )

    # 4) Top Value
    top_value = st.number_input("Top value:", value=10)

    # 5) Embedding Option
    st.subheader("4) Embedding Data Option")
    emb_option = st.selectbox(
        "Choose embedding option:",
        ["None", "CSV", "Tuple CSV"]
    )
    st.markdown(
        "**Options:**\n"
        "- **None**: No pre-computed embeddings or similarity matrix. "
        "Classificatron3000 will generate embeddings automatically (using GPT API).\n"
        "- **CSV**: Single CSV containing **similarity matrix** with columns:\n"
        "  - `Feature` (the feature name)\n"
        "  - One column for each label from the labels dict.\n"
        "- **Tuple CSV**: Two CSVs representing **feature embeddings** and **label embeddings**:\n"
        "  1) **Feature Embeddings**: must contain `[id, ..., V1, V2, ...]`\n"
        "  2) **Label Embeddings**: must contain `[Label, V1, V2, ...]`"
    )

    emb_data = None

    # Option: CSV with similarity
    if emb_option == "CSV":
        st.subheader("5) Similarity Matrix CSV")
        st.markdown(
            "**Required Columns:**\n"
            "- `Feature` (the feature name)\n"
            "- One column for each label specified in your labels dictionary"
        )
        uploaded_emb = st.file_uploader("Upload the similarity matrix CSV", type=["csv"])
        if uploaded_emb is not None:
            emb_data = pd.read_csv(uploaded_emb)

    # Option: Tuple CSV => features embeddings + labels embeddings
    elif emb_option == "Tuple CSV":
        st.subheader("5) Feature Embeddings CSV")
        st.markdown(
            "**Required Columns:**\n"
            "- `id` (unique identifier)\n"
            "- Optionally `description`\n"
            "- Embedding columns like `V1, V2, ..., Vn`"
        )
        uploaded_tuple_1 = st.file_uploader("Upload the first CSV (features embeddings)", type=["csv"])

        st.subheader("6) Label Embeddings CSV")
        st.markdown(
            "**Required Columns:**\n"
            "- `Label` (unique label name)\n"
            "- Embedding columns like `V1, V2, ..., Vn`"
        )
        uploaded_tuple_2 = st.file_uploader("Upload the second CSV (labels embeddings)", type=["csv"])

        if uploaded_tuple_1 is not None and uploaded_tuple_2 is not None:
            emb_data = (
                pd.read_csv(uploaded_tuple_1),
                pd.read_csv(uploaded_tuple_2)
            )

    # 6) Custom Prompts
    st.subheader("5) Custom Prompts")
    st.markdown(
        "Below are **two sets of prompts** used in two classification steps.\n"
        "The special placeholder `<\\labels\\>` will be replaced by your label dictionary."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### First Classification Prompts")
        default_context_1 = f"""
        Your task is to rank labels relevance to a specific feature based on how heavily the label draws on technical knowledge from this specific feature.
        The available labels are:

        <\\labels\\>

        Ranking Rules:
        - Use a scale of 1-{top_value} where 1 = most relevant label, and {top_value} = least relevant.
        - Base rankings on:
          * Direct application
          * Technical overlap
          * Relevance
          * Reliance on core principles
          * Integration in core products
          * Significance
        - Ties are NOT allowed

        Output Format (for 3 labels):

            {{"Label1": 1, "Label2": 2, "Label3": 3}}
        """.strip()
        default_question_1 = "Given this context, provide your answer in the instructed format for this Feature:"
        context_prompt_1 = st.text_area("Context Prompt 1:", value=default_context_1, height=400)
        question_prompt_1 = st.text_area("Question Prompt 1:", value=default_question_1, height=68)

    with col2:
        st.markdown("#### Second Classification Prompts")
        default_context_2 = """
        Your task is to determine whether a Label is relevant to a specific Feature by thinking in how heavily the Label draws on technical knowledge from this specific Feature.
        <\\labels\\>

        Rules:
        - Answer 0 if not relevant, 1 if relevant.
        - Base on:
          * Direct application
          * Technical overlap
          * Relevance
          * Reliance on core principles
          * Integration
          * Significance
        - You lose points for false positives
        - Aim for maximum correctness

        Output Format (for 3 labels):
            {"Label1": 1, "Label2": 0, "Label3": 1}
        """.strip()
        default_question_2 = "Given this context, provide your answer in the instructed format for this Feature:"
        context_prompt_2 = st.text_area("Context Prompt 2:", value=default_context_2, height=400)
        question_prompt_2 = st.text_area("Question Prompt 2:", value=default_question_2, height=68)

    # 7) RUN CLASSIFICATION
    if st.button("Run Classification"):
        if uploaded_df is not None and uploaded_rel is not None:
            df_main = pd.read_csv(uploaded_df)
            df_rel = pd.read_csv(uploaded_rel)

            # parse labels
            try:
                labels_dict = ast.literal_eval(labels_text)
            except Exception as e:
                st.error("Error in Labels Dictionary. Make sure it's valid JSON.")
                return

            if emb_option == "None":
                emb_data = None

            Classificatron3000(
                df=df_main,
                labels=labels_dict,
                emb_similarity=emb_data,
                rel_scores=df_rel,
                top=top_value,
                context_prompt_1=context_prompt_1,
                question_prompt_1=question_prompt_1,
                context_prompt_2=context_prompt_2,
                question_prompt_2=question_prompt_2
            )
            st.success("Classification completed. Check your output files.")
        else:
            st.error("Please upload both main DataFrame and relatedness CSV files.")


if __name__ == "__main__":
    main()