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

def Classificatron3000(df, labels: dict, emb_similarity, rel_scores, top=10):
    def Help():
        text = """
        This is the more recent version of CEPS Data Science Team Classification Tool. Parameters are:
            df: a Pandas DataFrame containing the information to be classified. It must contain the following columns:
                "id": a unique identifier for the row, usually the name of the thing to be classified.
                "description": the whole text to be classified, usually a description of 'id'.
            labels: a dictionary where the keys are the names of the labels for the classification and the values are optional descriptions.
            emb_similiarity: a Pandas DataFrame / a tuple of Pandas DataFrames containing the similarity matrix / the embeddings for both the topics and the labels.
            rel_scores: a Pandas DataFrame containing the relatedness scores.
            top: delimiter of how many features to consider for rankings. Default is 10.
        """
        print(text)

    def GetDefaultCall():
        print("Classificatron3000(df, labels, emb_similarity, rel_scores, top=10)")

    if emb_similarity is None:
        openai.api_key = st.secrets["openai"]["api_key"]
        def get_embedding(text, model="text-embedding-3-large"):
            response = openai.embeddings.create(input=[text], model=model)
            embedding = response.data[0].embedding
            return embedding
        embeddings = df['description'].apply(lambda x: get_embedding(x, model="text-embedding-3-large"))
        embedding_dim = len(embeddings.iloc[0])
        columns = [f"V{i+1}" for i in range(embedding_dim)]
        embeddings_df = pd.DataFrame(embeddings.tolist(), columns=columns)
        emb_similarity_df = pd.concat([df, embeddings_df], axis=1)
        emb_similarity_df.drop(columns=["description"], inplace=True)
        embeddings_data = []
        for label, description in labels.items():
            embedding = get_embedding(description)
            embedding_dict = {"Label": label}
            embedding_dict.update({f"V{i+1}": val for i, val in enumerate(embedding)})
            embeddings_data.append(embedding_dict)
        labels_df = pd.DataFrame(embeddings_data)
        emb_similarity = (emb_similarity_df, labels_df)

    if type(emb_similarity) == tuple:
        features_emb = emb_similarity[0]
        labels_emb = emb_similarity[1]
        features_names = features_emb.iloc[:, 0]
        features_embeddings = features_emb.iloc[:, 2:].values
        labels_names = labels_emb.iloc[:, 0]
        labels_embeddings = labels_emb.iloc[:, 2:].values
        similarity_matrix = cosine_similarity(features_embeddings, labels_embeddings)
        similarity_df = pd.DataFrame(similarity_matrix, index=features_names, columns=labels_names)
        index_name = similarity_df.index.name if similarity_df.index.name is not None else 'index'
        similarities_list = similarity_df.reset_index().melt(id_vars=index_name)
        similarities_list.columns = ['Feature', 'Label', 'Similarity']
        similarities = similarities_list.pivot(index='Feature', columns='Label', values='Similarity')
        similarities.columns.name = None
        similarities.reset_index(inplace=True)
        similarities = similarities[["Feature"] + list(labels.keys())]
    else:
        similarities = emb_similarity
    df_output = similarities.melt(id_vars=["Feature"], var_name="Label", value_name="Similarity")
    df_output['Label'] = pd.Categorical(df_output['Label'], categories=list(labels.keys()), ordered=True)
    df_output = df_output.sort_values(by=["Feature", "Label"]).reset_index(drop=True)
    df_output['Rank1'] = df_output.groupby('Feature')['Similarity'].rank(ascending=False, method='dense').astype(int)
    relatedness = rel_scores
    features_from = relatedness['from'].unique()
    features_to = relatedness['to'].unique()
    all_features = np.union1d(features_from, features_to)
    n = len(all_features)
    feature_to_index = {feature: idx for idx, feature in enumerate(all_features)}
    relatedness_matrix = np.zeros((n, n))
    for _, row in relatedness.iterrows():
        i = feature_to_index[row['from']]
        j = feature_to_index[row['to']]
        relatedness_matrix[i, j] = row['relatedness']
    relatedness_df = pd.DataFrame(relatedness_matrix, index=all_features, columns=all_features)
    similarity = similarities.set_index("Feature")
    similarity = similarity.reindex(all_features)
    similarity = similarity.fillna(0)
    A = relatedness_df.values
    B = similarity.values
    result_matrix = np.dot(A, B)
    result_df = pd.DataFrame(result_matrix, index=all_features, columns=similarity.columns)
    result_df = result_df.reset_index()
    result_df = result_df.rename(columns={"index": "Feature"})
    df_aux = result_df.melt(id_vars=["Feature"], var_name="Label", value_name="Relatedness")
    df_aux['Label'] = pd.Categorical(df_aux['Label'], categories=list(labels.keys()), ordered=True)
    df_aux = df_aux.sort_values(by=["Feature", "Label"]).reset_index(drop=True)
    df_output = df_output.merge(df_aux[['Feature', 'Label', 'Relatedness']], on=['Feature', 'Label'], how='left')
    df_output['Relatedness'] = df_output['Relatedness'].fillna(0)
    df_output['Rank2'] = df_output.groupby('Feature')['Relatedness'].rank(ascending=False, method='dense').astype(int)
    df_output['CombinedRank'] = (df_output['Rank1'] + df_output['Rank2']) / 2
    df_output['Rank3'] = df_output.groupby('Feature')['CombinedRank'].rank(method='first').astype(int)
    df_output = df_output.drop(columns=['CombinedRank'])
    new_df = df_output.copy()
    new_df = new_df[new_df["Rank3"] <= top]
    new_df = new_df.merge(df[["id", "description"]], left_on="Feature", right_on="id", how="left")
    feature_dict = {t: [] for t in new_df["Feature"].unique()}
    for index, row in new_df.iterrows():
        feature_dict[row["Feature"]].append(row["Label"] + f" ({labels[row['Label']]})")
    context = f"""
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
    """
    question = """
    Given this context, provide your answer in the instructed format for this Feature: 
    """
    send_df = new_df[["Feature", "description"]]
    send_df = send_df.drop_duplicates().reset_index()
    send_df["id"] = send_df.index
    send_df["Labels"] = send_df["Feature"].map(feature_dict)
    send_df["Labels"] = send_df["Labels"].apply(lambda x: "; ".join(x))
    batch = BatchUploader(send_df, "id", "description", context, question, label_col="Labels",
                          model="gpt-4o", max_tokens=2000, temp=0.0, description="Classificatron3000 classification PART 1", path="C:/Users/HP/downloads/Batches")
    status = ""
    while status != "completed" and status != "failed":
        time.sleep(5)
        a = BatchChecker(batch.id)
        status = a.status
    send_df = BatchRetriever(send_df, "id", "C:/Users/HP/downloads/output.txt")
    df_output["RankGPT"] = None
    for _, send_row in send_df.iterrows():
        feature = send_row["Feature"]
        response_dict = "ERROR"
        try:
            response_dict = ast.literal_eval(send_row["response"])
        except (ValueError, SyntaxError):
            pass
        matching_rows = df_output[df_output["Feature"] == feature]
        for idx, current_row in matching_rows.iterrows():
            label = current_row["Label"]
            if label in response_dict:
                df_output.at[idx, "RankGPT"] = response_dict[label]
    context = """
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
    """
    question = """
    Given this context, provide your answer in the instructed format for this Feature: 
    """
    send_df = new_df[["Feature", "description"]]
    send_df = send_df.drop_duplicates().reset_index()
    send_df["id"] = send_df.index
    send_df["Labels"] = send_df["Feature"].map(feature_dict)
    send_df["Labels"] = send_df["Labels"].apply(lambda x: "; ".join(x))
    batch = BatchUploader(send_df, "id", "description", context, question, label_col="Labels",
                          model="gpt-4o", max_tokens=2000, temp=0.0, description="Classificatron3000 classification PART 2", path="C:/Users/HP/downloads/Batches")
    status = ""
    while status != "completed" and status != "failed":
        time.sleep(5)
        a = BatchChecker(batch.id)
        status = a.status
    send_df = BatchRetriever(send_df, "id", "C:/Users/HP/downloads/output.txt")
    df_output["FinalCheck"] = None
    for _, send_row in send_df.iterrows():
        feature = send_row["Feature"]
        try:
            response_dict = ast.literal_eval(send_row["response"])
        except (ValueError, SyntaxError):
            continue
        matching_rows = df_output[df_output["Feature"] == feature]
        for idx, current_row in matching_rows.iterrows():
            label = current_row["Label"]
            if label in response_dict:
                df_output.at[idx, "FinalCheck"] = response_dict[label]
    try:
        df_output.to_csv("C:/Users/HP/downloads/Classification.csv", index=False)
    except:
        df_output.to_csv("C:/Users/HP/downloads/Classification.csv", index=False)
    final = df_output[df_output["FinalCheck"] == 1]
    final.to_csv("C:/Users/HP/downloads/ClassificationFiltered.csv", index=False)
    dict_result = {}
    for i in df["id"].unique():
        try:
            dict_result[i] = final[final["Feature"] == i].sort_values(by=["RankGPT"], ascending=True).head(1).iloc[0]["Label"]
        except:
            dict_result[i] = "None"
    classif = pd.DataFrame(data=list(dict_result.items()), columns=["Features", "Labels"])
    classif.to_csv("C:/Users/HP/downloads/Crosswalk.csv", index=False)

def main():
    st.title("Classificatron3000")

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
        "  1) **Feature Embeddings**: must contain columns like `[id, ..., V1, V2, ...]` "
        "     (where `id` and possibly `description` are present, then the embedding columns `V1, V2, ...`).\n"
        "  2) **Label Embeddings**: must contain `[Label, V1, V2, ...]` "
        "     (label name and the embedding vectors)."
    )

    emb_data = None

    # If user chooses "CSV" => we expect a single CSV with the precomputed similarity matrix
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

    # If user chooses "Tuple CSV" => 2 CSVs for (features embeddings, labels embeddings)
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
            emb_data = (pd.read_csv(uploaded_tuple_1), pd.read_csv(uploaded_tuple_2))

    # RUN CLASSIFICATION
    if st.button("Run Classification"):
        if uploaded_df is not None and uploaded_rel is not None:
            df = pd.read_csv(uploaded_df)
            rel_scores = pd.read_csv(uploaded_rel)

            # Convert text area to dictionary
            try:
                labels_dict = ast.literal_eval(labels_text)
            except Exception as e:
                st.error("Error in Labels Dictionary. Make sure it's valid JSON.")
                return

            # If embeddings set to None in the dropdown
            if emb_option == "None":
                emb_data = None

            Classificatron3000(
                df,
                labels_dict,
                emb_data,
                rel_scores,
                top=top_value
            )
            st.success("Classification completed. Check your output files.")
        else:
            st.error("Please upload both main DataFrame and relatedness CSV files.")

if __name__ == "__main__":
    main()
