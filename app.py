import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="Segmentation Client RFM",
    page_icon="🛍️",
    layout="wide"
)

# ============================================================
# CHARGEMENT DES ARTEFACTS (modèle, scaler, données RFM)
# ============================================================
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("rfm_scaler.pkl", "rb"))
    rfm = pd.read_csv("rfm_clustered.csv")
    # Construire la correspondance Cluster -> Segment à partir des données du notebook
    label_map = rfm.drop_duplicates("Cluster").set_index("Cluster")["Segment"].to_dict()
    return model, scaler, rfm, label_map

model, scaler, rfm, label_map = load_artifacts()

# Description / conseil marketing par segment (cohérent avec la Partie 5 du notebook)
SEGMENT_INFO = {
    "Premium": {
        "emoji": "🏆",
        "couleur": "#2ecc71",
        "description": "Clients qui achètent souvent, récemment, et dépensent beaucoup.",
        "conseil": "Programme VIP, accès anticipé aux nouveautés. Pas de relance agressive nécessaire."
    },
    "Fidèle": {
        "emoji": "⭐",
        "couleur": "#3498db",
        "description": "Clients réguliers, bonne valeur, encore actifs.",
        "conseil": "Maintenir l'engagement : cross-sell, programme de fidélité."
    },
    "Occasionnel": {
        "emoji": "🙂",
        "couleur": "#f1c40f",
        "description": "La majorité des clients : achats peu fréquents et modestes.",
        "conseil": "Campagnes d'activation pour augmenter la fréquence d'achat."
    },
    "À risque": {
        "emoji": "⚠️",
        "couleur": "#e74c3c",
        "description": "Clients non revus depuis longtemps, risque de perte totale.",
        "conseil": "Campagne de réactivation : coupon de réduction, email de relance personnalisé."
    },
}

# ============================================================
# EN-TÊTE
# ============================================================
st.title("🛍️ Segmentation Client — Analyse RFM & K-Means")
st.markdown(
    "Application de prédiction du segment client à partir de son comportement d'achat "
    "(**R**ecency, **F**requency, **M**onetary)."
)
st.divider()

# ============================================================
# INTERFACE — SAISIE DU CLIENT (sidebar)
# ============================================================
st.sidebar.header("📝 Profil du client à analyser")

recency = st.sidebar.number_input(
    "Recency — Jours depuis le dernier achat",
    min_value=0, max_value=730, value=30, step=1
)
frequency = st.sidebar.number_input(
    "Frequency — Nombre d'achats",
    min_value=1, max_value=300, value=3, step=1
)
monetary = st.sidebar.number_input(
    "Monetary — Montant total dépensé (€)",
    min_value=0.0, max_value=300000.0, value=500.0, step=10.0
)

predict_btn = st.sidebar.button("🔍 Prédire le segment", type="primary", use_container_width=True)

st.sidebar.divider()
st.sidebar.caption(
    "Modèle : K-Means (k=4) entraîné sur 4 338 clients réels "
    "(dataset Online Retail, après nettoyage et RFM)."
)

# ============================================================
# PRÉDICTION DU CLUSTER
# ============================================================
if predict_btn:
    nouveau_client = pd.DataFrame([{
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary
    }])

    nouveau_client_scaled = scaler.transform(nouveau_client)
    cluster_predit = model.predict(nouveau_client_scaled)[0]
    segment = label_map[cluster_predit]
    info = SEGMENT_INFO[segment]

    # ---- Affichage du profil client ----
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### {info['emoji']} Segment prédit")
        st.markdown(
            f"<h1 style='color:{info['couleur']}'>{segment}</h1>",
            unsafe_allow_html=True
        )
        st.metric("Recency", f"{recency} jours")
        st.metric("Frequency", f"{frequency} achats")
        st.metric("Monetary", f"{monetary:,.2f} €")

    with col2:
        st.markdown("### 📋 Profil du segment")
        st.info(info["description"])
        st.success(f"**Recommandation marketing :** {info['conseil']}")

        # Comparaison client vs moyenne du cluster
        profil_cluster = rfm[rfm["Cluster"] == cluster_predit][
            ["Recency", "Frequency", "Monetary"]
        ].mean()

        comparaison = pd.DataFrame({
            "Variable": ["Recency", "Frequency", "Monetary"],
            "Client saisi": [recency, frequency, monetary],
            f"Moyenne segment {segment}": [
                round(profil_cluster["Recency"], 1),
                round(profil_cluster["Frequency"], 1),
                round(profil_cluster["Monetary"], 1),
            ]
        })
        st.dataframe(comparaison, hide_index=True, use_container_width=True)

    st.divider()

    # ---- Visualisation graphique : positionnement du client dans le nuage de points ----
    st.markdown("### 📊 Positionnement du client dans la base existante")

    fig = px.scatter(
        rfm, x="Recency", y="Monetary",
        color="Segment",
        size="Frequency",
        opacity=0.5,
        title="Recency vs Monetary — tous les clients (taille = Frequency)",
        color_discrete_map={k: v["couleur"] for k, v in SEGMENT_INFO.items()}
    )
    fig.add_scatter(
        x=[recency], y=[monetary],
        mode="markers",
        marker=dict(size=22, color="black", symbol="star"),
        name="Client saisi"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Renseigne les informations du client dans le panneau de gauche, puis clique sur **Prédire le segment**.")

    # ---- Vue d'ensemble de la base client (page d'accueil) ----
    st.markdown("### 📊 Vue d'ensemble des segments existants (4 338 clients)")

    col1, col2, col3, col4 = st.columns(4)
    repartition = rfm["Segment"].value_counts()
    for col, segment in zip([col1, col2, col3, col4], SEGMENT_INFO.keys()):
        info = SEGMENT_INFO[segment]
        nb = repartition.get(segment, 0)
        pct = nb / len(rfm) * 100
        col.metric(f"{info['emoji']} {segment}", f"{nb} clients", f"{pct:.1f}%")

    fig_overview = px.scatter_3d(
        rfm, x="Recency", y="Frequency", z="Monetary",
        color="Segment",
        opacity=0.6,
        color_discrete_map={k: v["couleur"] for k, v in SEGMENT_INFO.items()},
        title="Vue 3D des 4 segments clients (RFM)"
    )
    st.plotly_chart(fig_overview, use_container_width=True)
