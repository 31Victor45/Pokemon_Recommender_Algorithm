import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os
import base64 

import streamlit as st 

# Configuración de la página (DEBE SER LO PRIMERO QUE LLAMES DE STREAMLIT)
st.set_page_config(
    page_title="Recomendador de Pokémon",
    page_icon="icon_pokeball.png",
    layout="centered"
)

# --- Parte 1: Funciones del Recomendador de Pokémon ---

@st.cache_data 
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Type 2'] = df['Type 2'].fillna('None')
    df_processed = df.copy()
    df_processed = pd.get_dummies(df_processed, columns=['Type 1', 'Type 2'], prefix=['Type1', 'Type2'], dummy_na=False)
    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    scaler = StandardScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    return df, df_processed, scaler, numeric_cols

DATA_FILE = 'p_items/pkmn.csv' 
IMAGE_DIR = 'p_items/img' 

df, df_processed, scaler, numeric_cols = load_and_preprocess_data(DATA_FILE)

def recommend_pokemons(df, df_processed, scaler, numeric_cols, user_type1, user_type2=None, user_hp=None, user_attack=None, user_defense=None, user_sp_atk=None, user_sp_def=None, user_speed=None, top_n=3):
    user_data = {
        'HP': [user_hp if user_hp is not None else df['HP'].mean()],
        'Attack': [user_attack if user_attack is not None else df['Attack'].mean()],
        'Defense': [user_defense if user_defense is not None else df['Defense'].mean()],
        'Sp. Atk': [user_sp_atk if user_sp_atk is not None else df['Sp. Atk'].mean()],
        'Sp. Def': [user_sp_def if user_sp_def is not None else df['Sp. Def'].mean()],
        'Speed': [user_speed if user_speed is not None else df['Speed'].mean()]
    }
    user_df = pd.DataFrame(user_data)
    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])

    pokemon_features_df = df_processed.drop(columns=['Name', '#'], errors='ignore') 
    user_features = pd.DataFrame(0, index=[0], columns=pokemon_features_df.columns)

    type1_col = f'Type1_{user_type1}'
    if type1_col in user_features.columns:
        user_features[type1_col] = 1
    else:
        pass 

    if user_type2 and user_type2 != 'None':
        type2_col = f'Type2_{user_type2}'
        if type2_col in user_features.columns:
            user_features[type2_col] = 1
        else:
            pass 

    for col in numeric_cols:
        user_features[col] = user_df[col].iloc[0]

    similarities = cosine_similarity(user_features, pokemon_features_df)
    similar_pokemon_internal_indices = similarities.argsort()[0][::-1]

    final_recommended_internal_indices = []
    
    temp_type1_matches = []
    for internal_idx in similar_pokemon_internal_indices:
        if df.loc[internal_idx, 'Type 1'] == user_type1:
            temp_type1_matches.append(internal_idx)
    
    if user_type2 and user_type2 != 'None':
        temp_type1_type2_matches = []
        temp_type1_only_matches = []
        for internal_idx in temp_type1_matches:
            if df.loc[internal_idx, 'Type 2'] == user_type2:
                temp_type1_type2_matches.append(internal_idx)
            else:
                temp_type1_only_matches.append(internal_idx)
        
        final_recommended_internal_indices.extend(temp_type1_type2_matches[:top_n])
        
        if len(final_recommended_internal_indices) < top_n:
            remaining_needed = top_n - len(final_recommended_internal_indices)
            final_recommended_internal_indices.extend(temp_type1_only_matches[:remaining_needed])
            
    else:
        final_recommended_internal_indices.extend(temp_type1_matches[:top_n])

    if len(final_recommended_internal_indices) < top_n:
        remaining_needed = top_n - len(final_recommended_internal_indices)
        count_added = 0
        for internal_idx in similar_pokemon_internal_indices:
            if internal_idx not in final_recommended_internal_indices:
                final_recommended_internal_indices.append(internal_idx)
                count_added += 1
            if count_added >= remaining_needed:
                break
    
    final_recommended_internal_indices = list(dict.fromkeys(final_recommended_internal_indices))[:top_n]

    recommended_pokemons = df.loc[final_recommended_internal_indices].copy()
    recommended_pokemons['Similarity'] = [similarities[0][idx] for idx in final_recommended_internal_indices]

    return recommended_pokemons


# --- Parte 2: Interfaz de Usuario con Streamlit ---

# Función auxiliar para obtener la imagen en base64
def get_image_as_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    return "" 

# Obtener la imagen de la pokebola en base64
pokeball_base64 = get_image_as_base64("p_items/icon_pokeball.png")

st.markdown(
    f"""
    <style>
    .title-container {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .title-text {{
        font-size: 2.5em;
        font-weight: bold;
    }}
    .pokeball-icon {{
        height: 40px;
        width: 40px;
    }}
    /* NUEVA CLASE CSS PARA ALINEACIÓN DE TIPOS */
    .pokemon-types-container {{
        line-height: 1.5; /* Espaciado entre líneas para los tipos */
    }}
    .pokemon-type-label {{
        font-weight: bold;
        display: inline-block; /* Para que "Type 1:" y "Type 2:" tomen el espacio necesario */
        width: 60px; /* Ancho fijo para alinear los valores. Ajusta si es necesario */
    }}
    </style>
    <div class="title-container">
        <h1 class="title-text">Recomendador de Pokémon</h1>
        <img src="data:image/png;base64,{pokeball_base64}" class="pokeball-icon">
    </div>
    """, 
    unsafe_allow_html=True
)

# Sidebar para los controles de entrada
st.sidebar.header("Escoge las características")

all_types = sorted(list(set(df['Type 1'].unique()).union(set(df['Type 2'].dropna().unique()))))
all_types.insert(0, 'None') 

user_type1 = st.sidebar.selectbox("Type 1 (Obligatorio):", sorted(df['Type 1'].unique().tolist()))
user_type2 = st.sidebar.selectbox("Type 2 (Opcional):", all_types, index=all_types.index('None'))

min_hp, max_hp = int(df['HP'].min()), int(df['HP'].max())
user_hp = st.sidebar.slider("HP", min_value=min_hp, max_value=max_hp, value=int(df['HP'].mean()))

min_attack, max_attack = int(df['Attack'].min()), int(df['Attack'].max())
user_attack = st.sidebar.slider("Attack", min_value=min_attack, max_value=max_attack, value=int(df['Attack'].mean()))

min_defense, max_defense = int(df['Defense'].min()), int(df['Defense'].max())
user_defense = st.sidebar.slider("Defense", min_value=min_defense, max_value=max_defense, value=int(df['Defense'].mean()))

min_sp_atk, max_sp_atk = int(df['Sp. Atk'].min()), int(df['Sp. Atk'].max())
user_sp_atk = st.sidebar.slider("Sp. Atk", min_value=min_sp_atk, max_value=max_sp_atk, value=int(df['Sp. Atk'].mean()))

min_sp_def, max_sp_def = int(df['Sp. Def'].min()), int(df['Sp. Def'].max())
user_sp_def = st.sidebar.slider("Sp. Def", min_value=min_sp_def, max_value=max_sp_def, value=int(df['Sp. Def'].mean()))

min_speed, max_speed = int(df['Speed'].min()), int(df['Speed'].max())
user_speed = st.sidebar.slider("Speed", min_value=min_speed, max_value=max_speed, value=int(df['Speed'].mean()))

if st.sidebar.button("Buscar Pokémon"):
    st.subheader("Resultados de la Recomendación:")
    
    recommended_pokemons_df = recommend_pokemons(
        df, df_processed, scaler, numeric_cols,
        user_type1=user_type1,
        user_type2=user_type2 if user_type2 != 'None' else None,
        user_hp=user_hp,
        user_attack=user_attack,
        user_defense=user_defense,
        user_sp_atk=user_sp_atk,
        user_sp_def=user_sp_def,
        user_speed=user_speed
    )

    if not recommended_pokemons_df.empty:
        for i, row in recommended_pokemons_df.iterrows():
            col1, col2 = st.columns([1, 2])

            with col1:
                pokemon_id = int(row['#'])
                image_path = os.path.join(IMAGE_DIR, f"{pokemon_id}.jpg")
                
                if os.path.exists(image_path):
                    st.image(image_path, caption=f"ID: {pokemon_id}", width=150)
                else:
                    st.write(f"No se encontró imagen para el ID: {pokemon_id}")
                    st.image("https://via.placeholder.com/150", caption="Imagen no disponible")
                
                st.markdown(
                    f"""
                    <div class="pokemon-types-container">
                        <div><span class="pokemon-type-label">Type 1:</span> {row['Type 1']}</div>
                        {'<div><span class="pokemon-type-label">Type 2:</span> ' + row['Type 2'] + '</div>' if row['Type 2'] and row['Type 2'] != 'None' else ''}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(f"**{row['Name']}**")
                stats = {
                    'HP': row['HP'],
                    'Attack': row['Attack'],
                    'Defense': row['Defense'],
                    'Sp. Atk': row['Sp. Atk'],
                    'Sp. Def': row['Sp. Def'],
                    'Speed': row['Speed']
                }
                
                colors = ['#4CAF50', '#2196F3', '#FFC107', '#E91E63', '#9C27B0', '#00BCD4'] 
                
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.bar(stats.keys(), stats.values(), color=colors) 
                ax.set_ylim(0, max(stats.values()) * 1.2)
                ax.set_ylabel('Valor')
                ax.set_title('Estadísticas Base')

                for bar in bars:
                    yval = bar.get_height()
                    # CAMBIO IMPLEMENTADO AQUÍ: Reducción del valor sumado a yval
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), # Cambiado de yval + 5 a yval + 1
                            ha='center', va='bottom', fontsize=9, color='black') 

                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
    else:
        st.write("No se encontraron Pokémon que coincidan con los criterios de búsqueda.")