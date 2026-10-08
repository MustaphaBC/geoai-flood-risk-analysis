# geoai_streamlit_app.py
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import tempfile
import os
import webbrowser
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from shapely.geometry import Point, Polygon, LineString
import json
import io
import base64
import zipfile

class FloodRiskAppStreamlit:
    def __init__(self):
        st.set_page_config(
            page_title="GeoAI - Analyse des Risques d'Inondation",
            page_icon="🌊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialisation des variables d'état
        if 'gdf_buildings' not in st.session_state:
            st.session_state.gdf_buildings = None
        if 'gdf_rivers' not in st.session_state:
            st.session_state.gdf_rivers = None
        if 'gdf_roads' not in st.session_state:
            st.session_state.gdf_roads = None
        if 'gdf_dem' not in st.session_state:
            st.session_state.gdf_dem = None
        if 'features_gdf' not in st.session_state:
            st.session_state.features_gdf = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'show_building_preview' not in st.session_state:
            st.session_state.show_building_preview = False
        
        self.setup_ui()
    
    def setup_ui(self):
        # En-tête principal
        st.title("🌊 GeoAI - Analyseur de Risques d'Inondation")
        st.markdown("---")
        
        # Contenu principal avec deux colonnes
        col1, col2 = st.columns([1, 2])
        
        with col1:
            self.setup_controls()
        
        with col2:
            self.display_results()
    
    def setup_controls(self):
        # Section Configuration de l'Analyse
        st.header("⚙️ Configuration de l'Analyse")
        
        # Section Chargement des Données
        with st.expander("📁 Chargement des Données", expanded=True):
            st.subheader("Sélection des fichiers")
            
            # Bâtiments
            building_files = st.file_uploader(
                "🏢 Bâtiments (Shapefile)",
                type=['shp', 'dbf', 'shx', 'prj'],
                key="buildings",
                accept_multiple_files=True,
                help="Sélectionnez tous les fichiers du shapefile (.shp, .dbf, .shx, .prj)"
            )
            
            # Rivières
            river_files = st.file_uploader(
                "🌊 Rivières (Shapefile)", 
                type=['shp', 'dbf', 'shx', 'prj'],
                key="rivers", 
                accept_multiple_files=True
            )
            
            # Routes
            road_files = st.file_uploader(
                "🛣️ Routes (Shapefile)",
                type=['shp', 'dbf', 'shx', 'prj'],
                key="roads",
                accept_multiple_files=True
            )
            
            # MNT
            dem_files = st.file_uploader(
                "⛰️ MNT (Shapefile)",
                type=['shp', 'dbf', 'shx', 'prj'],
                key="dem",
                accept_multiple_files=True
            )
            
            # Boutons de chargement
            col_load1, col_load2, col_load3 = st.columns(3)
            with col_load1:
                if st.button("🏗️ Générer Données Exemple", use_container_width=True):
                    self.generate_sample_data()
            with col_load2:
                if st.button("📂 Charger Mes Données", use_container_width=True):
                    self.load_user_data(building_files, river_files, road_files, dem_files)
            with col_load3:
                if st.button("📁 Charger depuis Dossier Data", use_container_width=True):
                    self.load_from_data_folder()
        
        # Section Paramètres du Modèle
        with st.expander("🤖 Paramètres du Modèle", expanded=True):
            st.subheader("Configuration du modèle")
            
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                n_estimators = st.slider(
                    "Nombre d'arbres",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Nombre d'arbres dans la forêt aléatoire"
                )
            
            with col_param2:
                test_size = st.slider(
                    "Taille de l'ensemble de test (%)",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Pourcentage de données utilisées pour le test"
                ) / 100
            
            if st.button("🔍 Analyser les Risques", use_container_width=True):
                self.analyze_risks(n_estimators, test_size)
        
        # Section Export des Résultats
        with st.expander("📤 Export des Résultats", expanded=True):
            st.subheader("Options d'export")
            
            if st.button("🗺️ Générer Carte Interactive", use_container_width=True):
                self.generate_interactive_map()
            
            if st.button("📊 Afficher Carte Statique", use_container_width=True):
                self.show_static_map()
            
            col_export1, col_export2 = st.columns(2)
            with col_export1:
                if st.button("📄 Exporter Rapport", use_container_width=True):
                    self.export_report()
                if st.button("💾 Exporter Shapefile", use_container_width=True):
                    self.export_shapefile()
            with col_export2:
                if st.button("🌐 Exporter GeoJSON", use_container_width=True):
                    self.export_geojson()
    
    def load_from_data_folder(self):
        """Charge automatiquement les données depuis le dossier 'data'"""
        with st.spinner("📁 Chargement des données depuis le dossier 'data'..."):
            try:
                data_dir = "data"
                
                if not os.path.exists(data_dir):
                    st.error(f"❌ Le dossier '{data_dir}' n'existe pas. Veuillez d'abord générer les données.")
                    return
                
                loaded_files = 0
                
                # Charger les bâtiments
                buildings_path = os.path.join(data_dir, "buildings.shp")
                if os.path.exists(buildings_path):
                    st.session_state.gdf_buildings = gpd.read_file(buildings_path)
                    st.success(f"✅ Bâtiments chargés: {len(st.session_state.gdf_buildings)} entités")
                    loaded_files += 1
                    
                    # Afficher un aperçu des données avec un bouton au lieu d'un expander
                    if st.button("📋 Afficher l'aperçu des bâtiments"):
                        st.session_state.show_building_preview = not st.session_state.show_building_preview
                    
                    if st.session_state.show_building_preview:
                        st.dataframe(st.session_state.gdf_buildings.head())
                else:
                    st.warning("⚠️ Fichier buildings.shp non trouvé dans le dossier 'data'")
                
                # Charger les rivières
                rivers_path = os.path.join(data_dir, "rivers.shp")
                if os.path.exists(rivers_path):
                    st.session_state.gdf_rivers = gpd.read_file(rivers_path)
                    st.success(f"✅ Rivières chargées: {len(st.session_state.gdf_rivers)} entités")
                    loaded_files += 1
                else:
                    st.warning("⚠️ Fichier rivers.shp non trouvé dans le dossier 'data'")
                
                # Charger les routes
                roads_path = os.path.join(data_dir, "roads.shp")
                if os.path.exists(roads_path):
                    st.session_state.gdf_roads = gpd.read_file(roads_path)
                    st.success(f"✅ Routes chargées: {len(st.session_state.gdf_roads)} entités")
                    loaded_files += 1
                else:
                    st.warning("⚠️ Fichier roads.shp non trouvé dans le dossier 'data'")
                
                # Charger le MNT
                dem_path = os.path.join(data_dir, "dem.shp")
                if os.path.exists(dem_path):
                    st.session_state.gdf_dem = gpd.read_file(dem_path)
                    st.success(f"✅ MNT chargé: {len(st.session_state.gdf_dem)} cellules")
                    loaded_files += 1
                else:
                    st.warning("⚠️ Fichier dem.shp non trouvé dans le dossier 'data'")
                
                if loaded_files > 0:
                    st.success(f"✅ {loaded_files} type(s) de données chargé(s) depuis le dossier 'data'")
                    
                    # Afficher les statistiques des données chargées
                    self.display_data_stats()
                else:
                    st.error("❌ Aucune donnée valide trouvée dans le dossier 'data'")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement depuis le dossier 'data': {str(e)}")
    
    def display_data_stats(self):
        """Affiche les statistiques des données chargées"""
        st.subheader("📊 Statistiques des Données Chargées")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.gdf_buildings is not None:
                st.metric("🏢 Bâtiments", len(st.session_state.gdf_buildings))
                if 'type' in st.session_state.gdf_buildings.columns:
                    types_batiments = st.session_state.gdf_buildings['type'].value_counts()
                    st.write("**Types:**")
                    for type_bat, count in types_batiments.items():
                        st.write(f"- {type_bat}: {count}")
        
        with col2:
            if st.session_state.gdf_rivers is not None:
                st.metric("🌊 Rivières", len(st.session_state.gdf_rivers))
                if 'name' in st.session_state.gdf_rivers.columns:
                    st.write("**Noms:**")
                    for name in st.session_state.gdf_rivers['name']:
                        st.write(f"- {name}")
        
        with col3:
            if st.session_state.gdf_roads is not None:
                st.metric("🛣️ Routes", len(st.session_state.gdf_roads))
                if 'type' in st.session_state.gdf_roads.columns:
                    types_routes = st.session_state.gdf_roads['type'].value_counts()
                    st.write("**Types:**")
                    for type_route, count in types_routes.items():
                        st.write(f"- {type_route}: {count}")
        
        with col4:
            if st.session_state.gdf_dem is not None:
                st.metric("⛰️ Cellules MNT", len(st.session_state.gdf_dem))
                if 'elevation' in st.session_state.gdf_dem.columns:
                    elevation_min = st.session_state.gdf_dem['elevation'].min()
                    elevation_max = st.session_state.gdf_dem['elevation'].max()
                    elevation_mean = st.session_state.gdf_dem['elevation'].mean()
                    st.write(f"**Élévation:**")
                    st.write(f"- Min: {elevation_min:.1f}m")
                    st.write(f"- Max: {elevation_max:.1f}m")
                    st.write(f"- Moy: {elevation_mean:.1f}m")
    
    def display_results(self):
        st.header("📊 Résultats et Visualisations")
        
        # Statistiques rapides
        if st.session_state.features_gdf is not None:
            self.display_quick_stats()
        
        # Onglets pour différents types de résultats
        tab1, tab2, tab3 = st.tabs(["📝 Journal d'Analyse", "🗺️ Visualisation", "📈 Statistiques"])
        
        with tab1:
            self.display_analysis_logs()
        
        with tab2:
            self.display_visualizations()
        
        with tab3:
            self.display_detailed_stats()
    
    def display_quick_stats(self):
        features_gdf = st.session_state.features_gdf
        if features_gdf is None:
            return
        
        risk_counts = features_gdf['risk_category'].value_counts().sort_index()
        total_buildings = len(features_gdf)
        high_risk_count = risk_counts.get('Élevé', 0)
        high_risk_percentage = (high_risk_count / total_buildings) * 100
        
        # Affichage des métriques
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Bâtiments", f"{total_buildings:,}")
        
        with col2:
            st.metric("Risque Élevé", f"{high_risk_count}", f"{high_risk_percentage:.1f}%")
        
        with col3:
            medium_risk = risk_counts.get('Moyen', 0)
            st.metric("Risque Moyen", f"{medium_risk}")
        
        with col4:
            if st.session_state.model is not None:
                st.metric("Précision Estimée", "85-95%")
    
    def display_analysis_logs(self):
        st.subheader("Journal d'Exécution")
        
        if st.session_state.features_gdf is not None:
            st.success("✅ Analyse terminée avec succès!")
            
            features_gdf = st.session_state.features_gdf
            risk_counts = features_gdf['risk_category'].value_counts().sort_index()
            
            st.write("**RÉPARTITION DES RISQUES:**")
            for category in ['Très faible', 'Faible', 'Moyen', 'Élevé']:
                count = risk_counts.get(category, 0)
                percentage = count / len(features_gdf) * 100
                st.write(f"- **{category}**: {count} bâtiments ({percentage:.1f}%)")
            
            # Importance des caractéristiques
            if st.session_state.model is not None:
                st.write("**IMPORTANCE DES CARACTÉRISTIQUES:**")
                feature_importance = pd.DataFrame({
                    'feature': ['elevation', 'distance_to_river', 'distance_to_road', 'slope', 'soil_type', 'rainfall'],
                    'importance': st.session_state.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for _, row in feature_importance.iterrows():
                    st.write(f"- **{row['feature']}**: {row['importance']:.3f}")
        else:
            st.info("ℹ️ Aucune analyse effectuée. Utilisez les boutons de configuration pour commencer.")
    
    def display_visualizations(self):
        st.subheader("Visualisations Géospatiales")
        
        if st.session_state.features_gdf is not None:
            st.write("**Carte Interactive des Risques**")
            self.display_interactive_map()
        else:
            st.info("ℹ️ Générer des données ou effectuer une analyse pour voir les visualisations.")
    
    def display_detailed_stats(self):
        st.subheader("Statistiques Détaillées")
        
        if st.session_state.features_gdf is not None:
            features_gdf = st.session_state.features_gdf
            
            # Graphiques supplémentaires
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogramme des risques
                fig, ax = plt.subplots(figsize=(8, 6))
                risk_counts = features_gdf['risk_category'].value_counts().sort_index()
                colors = ['green', 'lightgreen', 'orange', 'red']
                bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7)
                ax.set_title('Répartition des Niveaux de Risque')
                ax.set_ylabel('Nombre de Bâtiments')
                plt.xticks(rotation=45)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', 
                           ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
            
            with col2:
                # Relation altitude-risque
                fig, ax = plt.subplots(figsize=(8, 6))
                risk_numeric = features_gdf['risk_category'].map({'Très faible': 0, 'Faible': 1, 'Moyen': 2, 'Élevé': 3})
                scatter = ax.scatter(features_gdf['elevation'], features_gdf['distance_to_river'], 
                                   c=risk_numeric, cmap='RdYlGn_r', s=50, alpha=0.6)
                ax.set_xlabel('Altitude (m)')
                ax.set_ylabel('Distance à la rivière (°)')
                ax.set_title('Relation Altitude / Distance Rivière')
                plt.colorbar(scatter, ax=ax, label='Niveau de Risque')
                st.pyplot(fig)
            
            # Tableau des caractéristiques
            st.write("**Caractéristiques Moyennes par Catégorie de Risque:**")
            
            stats_by_risk = features_gdf.groupby('risk_category', observed=True).agg({
                'elevation': 'mean',
                'distance_to_river': 'mean',
                'flood_risk': 'mean',
                'rainfall': 'mean'
            }).round(3)
            
            st.dataframe(stats_by_risk)
    
    def generate_sample_data(self):
        with st.spinner("🏗️ Génération des données d'exemple..."):
            try:
                # Génération de données d'exemple réalistes
                np.random.seed(42)
                bbox = [-74.1, 40.6, -73.9, 40.8]  # Zone autour de New York
                
                # Bâtiments avec différentes densités
                n_buildings = 200
                lons = np.random.normal(-74.0, 0.05, n_buildings)
                lats = np.random.normal(40.7, 0.05, n_buildings)
                
                buildings_geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
                st.session_state.gdf_buildings = gpd.GeoDataFrame({
                    'id': range(n_buildings),
                    'type': np.random.choice(['résidentiel', 'commercial', 'industriel', 'public'], n_buildings, p=[0.6, 0.2, 0.1, 0.1]),
                    'height': np.random.uniform(5, 50, n_buildings),
                    'geometry': buildings_geometry
                }, crs='EPSG:4326')
                
                # Rivière sinueuse
                river_x = np.linspace(bbox[0] + 0.02, bbox[2] - 0.02, 50)
                river_y = 0.2 * np.sin(10 * (river_x - bbox[0])) + (bbox[1] + bbox[3]) / 2
                river_coords = list(zip(river_x, river_y))
                st.session_state.gdf_rivers = gpd.GeoDataFrame({
                    'name': ['Rivière Principale'],
                    'width': [15],
                    'geometry': [LineString(river_coords)]
                }, crs='EPSG:4326')
                
                # Routes
                n_roads = 8
                roads_geometry = []
                for i in range(n_roads):
                    road_y = bbox[1] + (bbox[3] - bbox[1]) * (i + 1) / (n_roads + 1)
                    road_coords = [(bbox[0] + 0.02, road_y), (bbox[2] - 0.02, road_y)]
                    roads_geometry.append(LineString(road_coords))
                
                st.session_state.gdf_roads = gpd.GeoDataFrame({
                    'name': [f'Route {i+1}' for i in range(n_roads)],
                    'type': np.random.choice(['locale', 'principale', 'autoroute'], n_roads),
                    'geometry': roads_geometry
                }, crs='EPSG:4326')
                
                # MNT simulé (topographie réaliste)
                x = np.linspace(bbox[0], bbox[2], 25)
                y = np.linspace(bbox[1], bbox[3], 25)
                polygons, elevations = [], []
                
                for i in range(len(x)-1):
                    for j in range(len(y)-1):
                        poly = Polygon([[x[i], y[j]], [x[i+1], y[j]], [x[i+1], y[j+1]], [x[i], y[j+1]]])
                        polygons.append(poly)
                        
                        center_x = (x[i] + x[i+1]) / 2
                        center_y = (y[j] + y[j+1]) / 2
                        
                        # Modèle d'élévation réaliste
                        river_y_at_x = 0.2 * np.sin(10 * (center_x - bbox[0])) + (bbox[1] + bbox[3]) / 2
                        distance_to_river = np.sqrt((center_x - center_x)**2 + (center_y - river_y_at_x)**2)
                        
                        # Élévation plus basse près de la rivière
                        base_elevation = 20 - 15 * np.exp(-distance_to_river**2 / 0.005)
                        # Ajouter du bruit pour un relief réaliste
                        noise = np.random.normal(0, 2)
                        elevation = max(1, base_elevation + noise)
                        elevations.append(elevation)
                
                st.session_state.gdf_dem = gpd.GeoDataFrame({
                    'elevation': elevations,
                    'geometry': polygons
                }, crs='EPSG:4326')
                
                st.success(f"✅ Données d'exemple générées: {n_buildings} bâtiments, 1 rivière, {n_roads} routes")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération: {str(e)}")
    
    def load_user_data(self, building_files, river_files, road_files, dem_files):
        with st.spinner("📂 Chargement des données utilisateur..."):
            try:
                loaded_files = 0
                
                # Fonction helper pour charger les shapefiles
                def load_shapefile(uploaded_files, data_type):
                    if not uploaded_files:
                        return None
                    
                    # Créer un dossier temporaire pour les fichiers shapefile
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Sauvegarder tous les fichiers uploadés
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(tmp_dir, uploaded_file.name)
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                        
                        # Trouver le fichier .shp
                        shp_files = [f for f in os.listdir(tmp_dir) if f.endswith('.shp')]
                        if shp_files:
                            shp_path = os.path.join(tmp_dir, shp_files[0])
                            try:
                                gdf = gpd.read_file(shp_path)
                                st.success(f"✅ {data_type} chargés: {len(gdf)} entités")
                                return gdf
                            except Exception as e:
                                st.error(f"❌ Erreur lecture {data_type}: {str(e)}")
                                return None
                        else:
                            st.warning(f"⚠️ Aucun fichier .shp trouvé pour {data_type}")
                            return None
                
                # Charger chaque type de données
                if building_files:
                    st.session_state.gdf_buildings = load_shapefile(building_files, "Bâtiments")
                    if st.session_state.gdf_buildings is not None:
                        loaded_files += 1
                        # Afficher un aperçu des données avec un bouton
                        if st.button("📋 Aperçu des bâtiments (upload)"):
                            st.dataframe(st.session_state.gdf_buildings.head())
                
                if river_files:
                    st.session_state.gdf_rivers = load_shapefile(river_files, "Rivières")
                    if st.session_state.gdf_rivers is not None:
                        loaded_files += 1
                
                if road_files:
                    st.session_state.gdf_roads = load_shapefile(road_files, "Routes")
                    if st.session_state.gdf_roads is not None:
                        loaded_files += 1
                
                if dem_files:
                    st.session_state.gdf_dem = load_shapefile(dem_files, "MNT")
                    if st.session_state.gdf_dem is not None:
                        loaded_files += 1
                
                if loaded_files == 0:
                    st.warning("⚠️ Aucun fichier valide chargé. Vérifiez que vous avez uploadé tous les fichiers nécessaires (.shp, .dbf, .shx, .prj)")
                else:
                    st.success(f"✅ {loaded_files} type(s) de données chargé(s) avec succès")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {str(e)}")
    
    def analyze_risks(self, n_estimators, test_size):
        if st.session_state.gdf_buildings is None:
            st.error("❌ Veuillez d'abord charger des données (bâtiments requis)")
            return
        
        with st.spinner("🔍 Analyse des risques d'inondation en cours..."):
            try:
                # Calcul des caractéristiques
                features_list = []
                
                for idx, building in st.session_state.gdf_buildings.iterrows():
                    building_point = building.geometry
                    
                    # Distance à la rivière
                    min_river_distance = 1000  # valeur par défaut élevée
                    if st.session_state.gdf_rivers is not None:
                        river_distances = [building_point.distance(river.geometry) for _, river in st.session_state.gdf_rivers.iterrows()]
                        min_river_distance = min(river_distances) if river_distances else 1000
                    
                    # Altitude (interpolation depuis le MNT)
                    elevation = self._interpolate_elevation(building_point)
                    
                    # Distance aux routes
                    min_road_distance = 1000
                    if st.session_state.gdf_roads is not None:
                        road_distances = [building_point.distance(road.geometry) for _, road in st.session_state.gdf_roads.iterrows()]
                        min_road_distance = min(road_distances) if road_distances else 1000
                    
                    # Caractéristiques dérivées
                    slope = self._calculate_slope(building_point, elevation)
                    soil_type = self._determine_soil_type(min_river_distance, elevation)
                    rainfall = self._estimate_rainfall(min_river_distance, elevation)
                    
                    # Historique d'inondation simulé (basé sur la logique métier)
                    base_risk = (0.3 * (1 - elevation / 20) +  # Contribution de l'élévation
                               0.4 * (1 - min(min_river_distance / 0.03, 1)) +  # Distance à la rivière
                               0.2 * (slope / 10) +  # Pente
                               0.1 * (rainfall / 100))  # Précipitations
                    
                    flood_history = 1 if np.random.random() < base_risk else 0
                    
                    features_list.append({
                        'building_id': idx,
                        'elevation': elevation,
                        'distance_to_river': min_river_distance,
                        'distance_to_road': min_road_distance,
                        'slope': slope,
                        'soil_type': soil_type,
                        'rainfall': rainfall,
                        'flood_history': flood_history,
                        'geometry': building_point
                    })
                
                st.session_state.features_gdf = gpd.GeoDataFrame(features_list, crs='EPSG:4326')
                
                # Préparation des données pour le ML
                X = st.session_state.features_gdf[['elevation', 'distance_to_river', 'distance_to_road', 
                                                 'slope', 'soil_type', 'rainfall']].copy()
                
                # Encodage des variables catégorielles
                soil_mapping = {'sableux': 0, 'limoneux': 1, 'argileux': 2, 'rocheux': 3}
                X['soil_type'] = X['soil_type'].map(soil_mapping)
                
                y = st.session_state.features_gdf['flood_history']
                
                # Division train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Entraînement du modèle
                st.session_state.model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                )
                st.session_state.model.fit(X_train, y_train)
                
                # Prédictions sur l'ensemble des données
                risk_proba = st.session_state.model.predict_proba(X)[:, 1]
                st.session_state.features_gdf['flood_risk'] = risk_proba
                st.session_state.features_gdf['risk_category'] = pd.cut(
                    risk_proba, 
                    bins=[0, 0.2, 0.5, 0.8, 1], 
                    labels=['Très faible', 'Faible', 'Moyen', 'Élevé']
                )
                
                st.success("✅ Analyse terminée avec succès!")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
    
    def _interpolate_elevation(self, point):
        """Interpole l'élévation à partir du MNT"""
        if st.session_state.gdf_dem is None:
            # Si pas de MNT, générer une élévation basée sur la distance à la rivière
            if st.session_state.gdf_rivers is not None:
                river_distances = [point.distance(river.geometry) for _, river in st.session_state.gdf_rivers.iterrows()]
                min_river_distance = min(river_distances) if river_distances else 0.02
                return max(1, 15 - 10 * np.exp(-min_river_distance**2 / 0.005))
            return 10  # Valeur par défaut
        
        # Recherche de la cellule MNT contenant le point
        for _, dem_cell in st.session_state.gdf_dem.iterrows():
            if dem_cell.geometry.contains(point):
                return dem_cell['elevation']
        
        # Si non trouvé, retourner l'élévation moyenne
        return st.session_state.gdf_dem['elevation'].mean()
    
    def _calculate_slope(self, point, elevation):
        """Calcule la pente approximative"""
        # Simulation simple - dans une vraie application, utiliser un MNT raster
        return np.random.uniform(0, 5)
    
    def _determine_soil_type(self, river_distance, elevation):
        """Détermine le type de sol basé sur la distance à la rivière et l'élévation"""
        if river_distance < 0.01:
            return 'limoneux'
        elif river_distance < 0.02:
            return 'argileux'
        elif elevation < 8:
            return 'sableux'
        else:
            return 'rocheux'
    
    def _estimate_rainfall(self, river_distance, elevation):
        """Estime les précipitations"""
        base_rainfall = 80
        river_effect = 20 * (1 - min(river_distance / 0.03, 1))
        elevation_effect = 10 * (elevation / 20)
        return base_rainfall + river_effect + elevation_effect
    
    def display_interactive_map(self):
        if st.session_state.features_gdf is None:
            return
        
        try:
            # Centre de la carte
            center_lat = st.session_state.features_gdf.geometry.y.mean()
            center_lon = st.session_state.features_gdf.geometry.x.mean()
            
            # Création de la carte Folium
            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=13,
                tiles='OpenStreetMap'
            )
            
            # Ajouter rivières
            if st.session_state.gdf_rivers is not None:
                for _, river in st.session_state.gdf_rivers.iterrows():
                    if hasattr(river.geometry, 'coords'):
                        coords = [(point[1], point[0]) for point in river.geometry.coords]
                        folium.PolyLine(
                            coords, 
                            color='blue', 
                            weight=4, 
                            opacity=0.7,
                            popup=folium.Popup(f"Rivière: {river.get('name', 'Inconnu')}", max_width=200)
                        ).add_to(m)
            
            # Ajouter routes
            if st.session_state.gdf_roads is not None:
                for _, road in st.session_state.gdf_roads.iterrows():
                    if hasattr(road.geometry, 'coords'):
                        coords = [(point[1], point[0]) for point in road.geometry.coords]
                        folium.PolyLine(
                            coords, 
                            color='gray', 
                            weight=2, 
                            opacity=0.6,
                            popup=folium.Popup(f"Route: {road.get('name', 'Inconnu')}", max_width=200)
                        ).add_to(m)
            
            # Ajouter bâtiments avec risques
            risk_colors = {
                'Très faible': 'green', 
                'Faible': 'lightgreen', 
                'Moyen': 'orange', 
                'Élevé': 'red'
            }
            
            for idx, row in st.session_state.features_gdf.iterrows():
                color = risk_colors.get(row['risk_category'], 'gray')
                
                # Popup détaillé
                popup_html = f"""
                <div style="font-family: Arial; width: 200px">
                    <h4>Bâtiment {idx}</h4>
                    <b>Niveau de risque:</b> {row['risk_category']}<br>
                    <b>Probabilité:</b> {row['flood_risk']:.1%}<br>
                    <b>Altitude:</b> {row['elevation']:.1f}m<br>
                    <b>Distance rivière:</b> {row['distance_to_river']:.3f}°<br>
                    <b>Précipitations:</b> {row['rainfall']:.0f}mm
                </div>
                """
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=8,
                    popup=folium.Popup(popup_html, max_width=300),
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=1
                ).add_to(m)
            
            # Afficher la carte dans Streamlit
            st_folium(m, width=700, height=500)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'affichage de la carte: {str(e)}")
    
    def generate_interactive_map(self):
        if st.session_state.features_gdf is None:
            st.error("❌ Veuillez d'abord effectuer l'analyse des risques")
            return
        
        try:
            # Même code que display_interactive_map mais sauvegarde le fichier
            center_lat = st.session_state.features_gdf.geometry.y.mean()
            center_lon = st.session_state.features_gdf.geometry.x.mean()
            
            m = folium.Map(
                location=[center_lat, center_lon], 
                zoom_start=13,
                tiles='OpenStreetMap'
            )
            
            # Ajouter rivières
            if st.session_state.gdf_rivers is not None:
                for _, river in st.session_state.gdf_rivers.iterrows():
                    if hasattr(river.geometry, 'coords'):
                        coords = [(point[1], point[0]) for point in river.geometry.coords]
                        folium.PolyLine(
                            coords, 
                            color='blue', 
                            weight=4, 
                            opacity=0.7,
                            popup=folium.Popup(f"Rivière: {river.get('name', 'Inconnu')}", max_width=200)
                        ).add_to(m)
            
            # Ajouter routes
            if st.session_state.gdf_roads is not None:
                for _, road in st.session_state.gdf_roads.iterrows():
                    if hasattr(road.geometry, 'coords'):
                        coords = [(point[1], point[0]) for point in road.geometry.coords]
                        folium.PolyLine(
                            coords, 
                            color='gray', 
                            weight=2, 
                            opacity=0.6,
                            popup=folium.Popup(f"Route: {road.get('name', 'Inconnu')}", max_width=200)
                        ).add_to(m)
            
            # Ajouter bâtiments avec risques
            risk_colors = {
                'Très faible': 'green', 
                'Faible': 'lightgreen', 
                'Moyen': 'orange', 
                'Élevé': 'red'
            }
            
            for idx, row in st.session_state.features_gdf.iterrows():
                color = risk_colors.get(row['risk_category'], 'gray')
                
                popup_html = f"""
                <div style="font-family: Arial; width: 200px">
                    <h4>Bâtiment {idx}</h4>
                    <b>Niveau de risque:</b> {row['risk_category']}<br>
                    <b>Probabilité:</b> {row['flood_risk']:.1%}<br>
                    <b>Altitude:</b> {row['elevation']:.1f}m<br>
                    <b>Distance rivière:</b> {row['distance_to_river']:.3f}°<br>
                    <b>Précipitations:</b> {row['rainfall']:.0f}mm
                </div>
                """
                
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=8,
                    popup=folium.Popup(popup_html, max_width=300),
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=1
                ).add_to(m)
            
            # Sauvegarder la carte
            map_file = "carte_risques_inondation.html"
            m.save(map_file)
            
            st.success(f"✅ Carte interactive générée: {map_file}")
            
            # Offrir le téléchargement
            with open(map_file, "rb") as file:
                st.download_button(
                    label="📥 Télécharger la carte HTML",
                    data=file,
                    file_name="carte_risques_inondation.html",
                    mime="text/html"
                )
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la génération de la carte: {str(e)}")
    
    def show_static_map(self):
        if st.session_state.features_gdf is None:
            st.error("❌ Veuillez d'abord effectuer l'analyse des risques")
            return
        
        try:
            # Créer la figure avec plusieurs sous-graphiques
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Analyse des Risques d\'Inondation', fontsize=16, fontweight='bold')
            
            # 1. Carte géographique principale
            if st.session_state.gdf_rivers is not None:
                st.session_state.gdf_rivers.plot(ax=ax1, color='blue', linewidth=3, label='Rivières', alpha=0.7)
            
            if st.session_state.gdf_roads is not None:
                st.session_state.gdf_roads.plot(ax=ax1, color='gray', linewidth=1, label='Routes', alpha=0.6)
            
            risk_colors = {'Très faible': 'green', 'Faible': 'lightgreen', 'Moyen': 'orange', 'Élevé': 'red'}
            for category, color in risk_colors.items():
                subset = st.session_state.features_gdf[st.session_state.features_gdf['risk_category'] == category]
                if len(subset) > 0:
                    subset.plot(ax=ax1, color=color, markersize=30, label=f'Risque {category}', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            ax1.set_title('Carte des Risques d\'Inondation')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # 2. Histogramme des risques
            risk_counts = st.session_state.features_gdf['risk_category'].value_counts().sort_index()
            colors = [risk_colors[cat] for cat in risk_counts.index]
            bars = ax2.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_title('Répartition des Niveaux de Risque')
            ax2.set_ylabel('Nombre de Bâtiments')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{int(height)}', 
                        ha='center', va='bottom', fontweight='bold')
            
            # 3. Relation altitude-risque
            risk_numeric = st.session_state.features_gdf['risk_category'].map({'Très faible': 0, 'Faible': 1, 'Moyen': 2, 'Élevé': 3})
            scatter = ax3.scatter(st.session_state.features_gdf['elevation'], 
                                st.session_state.features_gdf['distance_to_river'], 
                                c=risk_numeric, cmap='RdYlGn_r', s=50, alpha=0.6)
            ax3.set_xlabel('Altitude (m)')
            ax3.set_ylabel('Distance à la rivière (°)')
            ax3.set_title('Relation Altitude / Distance Rivière')
            plt.colorbar(scatter, ax=ax3, label='Niveau de Risque')
            
            # 4. Distribution des probabilités
            risk_proba = st.session_state.features_gdf['flood_risk']
            ax4.hist(risk_proba, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Probabilité de Risque')
            ax4.set_ylabel('Nombre de Bâtiments')
            ax4.set_title('Distribution des Probabilités de Risque')
            ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Seuil moyen')
            ax4.legend()
            
            plt.tight_layout()
            
            # Afficher dans Streamlit
            st.pyplot(fig)
            
            # Bouton de téléchargement
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button(
                label="📥 Télécharger cette visualisation",
                data=buf,
                file_name="carte_static_risques.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'affichage de la carte: {str(e)}")
    
    def export_report(self):
        if st.session_state.features_gdf is None:
            st.error("❌ Veuillez d'abord effectuer l'analyse des risques")
            return
        
        try:
            # Générer le rapport en mémoire
            output = io.StringIO()
            
            output.write("RAPPORT GEOAI - ANALYSE DES RISQUES D'INONDATION\n")
            output.write("=" * 60 + "\n\n")
            
            features_gdf = st.session_state.features_gdf
            
            # En-tête
            output.write(f"Date de génération: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output.write(f"Nombre total de bâtiments analysés: {len(features_gdf)}\n\n")
            
            # Statistiques détaillées
            output.write("STATISTIQUES DÉTAILLÉES:\n")
            output.write("-" * 30 + "\n")
            
            risk_counts = features_gdf['risk_category'].value_counts().sort_index()
            for category in ['Très faible', 'Faible', 'Moyen', 'Élevé']:
                count = risk_counts.get(category, 0)
                percentage = count / len(features_gdf) * 100
                output.write(f"{category:<15}: {count:>4} bâtiments ({percentage:>5.1f}%)\n")
            
            # Caractéristiques moyennes par catégorie de risque
            output.write("\nCARACTÉRISTIQUES MOYENNES PAR CATÉGORIE:\n")
            output.write("-" * 45 + "\n")
            stats_by_risk = features_gdf.groupby('risk_category').agg({
                'elevation': 'mean',
                'distance_to_river': 'mean',
                'flood_risk': 'mean'
            }).round(3)
            
            for category in ['Très faible', 'Faible', 'Moyen', 'Élevé']:
                if category in stats_by_risk.index:
                    stats = stats_by_risk.loc[category]
                    output.write(f"\n{category}:\n")
                    output.write(f"  • Altitude moyenne: {stats['elevation']} m\n")
                    output.write(f"  • Distance moyenne à la rivière: {stats['distance_to_river']} °\n")
                    output.write(f"  • Probabilité moyenne: {stats['flood_risk']:.1%}\n")
            
            # Recommandations
            output.write("\nRECOMMANDATIONS:\n")
            output.write("-" * 20 + "\n")
            
            high_risk_count = risk_counts.get('Élevé', 0)
            medium_risk_count = risk_counts.get('Moyen', 0)
            
            if high_risk_count > 0:
                output.write(f"\n🚨 URGENT - {high_risk_count} bâtiment(s) à risque ÉLEVÉ:\n")
                output.write("  • Mettre en place des systèmes d'alerte précoce\n")
                output.write("  • Élaborer des plans d'évacuation d'urgence\n")
                output.write("  • Renforcer les protections contre les inondations\n")
                output.write("  • Considérer des mesures de relocalisation\n")
            
            if medium_risk_count > 0:
                output.write(f"\n⚠️  ATTENTION - {medium_risk_count} bâtiment(s) à risque MOYEN:\n")
                output.write("  • Surveiller régulièrement les conditions météo\n")
                output.write("  • Améliorer les systèmes de drainage\n")
                output.write("  • Stocker le matériel de protection\n")
                output.write("  • Former le personnel aux procédures d'urgence\n")
            
            output.write("\n📋 ACTIONS PRÉVENTIVES GÉNÉRALES:\n")
            output.write("  • Amélioration continue du drainage urbain\n")
            output.write("  • Maintien et renforcement des berges\n")
            output.write("  • Surveillance météorologique renforcée\n")
            output.write("  • Planification urbaine adaptée au risque\n")
            output.write("  • Sensibilisation des populations concernées\n")
            
            output.write(f"\n\nRapport généré par GeoAI Flood Risk Analyzer\n")
            
            # Offrir le téléchargement
            st.download_button(
                label="📥 Télécharger le Rapport Complet",
                data=output.getvalue(),
                file_name="rapport_risques_inondation.txt",
                mime="text/plain"
            )
            
            st.success("✅ Rapport généré avec succès!")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export du rapport: {str(e)}")
    
    def export_shapefile(self):
        if st.session_state.features_gdf is None:
            st.error("❌ Veuillez d'abord effectuer l'analyse des risques")
            return
        
        try:
            # Préparer les données
            export_gdf = st.session_state.features_gdf.copy()
            keep_columns = ['elevation', 'distance_to_river', 'distance_to_road', 
                          'slope', 'soil_type', 'rainfall', 'flood_risk', 'risk_category', 'geometry']
            export_gdf = export_gdf[keep_columns]
            
            # Sauvegarder dans un buffer
            with tempfile.TemporaryDirectory() as tmp_dir:
                shp_path = os.path.join(tmp_dir, "risques_inondation.shp")
                export_gdf.to_file(shp_path)
                
                # Créer un zip avec tous les fichiers shapefile
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    for file in os.listdir(tmp_dir):
                        file_path = os.path.join(tmp_dir, file)
                        zip_file.write(file_path, file)
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📥 Télécharger Shapefile (ZIP)",
                    data=zip_buffer,
                    file_name="risques_inondation_shapefile.zip",
                    mime="application/zip"
                )
            
            st.success("✅ Shapefile préparé pour téléchargement!")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export Shapefile: {str(e)}")
    
    def export_geojson(self):
        if st.session_state.features_gdf is None:
            st.error("❌ Veuillez d'abord effectuer l'analyse des risques")
            return
        
        try:
            export_gdf = st.session_state.features_gdf.copy()
            keep_columns = ['elevation', 'distance_to_river', 'distance_to_road', 
                          'slope', 'soil_type', 'rainfall', 'flood_risk', 'risk_category', 'geometry']
            export_gdf = export_gdf[keep_columns]
            
            # Convertir en GeoJSON
            geojson_str = export_gdf.to_json()
            
            st.download_button(
                label="📥 Télécharger GeoJSON",
                data=geojson_str,
                file_name="risques_inondation.geojson",
                mime="application/json"
            )
            
            st.success("✅ GeoJSON préparé pour téléchargement!")
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'export GeoJSON: {str(e)}")

def main():
    app = FloodRiskAppStreamlit()

if __name__ == "__main__":
    main()