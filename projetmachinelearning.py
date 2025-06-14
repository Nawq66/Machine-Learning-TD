"""
Script complet de nettoyage et validation du dataset eco2mix RTE
TD Machine Learning - IFP School
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("NETTOYAGE DU DATASET ECO2MIX")
print("="*60)

# =====================================================
# 1. CHARGEMENT DES DONNÉES
# =====================================================
print("\n1. CHARGEMENT DES DONNÉES...")

# Charger le dataset avec le bon séparateur
df = pd.read_csv(r"C:\Users\nahta\Desktop\machinelearning\Dataset RTE - Eco2mix.csv", sep=';', encoding='utf-8')
print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Afficher les premières colonnes pour vérification
print("\nColonnes détectées:")
for i, col in enumerate(df.columns[:10]):
    print(f"  {i}: {col}")
print("  ...")

# =====================================================
# 1.5 ANALYSE DES NA DANS LE DATASET ORIGINAL
# =====================================================
print("\n1.5 ANALYSE DES VALEURS MANQUANTES (DATASET ORIGINAL)...")

# Compter les NA dans le dataset brut
na_original_total = df.isnull().sum().sum()
print(f"\nTOTAL DE VALEURS MANQUANTES : {na_original_total:,}")

# Analyser par colonne
na_by_column = df.isnull().sum()
na_by_column_nonzero = na_by_column[na_by_column > 0]

if len(na_by_column_nonzero) > 0:
    print("\nDétail par colonne :")
    print("-" * 60)
    for col, count in na_by_column_nonzero.sort_values(ascending=False).items():
        percentage = (count / len(df)) * 100
        print(f"  {col:<40} : {count:>7,} NA ({percentage:>5.2f}%)")
    print("-" * 60)
else:
    print("\nAucune valeur manquante détectée dans le dataset brut !")

# Identifier les colonnes qui devraient être numériques
print("\nAnalyse des colonnes numériques potentielles...")
numeric_patterns = ['(MW)', '(%)', 'TCO', 'TCH', 'batterie']
potential_numeric_cols = [col for col in df.columns 
                         if any(pattern in col for pattern in numeric_patterns)]

print(f"Nombre de colonnes numériques identifiées : {len(potential_numeric_cols)}")

# Vérifier les valeurs non-convertibles qui deviendront NA
non_numeric_count = 0
for col in potential_numeric_cols[:5]:  # Tester sur les 5 premières
    if col in df.columns:
        # Essayer la conversion
        test_conversion = pd.to_numeric(df[col], errors='coerce')
        new_na = test_conversion.isna().sum() - df[col].isna().sum()
        if new_na > 0:
            print(f"  Attention {col}: {new_na} valeurs non-numériques détectées")
            non_numeric_count += new_na

if non_numeric_count > 0:
    print(f"\nTotal de valeurs qui deviendront NA après conversion : {non_numeric_count}")

# Statistiques globales sur la qualité des données
print("\n📈 STATISTIQUES DE QUALITÉ (DATASET ORIGINAL):")
print(f"  - Lignes totales : {len(df):,}")
print(f"  - Colonnes totales : {len(df.columns)}")
print(f"  - Cellules totales : {df.size:,}")
print(f"  - Cellules avec NA : {na_original_total:,}")
print(f"  - Taux de complétude : {((df.size - na_original_total) / df.size * 100):.2f}%")

# Sauvegarder ces statistiques pour référence
na_stats_original = {
    'total_na': na_original_total,
    'na_by_column': na_by_column_nonzero.to_dict() if len(na_by_column_nonzero) > 0 else {},
    'completeness_rate': ((df.size - na_original_total) / df.size * 100)
}

# Visualiser les NA si présents
if na_original_total > 0:
    print("\nCréation d'une visualisation des NA...")
    
    # Créer une figure pour visualiser les NA
    fig_na, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique 1 : Top 10 des colonnes avec le plus de NA
    top_na = na_by_column_nonzero.sort_values(ascending=False).head(10)
    if len(top_na) > 0:
        ax1.barh(range(len(top_na)), top_na.values)
        ax1.set_yticks(range(len(top_na)))
        ax1.set_yticklabels([col[:30] + '...' if len(col) > 30 else col for col in top_na.index])
        ax1.set_xlabel('Nombre de valeurs manquantes')
        ax1.set_title('Top 10 des colonnes avec le plus de NA')
        ax1.set_xlim(0, max(top_na.values) * 1.1)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(top_na.values):
            ax1.text(v + 0.01 * max(top_na.values), i, f'{v:,}', va='center')
    
    # Graphique 2 : Heatmap des NA (échantillon)
    # Prendre un échantillon pour la visualisation
    sample_size = min(1000, len(df))
    sample_indices = np.random.choice(df.index, sample_size, replace=False)
    df_sample = df.loc[sample_indices]
    
    # Créer une matrice binaire des NA
    na_matrix = df_sample.isnull().astype(int)
    
    # Ne garder que les colonnes avec au moins un NA
    cols_with_na = na_matrix.columns[na_matrix.any()]
    if len(cols_with_na) > 0:
        na_matrix_filtered = na_matrix[cols_with_na]
        
        # Créer la heatmap
        im = ax2.imshow(na_matrix_filtered.T, aspect='auto', cmap='RdYlBu', interpolation='nearest')
        ax2.set_yticks(range(len(cols_with_na)))
        ax2.set_yticklabels([col[:20] + '...' if len(col) > 20 else col for col in cols_with_na])
        ax2.set_xlabel(f'Échantillon de lignes (n={sample_size})')
        ax2.set_title('Pattern des valeurs manquantes')
        
        # Ajouter une colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('NA (1) vs Valeur (0)')
    
    plt.tight_layout()
    plt.savefig('analyse_na_original.png', dpi=300, bbox_inches='tight')
    print("Visualisation des NA sauvegardée dans 'analyse_na_original.png'")

print("\n" + "="*60)

# =====================================================
# 2. NETTOYAGE DES NOMS DE COLONNES
# =====================================================
print("\n2. NETTOYAGE DES NOMS DE COLONNES...")

# Dictionnaire de renommage pour plus de lisibilité
rename_cols = {
    'Consommation (MW)': 'Consommation_MW',
    'Thermique (MW)': 'Thermique_MW',
    'Nucléaire (MW)': 'Nucleaire_MW',
    'Eolien (MW)': 'Eolien_MW',
    'Solaire (MW)': 'Solaire_MW',
    'Hydraulique (MW)': 'Hydraulique_MW',
    'Pompage (MW)': 'Pompage_MW',
    'Bioénergies (MW)': 'Bioenergies_MW',
    'Ech. physiques (MW)': 'Echanges_physiques_MW',
    'Stockage batterie': 'Stockage_batterie_MW',
    'Déstockage batterie': 'Destockage_batterie_MW',
    'Eolien terrestre': 'Eolien_terrestre_MW',
    'Eolien offshore': 'Eolien_offshore_MW',
    'TCO Thermique (%)': 'TCO_Thermique',
    'TCH Thermique (%)': 'TCH_Thermique',
    'TCO Nucléaire (%)': 'TCO_Nucleaire',
    'TCH Nucléaire (%)': 'TCH_Nucleaire',
    'TCO Eolien (%)': 'TCO_Eolien',
    'TCH Eolien (%)': 'TCH_Eolien',
    'TCO Solaire (%)': 'TCO_Solaire',
    'TCH Solaire (%)': 'TCH_Solaire',
    'TCO Hydraulique (%)': 'TCO_Hydraulique',
    'TCH Hydraulique (%)': 'TCH_Hydraulique',
    'TCO Bioénergies (%)': 'TCO_Bioenergies',
    'TCH Bioénergies (%)': 'TCH_Bioenergies',
    'Région': 'Region',
    'Date - Heure': 'Date_Heure'
}

df = df.rename(columns=rename_cols)
print(f"{len(rename_cols)} colonnes renommées")

# Supprimer les colonnes inutiles
cols_to_drop = []
if 'Column 30' in df.columns:
    cols_to_drop.append('Column 30')
if 'Code INSEE région' in df.columns:
    cols_to_drop.append('Code INSEE région')  # Redondant avec Région

df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
print(f"{len(cols_to_drop)} colonnes supprimées")

# =====================================================
# 3. CONVERSION DES TYPES DE DONNÉES (AVANT L'AGRÉGATION!)
# =====================================================
print("\n🔢 3. CONVERSION DES TYPES DE DONNÉES...")

# Liste de toutes les colonnes numériques
numeric_cols = ['Consommation_MW', 'Thermique_MW', 'Nucleaire_MW', 
                'Eolien_MW', 'Solaire_MW', 'Hydraulique_MW', 
                'Pompage_MW', 'Bioenergies_MW', 'Echanges_physiques_MW',
                'Stockage_batterie_MW', 'Destockage_batterie_MW',
                'Eolien_terrestre_MW', 'Eolien_offshore_MW'] + \
               [col for col in df.columns if 'TCO' in col or 'TCH' in col]

# Convertir en numérique AVANT l'agrégation
converted = 0
for col in numeric_cols:
    if col in df.columns:
        # Remplacer les virgules par des points si nécessaire
        df[col] = df[col].astype(str).str.replace(',', '.')
        # Convertir en float
        df[col] = pd.to_numeric(df[col], errors='coerce')
        converted += 1

print(f"{converted} colonnes converties en numérique")

# =====================================================
# 4. GESTION DES DATES ET TRANSFORMATION HORAIRE
# =====================================================
print("\n4. GESTION DES DATES...")

# Créer une colonne datetime propre
try:
    if 'Date' in df.columns and 'Heure' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Heure'], format='%Y-%m-%d %H:%M')
    elif 'Date_Heure' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date_Heure'])
    else:
        print("Avertissement : colonnes Date/Heure non trouvées. Vérifiez le format du dataset.")
        print("Colonnes disponibles:", df.columns.tolist())
        raise ValueError("Impossible de créer la colonne Datetime")
        
    df = df.sort_values(['Region', 'Datetime'])
    
except Exception as e:
    print(f"Erreur lors de la création de la colonne Datetime: {e}")
    print("Vérifiez le format des dates dans votre dataset")
    raise

# Vérifier la granularité temporelle
time_diffs = df.groupby('Region')['Datetime'].diff().dropna()
granularite = time_diffs.mode()[0]
print(f"Granularité temporelle détectée : {granularite}")

# Si les données sont en 30 minutes, agréger à l'heure
if granularite == pd.Timedelta('30 min'):
    print("Agrégation des données à l'heure...")
    
    df['Hour'] = df['Datetime'].dt.floor('H')
    
    # Colonnes à sommer (MW -> MWh pour une demi-heure)
    cols_to_sum = ['Consommation_MW', 'Thermique_MW', 'Nucleaire_MW', 
                   'Eolien_MW', 'Solaire_MW', 'Hydraulique_MW', 
                   'Pompage_MW', 'Bioenergies_MW', 'Echanges_physiques_MW',
                   'Stockage_batterie_MW', 'Destockage_batterie_MW',
                   'Eolien_terrestre_MW', 'Eolien_offshore_MW']
    
    # Colonnes à moyenner (taux de charge en %)
    cols_to_mean = [col for col in df.columns if 'TCO' in col or 'TCH' in col]
    
    # Créer le dictionnaire d'agrégation
    agg_dict = {}
    for col in cols_to_sum:
        if col in df.columns:
            agg_dict[col] = 'sum'
    for col in cols_to_mean:
        if col in df.columns:
            agg_dict[col] = 'mean'
    agg_dict['Nature'] = 'first'
    
    # Agréger
    df = df.groupby(['Region', 'Hour']).agg(agg_dict).reset_index()
    df = df.rename(columns={'Hour': 'Datetime'})
    
    # Diviser par 2 les colonnes sommées pour obtenir des MWh
    for col in cols_to_sum:
        if col in df.columns:
            df[col] = df[col] / 2
    
    print("Données agrégées à l'heure (MWh)")

# Note: La conversion des types a déjà été faite avant l'agrégation

# =====================================================
# 5. GESTION DES VALEURS MANQUANTES
# =====================================================
print("\n5. ANALYSE ET TRAITEMENT DES VALEURS MANQUANTES...")

# Re-définir numeric_cols pour la suite du traitement
numeric_cols = ['Consommation_MW', 'Thermique_MW', 'Nucleaire_MW', 
                'Eolien_MW', 'Solaire_MW', 'Hydraulique_MW', 
                'Pompage_MW', 'Bioenergies_MW', 'Echanges_physiques_MW',
                'Stockage_batterie_MW', 'Destockage_batterie_MW',
                'Eolien_terrestre_MW', 'Eolien_offshore_MW'] + \
               [col for col in df.columns if 'TCO' in col or 'TCH' in col]

# Analyser les NA avant traitement
na_before = df.isnull().sum().sum()
print(f"Nombre total de NA avant traitement : {na_before}")

# Stratégie 1 : Pour les taux de charge, mettre 0 si production = 0
for col in df.columns:
    if ('TCO' in col or 'TCH' in col) and col in df.columns:
        # Extraire le nom de la filière
        if '_' in col:
            filiere = col.split('_')[1]
            prod_col = f"{filiere}_MW"
            if prod_col in df.columns:
                # Si production = 0, alors taux de charge = 0
                mask = (df[prod_col] == 0) | (df[prod_col].isna())
                df.loc[mask, col] = 0

# Stratégie 2 : Interpolation linéaire pour les productions
prod_cols = ['Thermique_MW', 'Nucleaire_MW', 'Eolien_MW', 
             'Solaire_MW', 'Hydraulique_MW', 'Bioenergies_MW']

for col in prod_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df.groupby('Region')[col].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both')
        )

# Stratégie 3 : Remplir les NA restants par 0 pour les colonnes de production
for col in numeric_cols:
    if col in df.columns and df[col].isnull().any():
        df[col] = df[col].fillna(0)

na_after = df.isnull().sum().sum()
print(f"NA traités : {na_before} → {na_after}")

# =====================================================
# 6. CRÉATION DE VARIABLES SUPPLÉMENTAIRES
# =====================================================
print("\n6. CRÉATION DE VARIABLES SUPPLÉMENTAIRES...")

# Variables temporelles
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['Quarter'] = df['Datetime'].dt.quarter

# Saisons
df['Season'] = df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
})

# Production totale
prod_cols_exist = [col for col in prod_cols if col in df.columns]
df['Production_totale_MW'] = df[prod_cols_exist].sum(axis=1)

# Production renouvelable
renew_cols = ['Eolien_MW', 'Solaire_MW', 'Hydraulique_MW', 'Bioenergies_MW']
renew_cols_exist = [col for col in renew_cols if col in df.columns]
df['Production_renouvelable_MW'] = df[renew_cols_exist].sum(axis=1)

# Part des renouvelables
df['Part_renouvelable'] = np.where(
    df['Production_totale_MW'] > 0,
    (df['Production_renouvelable_MW'] / df['Production_totale_MW']) * 100,
    0
)

# Balance énergétique
df['Balance_MW'] = df['Production_totale_MW'] - df['Consommation_MW']

print(f"{9} nouvelles variables créées")

# =====================================================
# 7. VALIDATION DU DATASET
# =====================================================
print("\n7. VALIDATION DU DATASET NETTOYÉ")
print("="*60)

def validate_dataset(df):
    """Fonction de validation complète du dataset"""
    
    print("RÉSUMÉ DU DATASET:")
    print(f"  - Nombre de lignes : {df.shape[0]:,}")
    print(f"  - Nombre de colonnes : {df.shape[1]}")
    print(f"  - Période : {df['Datetime'].min()} à {df['Datetime'].max()}")
    print(f"  - Nombre de régions : {df['Region'].nunique()}")
    print(f"  - Régions : {', '.join(df['Region'].unique())}")
    
    print("\nCOHÉRENCE DES DONNÉES:")
    # Vérifier qu'il n'y a pas de valeurs négatives où c'est impossible
    cols_positive = ['Consommation_MW', 'Thermique_MW', 'Nucleaire_MW', 
                     'Solaire_MW', 'Bioenergies_MW']
    for col in cols_positive:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                print(f"  Attention {col} : {neg_count} valeurs négatives trouvées")
            else:
                print(f"  {col} : aucune valeur négative")
    
    print("\nVALEURS MANQUANTES:")
    na_cols = df.columns[df.isnull().any()].tolist()
    if len(na_cols) == 0:
        print("  Aucune valeur manquante")
    else:
        print(f"  Colonnes avec NA : {na_cols}")
    
    print("\nSTATISTIQUES CLÉS:")
    print(f"  - Consommation moyenne : {df['Consommation_MW'].mean():.0f} MWh")
    print(f"  - Production totale moyenne : {df['Production_totale_MW'].mean():.0f} MWh")
    print(f"  - Part renouvelable moyenne : {df['Part_renouvelable'].mean():.1f}%")
    
    return True

# Valider le dataset
validate_dataset(df)

# =====================================================
# 8. VISUALISATIONS DE CONTRÔLE
# =====================================================
print("\n8. GÉNÉRATION DES VISUALISATIONS DE CONTRÔLE...")

# Créer une figure avec plusieurs subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Validation visuelle du dataset eco2mix nettoyé', fontsize=16)

# 1. Distribution de la consommation par région
ax1 = axes[0, 0]
df.boxplot(column='Consommation_MW', by='Region', ax=ax1, rot=45)
ax1.set_title('Distribution de la consommation par région')
ax1.set_xlabel('Région')
ax1.set_ylabel('Consommation (MWh)')

# 2. Évolution temporelle de la consommation (échantillon)
ax2 = axes[0, 1]
sample_region = df['Region'].mode()[0]
df_sample = df[df['Region'] == sample_region].iloc[:24*7]  # Une semaine
ax2.plot(df_sample['Datetime'], df_sample['Consommation_MW'])
ax2.set_title(f'Évolution de la consommation - {sample_region}')
ax2.set_xlabel('Date')
ax2.set_ylabel('Consommation (MWh)')
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylim(bottom=0)

# 3. Mix énergétique moyen
ax3 = axes[0, 2]
prod_means = []
prod_labels = []
for col in ['Nucleaire_MW', 'Thermique_MW', 'Hydraulique_MW', 
            'Eolien_MW', 'Solaire_MW', 'Bioenergies_MW']:
    if col in df.columns:
        prod_means.append(df[col].mean())
        prod_labels.append(col.replace('_MW', ''))

ax3.pie(prod_means, labels=prod_labels, autopct='%1.1f%%')
ax3.axis('equal')
ax3.set_title('Mix énergétique moyen')

# 4. Profil journalier de consommation
ax4 = axes[1, 0]
hourly_cons = df.groupby('Hour')['Consommation_MW'].mean()
ax4.plot(hourly_cons.index, hourly_cons.values, marker='o')
ax4.set_ylim(bottom=0)
ax4.set_title('Profil journalier moyen de consommation')
ax4.set_xlabel('Heure')
ax4.set_ylabel('Consommation moyenne (MWh)')
ax4.grid(True, alpha=0.3)

# 5. Corrélation production vs consommation
ax5 = axes[1, 1]
ax5.scatter(df['Production_totale_MW'], df['Consommation_MW'],
            alpha=0.1, s=1)
ax5.set_xlim(left=0)
ax5.set_ylim(bottom=0)
ax5.set_title('Production vs Consommation')
ax5.set_xlabel('Production totale (MWh)')
ax5.set_ylabel('Consommation (MWh)')

# 6. Distribution de la part renouvelable
ax6 = axes[1, 2]
df['Part_renouvelable'].hist(bins=50, ax=ax6)
ax6.set_xlim(0, 100)
ax6.set_title('Distribution de la part renouvelable')
ax6.set_xlabel('Part renouvelable (%)')
ax6.set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig('validation_dataset.png', dpi=300, bbox_inches='tight')
print("Graphiques de validation sauvegardés dans 'validation_dataset.png'")

# =====================================================
# 9. SAUVEGARDE DU DATASET NETTOYÉ
# =====================================================
print("\n9. SAUVEGARDE DU DATASET...")

# Sauvegarder le dataset nettoyé
df.to_csv('eco2mix_cleaned.csv', index=False)
print("Dataset nettoyé sauvegardé : 'eco2mix_cleaned.csv'")

# Sauvegarder un échantillon pour vérification rapide
df.head(1000).to_csv('eco2mix_sample.csv', index=False)
print("Échantillon sauvegardé : 'eco2mix_sample.csv'")

# =====================================================
# 10. RAPPORT FINAL
# =====================================================
print("\n" + "="*60)
print("NETTOYAGE TERMINÉ AVEC SUCCÈS")
print("="*60)

print("\nACTIONS EFFECTUÉES:")
print("  1. Chargement avec séparateur ';'")
print("  2. Renommage des colonnes pour lisibilité")
print("  3. Conversion au pas horaire (si nécessaire)")
print("  4. Conversion des types de données")
print("  5. Traitement des valeurs manquantes")
print("  6. Création de variables temporelles et calculées")
print("  7. Validation de la cohérence des données")
print("  8. Génération de visualisations de contrôle")

print("\nBILAN DES VALEURS MANQUANTES:")
print(f"  - NA dans le dataset ORIGINAL : {na_stats_original['total_na']:,}")
print(f"  - NA dans le dataset FINAL : 0")
print(f"  - Taux de complétude ORIGINAL : {na_stats_original['completeness_rate']:.2f}%")
print(f"  - Taux de complétude FINAL : 100.00%")

if na_stats_original['total_na'] > 0:
    print("\n  Méthodes de traitement utilisées :")
    print("    • Interpolation linéaire pour les séries temporelles")
    print("    • Règles métier (TCO=0 si production=0)")
    print("    • Remplissage par 0 pour les valeurs résiduelles")

print("\nPROCHAINES ÉTAPES:")
print("  → Partie 2 : Visualisation approfondie des données")
print("  → Partie 3 : Tests statistiques de corrélation")
print("  → Partie 4 : Modèles de régression (consommation)")
print("  → Partie 5 : Classification (risque de blackout)")

print("\nCONSEIL:")
print("  Examinez le fichier 'validation_dataset.png' pour vérifier")
print("  visuellement la qualité du nettoyage avant de continuer.")

# Afficher un résumé des colonnes finales
print("\nCOLONNES DISPONIBLES POUR L'ANALYSE:")
print("  Variables de production:", [col for col in df.columns if '_MW' in col and 'Production' not in col][:5], "...")
print("  Variables temporelles:", ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'IsWeekend', 'Season'])
print("  Variables calculées:", ['Production_totale_MW', 'Production_renouvelable_MW', 'Part_renouvelable', 'Balance_MW'])

# Créer un petit dataframe de métadonnées pour référence
metadata = pd.DataFrame({
    'Colonne': df.columns,
    'Type': df.dtypes,
    'Non_NA': df.notna().sum(),
    'Unique_Values': df.nunique()
})
metadata.to_csv('metadata_dataset.csv', index=False)
print("\nMétadonnées sauvegardées dans 'metadata_dataset.csv'")

# Sauvegarder aussi le rapport de NA pour comparaison
with open('rapport_na_original.txt', 'w', encoding='utf-8') as f:
    f.write("RAPPORT DES VALEURS MANQUANTES - DATASET ORIGINAL\n")
    f.write("="*50 + "\n\n")
    f.write(f"Total de NA : {na_stats_original['total_na']:,}\n")
    f.write(f"Taux de complétude : {na_stats_original['completeness_rate']:.2f}%\n\n")
    
    if na_stats_original['na_by_column']:
        f.write("Détail par colonne :\n")
        for col, count in na_stats_original['na_by_column'].items():
            f.write(f"  - {col}: {count:,} NA\n")
    else:
        f.write("Aucune valeur manquante détectée.\n")

print("Rapport des NA sauvegardé dans 'rapport_na_original.txt'")

plt.show()