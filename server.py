#!/usr/bin/env python3
"""
Serveur MCP pour la recherche sémantique locale dans les datasets AgentPublic
Utilise BGE-M3 directement via FlagEmbedding (pas d'Ollama requis)
"""
from typing import Any
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, InferenceClient
from mcp.server.fastmcp import FastMCP
import re
import numpy as np
import warnings
import sys
import logging
import gc

# Lazy import FlagEmbedding only when needed to speed up startup
BGEM3FlagModel = None

# Suppress multiprocessing resource tracker warnings
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

# Configure logging to file instead of stderr to avoid MCP protocol issues
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/tmp/mcp_legifrance_vector.log'),
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
# Note: Dependencies are listed in pyproject.toml and should be pre-installed
mcp = FastMCP("Legifrance Vector")

# Configuration
COLLECTION_ID = "AgentPublic/mediatech"
CONFIG_FILE = "datasets_config.json"
# Lire depuis les variables d'environnement, avec fallback pour développement local
CACHE_DIR = os.getenv("SEMANTIC_CACHE_DIR", "semantic_cache")
# Configuration Hugging Face (optionnel)
TOKEN_HF = os.getenv("HF_TOKEN", None)  # Token HF depuis env (requis pour l'API HF)
USE_HF_API = False  # Sera défini automatiquement selon la disponibilité

# Global cache
datasets_cache = {}
embedding_model = None
hf_client = None
search_results_cache = {}

# Colonnes nécessaires pour optimiser le chargement parquet
# Service-Public: text, embedding, title, doc_id, theme, audience, surtitle, url
# LEGI: text, embedding, title, doc_id, number, status, start_date, end_date, date_debut, date_fin, nature, category, subtitles
REQUIRED_COLUMNS_SERVICE_PUBLIC = ['text', 'embedding', 'title', 'doc_id', 'theme', 'audience', 'surtitle', 'url']
REQUIRED_COLUMNS_LEGI = ['text', 'embedding', 'title', 'doc_id', 'number', 'status', 'start_date', 'end_date', 'date_debut', 'date_fin', 'nature', 'category', 'subtitles']

def get_hf_client():
    """Retourne le client Hugging Face Inference (lazy loading)"""
    global hf_client
    if hf_client is None and TOKEN_HF:
        hf_client = InferenceClient(token=TOKEN_HF)
    return hf_client

def check_hf_api_available() -> bool:
    """Vérifie si l'API Hugging Face est disponible avec le token fourni"""
    if not TOKEN_HF:
        return False

    try:
        client = get_hf_client()
        # Test simple avec feature_extraction
        result = client.feature_extraction("test", model="BAAI/bge-m3")
        return result is not None
    except:
        return False

def get_hf_embeddings(texts: list[str], token: str) -> list[list[float]]:
    """Génère des embeddings via l'API Hugging Face"""
    client = get_hf_client()
    embeddings = []

    for text in texts:
        embedding = client.feature_extraction(text, model="BAAI/bge-m3")
        # Convertir numpy array en liste
        if isinstance(embedding, np.ndarray):
            embeddings.append(embedding.tolist())
        else:
            embeddings.append(embedding)

    return embeddings

def get_hf_rerank_scores(query: str, texts: list[str], token: str) -> list[float]:
    """Calcule les scores de reranking via l'API Hugging Face

    Note: L'API HF Inference ne supporte pas nativement le reranking cross-encoder.
    Cette fonction utilise un fallback avec sentence_similarity qui est moins précis
    que le vrai reranking. Pour de meilleurs résultats, utilisez le modèle local.
    """
    client = get_hf_client()

    try:
        # L'API HF ne supporte pas sentence-ranking nativement
        # On utilise sentence_similarity comme fallback (moins précis)
        scores = client.sentence_similarity(
            query,
            other_sentences=texts
        )
        return scores if isinstance(scores, list) else [scores]
    except Exception as e:
        logger.warning(f"HF API sentence_similarity failed: {e}")
        # Fallback: retourner des scores neutres
        return [0.5] * len(texts)

def get_cache_dir() -> Path:
    """Retourne le répertoire de cache (créé si nécessaire)"""
    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(exist_ok=True)
    return cache_path

def load_config() -> dict:
    """Charge la configuration des datasets sélectionnés"""
    config_path = get_cache_dir() / CONFIG_FILE
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"selected_datasets": {}, "last_updated": {}}

def save_config(config: dict):
    """Sauvegarde la configuration"""
    config_path = get_cache_dir() / CONFIG_FILE
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def get_embedding_model():
    """Charge le modèle BGE-M3 (lazy loading) ou vérifie l'API HF"""
    global embedding_model, USE_HF_API, BGEM3FlagModel

    # Vérifier d'abord l'API HF si token fourni
    if TOKEN_HF and not USE_HF_API and embedding_model is None:
        logger.debug("Checking if HuggingFace API is available...")
        if check_hf_api_available():
            logger.info("HuggingFace API is available and will be used for embeddings")
            USE_HF_API = True
            return None
        else:
            logger.info("HuggingFace API check failed, falling back to local model")

    # Charger le modèle local si nécessaire
    if not USE_HF_API and embedding_model is None:
        logger.info("Loading local BGE-M3 model (this may take a while on first run)...")
        # Lazy import FlagEmbedding only when actually needed
        if BGEM3FlagModel is None:
            from FlagEmbedding import BGEM3FlagModel as _BGEM3FlagModel
            BGEM3FlagModel = _BGEM3FlagModel
        embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        logger.info("Local BGE-M3 model loaded successfully")

    return embedding_model
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs"""
    import math
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def get_latest_legi_folder(api: HfApi, repo_id: str = "AgentPublic/legi") -> str | None:
    """Trouve le dossier LEGI le plus récent (cherche d'abord legi-latest, puis legi-YYYYMMDD)"""
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")

        # 1. Chercher d'abord le dossier legi-latest
        has_legi_latest = any(f.startswith("data/legi-latest/") for f in files)
        if has_legi_latest:
            return "legi-latest"

        # 2. Si legi-latest n'existe pas, utiliser la logique actuelle (legi-YYYYMMDD)
        legi_folders = set()
        for f in files:
            if f.startswith("data/legi-") and "/" in f:
                match = re.match(r"data/(legi-\d{8})/", f)
                if match:
                    legi_folders.add(match.group(1))

        if not legi_folders:
            return None

        # Trier par date (le format YYYYMMDD permet le tri alphabétique)
        return sorted(legi_folders, reverse=True)[0]
    except:
        return None

def list_legi_codes(api: HfApi, legi_folder: str, repo_id: str = "AgentPublic/legi") -> list[dict]:
    """Liste tous les codes disponibles dans le dossier LEGI"""
    try:
        files = api.list_repo_files(repo_id, repo_type="dataset")
        
        # Filtrer les fichiers parquet dans le dossier LEGI
        code_files = []
        prefix = f"data/{legi_folder}/"
        
        for f in files:
            if f.startswith(prefix) and f.endswith('.parquet'):
                # Extraire le nom du code depuis le nom de fichier
                filename = f.split('/')[-1]
                code_name = filename.replace('.parquet', '')
                
                code_files.append({
                    "code": code_name,
                    "path": f,
                    "filename": filename
                })
        
        return sorted(code_files, key=lambda x: x['code'])
    except:
        return []


@mcp.tool()
async def configurer_datasets(
    action: str,
    datasets: list[str] | None = None,
    codes_legi: list[str] | None = None
) -> str:
    """Gestion des datasets pour la recherche sémantique

    Cet outil permet de gérer les datasets disponibles pour la recherche.

    Args:
        action: Action à effectuer - "lister_datasets", "ajouter_datasets", ou "query_mediatech"
        datasets: Liste des datasets à ajouter (pour action="ajouter_datasets")
                 Valeurs possibles: ["service-public"], ["legi"], ["constit"], ["dole"], ["cnil"]
        codes_legi: Liste des codes LEGI à télécharger (uniquement si "legi" est dans datasets)
                   Exemples: ["code_civil", "code_penal"], ["code_du_travail"]

    Exemples d'utilisation:
        - Lister les datasets locaux: action="lister_datasets"
        - Lister datasets HuggingFace: action="query_mediatech"
        - Ajouter Service-Public: action="ajouter_datasets", datasets=["service-public"]
        - Ajouter codes LEGI: action="ajouter_datasets", datasets=["legi"], codes_legi=["code_civil", "code_penal"]
        - Ajouter Constitution: action="ajouter_datasets", datasets=["constit"]
        - Ajouter DOLE: action="ajouter_datasets", datasets=["dole"]
        - Ajouter CNIL: action="ajouter_datasets", datasets=["cnil"]
        - Ajouter plusieurs: action="ajouter_datasets", datasets=["constit", "dole", "cnil"]
    """
    try:
        api = HfApi()

        # ACTION 1: LISTER LES DATASETS LOCAUX
        if action == "lister_datasets":
            config = load_config()
            selected = config.get("selected_datasets", {})

            output = "📂 DATASETS TÉLÉCHARGÉS EN LOCAL\n"

            if not selected:
                output += "❌ Aucun dataset téléchargé localement.\n\n"
                output += "💡 Utilisez action='query_mediatech' pour voir les datasets disponibles\n"
                output += "💡 Utilisez action='ajouter_datasets' pour télécharger des datasets\n"
                return output

            # Service-Public
            if 'service-public' in selected:
                sp_config = selected['service-public']
                output += "✅ SERVICE-PUBLIC (fiches particuliers & entreprises)\n"
                output += f"   📦 Repo: {sp_config['repo_id']}\n"
                output += f"   📄 {len(sp_config['files'])} fichier(s) parquet\n\n"

            # LEGI
            if 'legi' in selected:
                legi_config = selected['legi']
                output += "✅ LEGI (Codes)\n"
                output += f"   📦 Repo: {legi_config['repo_id']}\n"
                output += f"   📁 Version: {legi_config['folder']}\n"
                output += f"   📚 {len(legi_config['codes'])} code(s) disponible(s):\n"

                for code in legi_config['codes']:
                    output += f"      • {code['code']}\n"

                output += "\n"

            # Constitution
            if 'constit' in selected:
                constit_config = selected['constit']
                output += "✅ CONSTITUTION\n"
                output += f"   📦 Repo: {constit_config['repo_id']}\n"
                output += f"   📄 {len(constit_config['files'])} fichier(s) parquet\n\n"

            # DOLE
            if 'dole' in selected:
                dole_config = selected['dole']
                output += "✅ DOLE (Dossiers Legislatifs)\n"
                output += f"   📦 Repo: {dole_config['repo_id']}\n"
                output += f"   📄 {len(dole_config['files'])} fichier(s) parquet\n\n"

            # CNIL
            if 'cnil' in selected:
                cnil_config = selected['cnil']
                output += "✅ CNIL (Délibérations)\n"
                output += f"   📦 Repo: {cnil_config['repo_id']}\n"
                output += f"   📄 {len(cnil_config['files'])} fichier(s) parquet\n\n"

            return output

        # ACTION 2: QUERY MEDIATECH (Lister datasets HuggingFace)
        elif action == "query_mediatech":
            output = "🌐 DATASETS DISPONIBLES SUR HUGGINGFACE\n"

            # 1. SERVICE-PUBLIC
            output += "1️⃣ SERVICE-PUBLIC\n"
            output += "   📦 Repo: AgentPublic/service-public\n"
            output += "   📁 Chemin: data/service-public-latest/\n"

            try:
                files = api.list_repo_files("AgentPublic/service-public", repo_type="dataset")
                sp_files = [f for f in files if f.startswith("data/service-public-latest/") and f.endswith('.parquet')]
                output += f"   📄 {len(sp_files)} fichier(s) parquet\n\n"
            except Exception as e:
                output += f"   ⚠️  Erreur: {e}\n\n"

            # 2. LEGI
            output += "2️⃣ LEGI (Légifrance)\n"
            output += "   📦 Repo: AgentPublic/legi\n"

            legi_folder = get_latest_legi_folder(api)

            if legi_folder:
                output += f"   📁 Version la plus récente: {legi_folder}\n"

                codes = list_legi_codes(api, legi_folder)

                if codes:
                    output += f"   📚 {len(codes)} code(s) disponible(s):\n\n"

                    for code in codes:
                        output += f"      • {code['code']}\n"
                else:
                    output += "   ⚠️  Aucun code trouvé\n"
            else:
                output += "   ⚠️  Aucune version LEGI trouvée\n"

            output += "\n"

            # 3. CONSTITUTION
            output += "3️⃣ CONSTITUTION\n"
            output += "   📦 Repo: AgentPublic/constit\n"
            output += "   📁 Chemin: data/constit-latest/constit_part_0.parquet\n"
            output += "   📄 1 fichier parquet\n\n"

            # 4. DOLE
            output += "4️⃣ DOLE (Dossiers legislatifs)\n"
            output += "   📦 Repo: AgentPublic/dole\n"
            output += "   📁 Chemin: data/dole-latest/dole_part_0.parquet\n"
            output += "   📄 1 fichier parquet\n\n"

            # 5. CNIL
            output += "5️⃣ CNIL\n"
            output += "   📦 Repo: AgentPublic/cnil\n"
            output += "   📁 Chemin: data/cnil-latest/\n"
            output += "   📄 2 fichiers parquet (cnil_part_0.parquet, cnil_part_1.parquet)\n\n"

            output += "💡 Pour télécharger:\n"
            output += "   • Service-Public: action='ajouter_datasets', datasets=['service-public']\n"
            output += "   • LEGI: action='ajouter_datasets', datasets=['legi'], codes_legi=['code_civil', 'code_penal']\n"
            output += "   • Constitution: action='ajouter_datasets', datasets=['constit']\n"
            output += "   • DOLE: action='ajouter_datasets', datasets=['dole']\n"
            output += "   • CNIL: action='ajouter_datasets', datasets=['cnil']\n"

            return output

        # ACTION 3: AJOUTER DATASETS
        elif action == "ajouter_datasets":
            if not datasets:
                return "❌ Paramètre 'datasets' requis pour ajouter des datasets.\n" \
                       "Exemples: datasets=['service-public'] ou datasets=['legi']"

            config = load_config()
            current_config = config.get("selected_datasets", {})

            output = "📥 AJOUT DE DATASETS\n"

            datasets_to_add = {}

            # Traiter Service-Public
            if 'service-public' in datasets:
                output += "📦 SERVICE-PUBLIC\n"
                try:
                    files = api.list_repo_files("AgentPublic/service-public", repo_type="dataset")
                    sp_files = [f for f in files if f.startswith("data/service-public-latest/") and f.endswith('.parquet')]

                    if sp_files:
                        datasets_to_add['service-public'] = {
                            'repo_id': 'AgentPublic/service-public',
                            'files': sp_files
                        }
                        output += f"   ✅ {len(sp_files)} fichier(s) parquet à télécharger\n"
                    else:
                        output += "   ⚠️  Aucun fichier parquet trouvé\n"
                except Exception as e:
                    output += f"   ❌ Erreur: {e}\n"

                output += "\n"

            # Traiter LEGI
            if 'legi' in datasets:
                output += "📚 LEGI\n"

                if not codes_legi:
                    output += "   ❌ Paramètre 'codes_legi' requis pour télécharger LEGI\n"
                    output += "   💡 Utilisez action='query_mediatech' pour voir les codes disponibles\n"
                    output += "   💡 Exemple: codes_legi=['code_civil', 'code_penal']\n\n"
                else:
                    legi_folder = get_latest_legi_folder(api)

                    if not legi_folder:
                        output += "   ❌ Aucune version LEGI trouvée sur HuggingFace\n\n"
                    else:
                        output += f"   📁 Version: {legi_folder}\n"

                        available_codes = list_legi_codes(api, legi_folder)
                        available_code_names = {c['code'] for c in available_codes}
                        selected_codes = []

                        for code_name in codes_legi:
                            if code_name in available_code_names:
                                code_info = next(c for c in available_codes if c['code'] == code_name)
                                selected_codes.append(code_info)
                                output += f"   ✅ {code_name}\n"
                            else:
                                output += f"   ⚠️  {code_name} - non trouvé sur HuggingFace\n"

                        if selected_codes:
                            datasets_to_add['legi'] = {
                                'repo_id': 'AgentPublic/legi',
                                'folder': legi_folder,
                                'codes': selected_codes
                            }
                            output += f"   📊 Total: {len(selected_codes)} code(s) à télécharger\n"

                        output += "\n"

            # Traiter Constitution
            if 'constit' in datasets:
                output += "📜 CONSTITUTION\n"
                try:
                    constit_file = "data/constit-latest/constit_part_0.parquet"
                    datasets_to_add['constit'] = {
                        'repo_id': 'AgentPublic/constit',
                        'files': [constit_file]
                    }
                    output += f"   ✅ 1 fichier parquet à télécharger\n"
                except Exception as e:
                    output += f"   ❌ Erreur: {e}\n"

                output += "\n"

            # Traiter DOLE
            if 'dole' in datasets:
                output += "📚 DOLE (Doctrine en ligne)\n"
                try:
                    dole_file = "data/dole-latest/dole_part_0.parquet"
                    datasets_to_add['dole'] = {
                        'repo_id': 'AgentPublic/dole',
                        'files': [dole_file]
                    }
                    output += f"   ✅ 1 fichier parquet à télécharger\n"
                except Exception as e:
                    output += f"   ❌ Erreur: {e}\n"

                output += "\n"

            # Traiter CNIL
            if 'cnil' in datasets:
                output += "🔐 CNIL (Délibérations)\n"
                try:
                    cnil_files = [
                        "data/cnil-latest/cnil_part_0.parquet",
                        "data/cnil-latest/cnil_part_1.parquet"
                    ]
                    datasets_to_add['cnil'] = {
                        'repo_id': 'AgentPublic/cnil',
                        'files': cnil_files
                    }
                    output += f"   ✅ 2 fichiers parquet à télécharger\n"
                except Exception as e:
                    output += f"   ❌ Erreur: {e}\n"

                output += "\n"

            if not datasets_to_add:
                return output + "❌ Aucun dataset valide à ajouter\n"

            # Fusionner avec la configuration existante
            for key, value in datasets_to_add.items():
                if key == 'legi' and key in current_config:
                    # Pour LEGI, ajouter les nouveaux codes aux codes existants
                    existing_codes = current_config['legi'].get('codes', [])
                    existing_code_names = {c['code'] for c in existing_codes}

                    for new_code in value['codes']:
                        if new_code['code'] not in existing_code_names:
                            existing_codes.append(new_code)

                    current_config['legi']['codes'] = existing_codes
                    current_config['legi']['folder'] = value['folder']  # Mettre à jour le folder
                else:
                    current_config[key] = value

            # Sauvegarder la configuration
            config["selected_datasets"] = current_config
            save_config(config)

            output += "✅ Configuration sauvegardée\n\n"

            # Télécharger immédiatement
            output += "📥 TÉLÉCHARGEMENT EN COURS\n"

            try:
                # Télécharger Service-Public
                if 'service-public' in datasets_to_add:
                    sp_config = datasets_to_add['service-public']
                    output += f"📦 Téléchargement de {len(sp_config['files'])} fichier(s) Service-Public...\n"
                    files_to_download = [
                        {'repo_id': sp_config['repo_id'], 'path': f}
                        for f in sp_config['files']
                    ]
                    await download_and_cache_files(files_to_download)
                    output += "   ✅ Service-Public téléchargé\n\n"

                # Télécharger LEGI
                if 'legi' in datasets_to_add:
                    legi_config = datasets_to_add['legi']
                    output += f"📚 Téléchargement de {len(legi_config['codes'])} code(s) LEGI...\n"
                    files_to_download = [
                        {'repo_id': legi_config['repo_id'], 'path': code['path']}
                        for code in legi_config['codes']
                    ]
                    await download_and_cache_files(files_to_download)
                    output += "   ✅ LEGI téléchargé\n\n"

                # Télécharger Constitution
                if 'constit' in datasets_to_add:
                    constit_config = datasets_to_add['constit']
                    output += f"📜 Téléchargement de {len(constit_config['files'])} fichier(s) Constitution...\n"
                    files_to_download = [
                        {'repo_id': constit_config['repo_id'], 'path': f}
                        for f in constit_config['files']
                    ]
                    await download_and_cache_files(files_to_download)
                    output += "   ✅ Constitution téléchargée\n\n"

                # Télécharger DOLE
                if 'dole' in datasets_to_add:
                    dole_config = datasets_to_add['dole']
                    output += f"📚 Téléchargement de {len(dole_config['files'])} fichier(s) DOLE...\n"
                    files_to_download = [
                        {'repo_id': dole_config['repo_id'], 'path': f}
                        for f in dole_config['files']
                    ]
                    await download_and_cache_files(files_to_download)
                    output += "   ✅ DOLE téléchargé\n\n"

                # Télécharger CNIL
                if 'cnil' in datasets_to_add:
                    cnil_config = datasets_to_add['cnil']
                    output += f"🔐 Téléchargement de {len(cnil_config['files'])} fichier(s) CNIL...\n"
                    files_to_download = [
                        {'repo_id': cnil_config['repo_id'], 'path': f}
                        for f in cnil_config['files']
                    ]
                    await download_and_cache_files(files_to_download)
                    output += "   ✅ CNIL téléchargé\n\n"

                output += "✅ Tous les fichiers sont prêts pour la recherche!\n"

            except Exception as e:
                logger.error(f"Error downloading files: {e}", exc_info=True)
                output += f"\n⚠️  Erreur lors du téléchargement: {e}\n"
                output += "💡 Les fichiers seront téléchargés lors de la première recherche\n"

            return output

        else:
            return f"❌ Action '{action}' non reconnue.\n" \
                   f"Actions valides: 'lister_datasets', 'ajouter_datasets', 'query_mediatech'"

    except Exception as e:
        logger.error(f"Error in configurer_datasets: {e}", exc_info=True)
        return f"❌ Erreur lors de l'exécution: {e}"

async def download_and_cache_files(files_to_download: list[dict]) -> list[pd.DataFrame]:
    """Télécharge et met en cache des fichiers parquet"""
    cache_dir = get_cache_dir()
    dataframes = []

    for file_info in files_to_download:
        repo_id = file_info['repo_id']
        file_path = file_info['path']

        # Créer le dossier de cache
        repo_cache_dir = cache_dir / repo_id.replace("/", "_")
        repo_cache_dir.mkdir(exist_ok=True)

        # Télécharger avec hf_hub_download qui gère automatiquement le cache
        # et crée la structure de dossiers appropriée
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            repo_type="dataset",
            local_dir=str(repo_cache_dir)
        )

        # Charger le dataframe directement depuis le fichier téléchargé
        # Sans créer de copie avec un nom simplifié
        # Optimisation: charger uniquement les colonnes nécessaires
        # Note: impossible de déterminer le type ici, on charge toutes les colonnes
        # Le filtrage se fera dans load_datasets()
        df = pd.read_parquet(downloaded_path)
        # Convertir la colonne embeddings de string vers liste (selon doc HuggingFace)
        for col in df.columns:
            if 'embedding' in col.lower():
                df[col] = df[col].apply(json.loads)
                break
        dataframes.append(df)

    return dataframes

async def auto_discover_local_parquets() -> dict:
    """Découvre automatiquement les fichiers parquet déjà téléchargés localement"""
    discovered = {}

    # Chercher les dossiers AgentPublic_*
    current_dir = Path(".")

    # Service-Public
    sp_pattern = "AgentPublic_service-public/data*/service-public-latest/*.parquet"
    sp_files = list(current_dir.glob(sp_pattern))
    if sp_files:
        discovered['service-public'] = {
            'repo_id': 'AgentPublic/service-public',
            'files': [str(f) for f in sp_files]
        }

    # LEGI - chercher tous les codes
    legi_pattern = "AgentPublic_legi/data*/legi-*/*/*.parquet"
    legi_files = list(current_dir.glob(legi_pattern))

    if legi_files:
        # Organiser par code
        codes_dict = {}
        for file_path in legi_files:
            # Extraire le nom du code depuis le chemin
            parts = file_path.parts
            # Format attendu: AgentPublic_legi/data.../legi-YYYYMMDD/code_name/file.parquet
            if len(parts) >= 4:
                code_name = parts[-2]  # le nom du dossier contenant le code
                if code_name not in codes_dict:
                    codes_dict[code_name] = []
                codes_dict[code_name].append(str(file_path))

        if codes_dict:
            # Trouver le dossier legi le plus récent
            legi_folders = set()
            for file_path in legi_files:
                parts = file_path.parts
                for part in parts:
                    if part.startswith('legi-'):
                        legi_folders.add(part)

            legi_folder = sorted(legi_folders, reverse=True)[0] if legi_folders else 'legi-unknown'

            # Créer la structure pour chaque code
            codes_list = []
            for code_name, files in codes_dict.items():
                codes_list.append({
                    'code': code_name,
                    'path': files[0],  # Premier fichier parquet du code
                    'filename': Path(files[0]).name
                })

            discovered['legi'] = {
                'repo_id': 'AgentPublic/legi',
                'folder': legi_folder,
                'codes': codes_list
            }

    return discovered

async def load_datasets() -> dict[str, list[pd.DataFrame]]:
    """Charge tous les datasets configurés ou découverts automatiquement"""
    global datasets_cache

    logger.debug("Loading datasets configuration...")
    config = load_config()
    selected = config.get("selected_datasets", {})

    # Si aucun dataset configuré, essayer de découvrir automatiquement
    if not selected:
        logger.info("No configuration found, auto-discovering local parquet files...")
        discovered = await auto_discover_local_parquets()

        if discovered:
            logger.info(f"Discovered parquet files: {list(discovered.keys())}")
            # Sauvegarder la configuration découverte
            config["selected_datasets"] = discovered
            save_config(config)
            selected = discovered
        else:
            logger.warning("No local parquet files found")
            return {}

    # Charger Service-Public
    if 'service-public' in selected and 'service-public' not in datasets_cache:
        logger.info("Loading Service-Public dataset...")
        sp_config = selected['service-public']

        # Charger directement depuis les fichiers locaux si possible
        local_files = []
        cache_dir = get_cache_dir()
        repo_cache_dir = cache_dir / sp_config['repo_id'].replace("/", "_")

        for file_path in sp_config['files']:
            # Try multiple possible locations for the parquet file
            possible_paths = [
                Path(file_path),  # Relative path
                cache_dir / file_path,  # In semantic cache dir
                repo_cache_dir / file_path,  # In repo-specific cache dir
            ]

            for local_path in possible_paths:
                if local_path.exists():
                    local_files.append(local_path)
                    logger.debug(f"Found local file: {local_path}")
                    break

        if local_files:
            logger.info(f"Loading {len(local_files)} Service-Public file(s) from local disk...")
            dfs = []
            for f in local_files:
                # Optimisation: charger uniquement les colonnes nécessaires
                # Vérifier d'abord quelles colonnes sont disponibles
                available_cols = pd.read_parquet(f, columns=[]).columns.tolist()
                cols_to_load = [col for col in REQUIRED_COLUMNS_SERVICE_PUBLIC if col in available_cols]

                # Toujours inclure la colonne embedding
                for col in available_cols:
                    if 'embedding' in col.lower() and col not in cols_to_load:
                        cols_to_load.append(col)

                df = pd.read_parquet(f, columns=cols_to_load if cols_to_load else None)

                # Convertir la colonne embeddings de string vers liste (selon doc HuggingFace)
                for col in df.columns:
                    if 'embedding' in col.lower():
                        df[col] = df[col].apply(json.loads)
                        break
                dfs.append(df)
            datasets_cache['service-public'] = dfs
            gc.collect()  # Libérer la mémoire après chargement
            logger.info("Service-Public dataset loaded successfully")
        else:
            # Télécharger si nécessaire
            logger.info("Downloading Service-Public files...")
            files_to_download = [
                {'repo_id': sp_config['repo_id'], 'path': f}
                for f in sp_config['files']
            ]
            datasets_cache['service-public'] = await download_and_cache_files(files_to_download)
            logger.info("Service-Public files downloaded and loaded")

    # Charger LEGI
    if 'legi' in selected and 'legi' not in datasets_cache:
        logger.info("Loading LEGI dataset...")
        legi_config = selected['legi']

        # Charger directement depuis les fichiers locaux si possible
        local_files = []
        cache_dir = get_cache_dir()
        repo_cache_dir = cache_dir / legi_config['repo_id'].replace("/", "_")

        for code in legi_config['codes']:
            # Try multiple possible locations for the parquet file
            possible_paths = [
                Path(code['path']),  # Relative path
                cache_dir / code['path'],  # In semantic cache dir
                repo_cache_dir / code['path'],  # In repo-specific cache dir
            ]

            for local_path in possible_paths:
                if local_path.exists():
                    local_files.append(local_path)
                    logger.debug(f"Found local file: {local_path}")
                    break

        if local_files:
            logger.info(f"Loading {len(local_files)} LEGI code(s) from local disk...")
            dfs = []
            for f in local_files:
                # Optimisation: charger uniquement les colonnes nécessaires
                # Vérifier d'abord quelles colonnes sont disponibles
                available_cols = pd.read_parquet(f, columns=[]).columns.tolist()
                cols_to_load = [col for col in REQUIRED_COLUMNS_LEGI if col in available_cols]

                # Toujours inclure la colonne embedding
                for col in available_cols:
                    if 'embedding' in col.lower() and col not in cols_to_load:
                        cols_to_load.append(col)

                df = pd.read_parquet(f, columns=cols_to_load if cols_to_load else None)

                # Convertir la colonne embeddings de string vers liste (selon doc HuggingFace)
                for col in df.columns:
                    if 'embedding' in col.lower():
                        df[col] = df[col].apply(json.loads)
                        break
                dfs.append(df)
            datasets_cache['legi'] = dfs
            gc.collect()  # Libérer la mémoire après chargement
            logger.info("LEGI dataset loaded successfully")
        else:
            # Télécharger si nécessaire
            logger.info("Downloading LEGI files...")
            files_to_download = [
                {'repo_id': legi_config['repo_id'], 'path': code['path']}
                for code in legi_config['codes']
            ]
            datasets_cache['legi'] = await download_and_cache_files(files_to_download)
            logger.info("LEGI files downloaded and loaded")

    # Charger les nouveaux datasets (Constitution, DOLE, CNIL)
    # Ces datasets utilisent la même structure que Service-Public
    for dataset_name in ['constit', 'dole', 'cnil']:
        if dataset_name in selected and dataset_name not in datasets_cache:
            logger.info(f"Loading {dataset_name} dataset...")
            dataset_config = selected[dataset_name]

            # Charger directement depuis les fichiers locaux si possible
            local_files = []
            cache_dir = get_cache_dir()
            repo_cache_dir = cache_dir / dataset_config['repo_id'].replace("/", "_")

            for file_path in dataset_config['files']:
                # Try multiple possible locations for the parquet file
                possible_paths = [
                    Path(file_path),  # Relative path
                    cache_dir / file_path,  # In semantic cache dir
                    repo_cache_dir / file_path,  # In repo-specific cache dir
                ]

                for local_path in possible_paths:
                    if local_path.exists():
                        local_files.append(local_path)
                        logger.debug(f"Found local file: {local_path}")
                        break

            if local_files:
                logger.info(f"Loading {len(local_files)} {dataset_name} file(s) from local disk...")
                dfs = []
                for f in local_files:
                    # Charger le dataframe avec les colonnes Service-Public par défaut
                    available_cols = pd.read_parquet(f, columns=[]).columns.tolist()
                    cols_to_load = [col for col in REQUIRED_COLUMNS_SERVICE_PUBLIC if col in available_cols]

                    # Toujours inclure la colonne embedding
                    for col in available_cols:
                        if 'embedding' in col.lower() and col not in cols_to_load:
                            cols_to_load.append(col)

                    df = pd.read_parquet(f, columns=cols_to_load if cols_to_load else None)

                    # Convertir la colonne embeddings de string vers liste
                    for col in df.columns:
                        if 'embedding' in col.lower():
                            df[col] = df[col].apply(json.loads)
                            break
                    dfs.append(df)
                datasets_cache[dataset_name] = dfs
                gc.collect()
                logger.info(f"{dataset_name} dataset loaded successfully")
            else:
                # Télécharger si nécessaire
                logger.info(f"Downloading {dataset_name} files...")
                files_to_download = [
                    {'repo_id': dataset_config['repo_id'], 'path': f}
                    for f in dataset_config['files']
                ]
                datasets_cache[dataset_name] = await download_and_cache_files(files_to_download)
                logger.info(f"{dataset_name} files downloaded and loaded")

    logger.debug(f"Datasets cache contains: {list(datasets_cache.keys())}")
    return datasets_cache

def extract_text_from_row(row: pd.Series) -> str:
    """Extrait le texte pertinent d'une ligne"""
    text_fields = ['text', 'chunk_text', 'content', 'description', 'title', 'texte']

    for field in text_fields:
        if field in row.index:
            val = row[field]
            # Ignorer les numpy arrays et autres types non-textuels
            if isinstance(val, (np.ndarray, list)):
                continue
            if pd.notna(val):
                return str(val)

    # Fallback: concaténer tous les champs textuels
    texts = []
    for col in row.index:
        if col.startswith('_') or 'embedding' in col.lower():
            continue
        val = row[col]
        # Ignorer les numpy arrays et listes
        if isinstance(val, (np.ndarray, list)):
            continue
        if pd.notna(val) and isinstance(val, str) and len(val) > 10:
            texts.append(val)

    return ' '.join(texts[:3]) if texts else "Document sans texte"

@mcp.tool()
async def rechercher_article(
    numero_article: str,
    code: str | None = None,
    date_vigueur: str | None = None
) -> str:
    """Recherche directe d'un article par son numéro (sans recherche vectorielle)
    Args:
        numero_article: Numéro exact de l'article (ex: "1240", "16-3", etc.)
        code: Nom du code (ex: "legi_code_civil_part_0", "legi_code_de_commerce_part_0")
              Si non spécifié, cherche dans tous les codes LEGI configurés
        date_vigueur: Date au format YYYY-MM-DD pour filtrer les articles en vigueur à cette date
                     Si non spécifiée, retourne uniquement les articles avec status='VIGUEUR'
    """
    # Charger les datasets
    datasets = await load_datasets()

    if not datasets or 'legi' not in datasets:
        return "❌ Aucun dataset LEGI configuré. Cette fonction ne fonctionne qu'avec les codes LEGI."

    # Récupérer la configuration LEGI pour savoir quels codes sont chargés
    config = load_config()
    legi_config = config.get("selected_datasets", {}).get("legi", {})
    available_codes = [c['code'] for c in legi_config.get('codes', [])]

    # Si un code spécifique est demandé, vérifier qu'il est configuré
    if code:
        # Chercher un code qui contient le filtre
        matching_codes = [c for c in available_codes if code in c]
        if not matching_codes:
            return f"❌ Le code '{code}' n'est pas configuré. Codes disponibles: {', '.join(available_codes)}\n" \
                   f"Utilisez configurer_datasets() pour ajouter ce code."

    # Rechercher dans les dataframes LEGI
    found_articles = []

    legi_dataframes = datasets.get('legi', [])

    for df_idx, df in enumerate(legi_dataframes):
        # Si un code spécifique est demandé, vérifier qu'on est dans le bon dataframe
        if code:
            # Trouver le code correspondant à ce dataframe
            df_code = available_codes[df_idx] if df_idx < len(available_codes) else None
            # Vérifier si le code est présent dans le nom du parquet
            if df_code and code not in df_code:
                continue

        # Appliquer les filtres de vigueur
        filtered_df = df.copy()

        # Filtrer par statut ou date de vigueur
        if date_vigueur:
            # Si une date est spécifiée, filtrer les articles en vigueur à cette date
            try:
                from datetime import datetime
                target_date = datetime.strptime(date_vigueur, '%Y-%m-%d')

                filtered_df['_start_date_parsed'] = pd.to_datetime(filtered_df['start_date'], errors='coerce')
                filtered_df['_end_date_parsed'] = pd.to_datetime(filtered_df['end_date'], errors='coerce')

                mask = (filtered_df['_start_date_parsed'] <= target_date)
                mask &= ((filtered_df['_end_date_parsed'] >= target_date) | filtered_df['_end_date_parsed'].isna())

                filtered_df = filtered_df[mask]
                filtered_df = filtered_df.drop(columns=['_start_date_parsed', '_end_date_parsed'])
            except Exception:
                # En cas d'erreur, filtrer par statut VIGUEUR
                if 'status' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['status'] == 'VIGUEUR']
        else:
            # Par défaut, retourner uniquement les articles en vigueur
            if 'status' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['status'] == 'VIGUEUR']

        # Chercher une correspondance exacte sur le champ 'number'
        if 'number' not in filtered_df.columns:
            continue

        # Recherche exacte sur le numéro d'article
        matching_rows = filtered_df[filtered_df['number'].astype(str) == str(numero_article)]

        for idx, row in matching_rows.iterrows():
            # Créer le résultat
            result = {
                '_source': 'legi',
                '_index': int(idx),
                '_df_index': df_idx,
                '_matched_field': 'number',
                '_code': available_codes[df_idx] if df_idx < len(available_codes) else 'unknown'
            }

            # Ajouter tous les champs sauf embedding
            for col in df.columns:
                if 'embedding' not in col.lower():
                    val = row[col]
                    # Ignorer les numpy arrays et listes
                    if isinstance(val, (np.ndarray, list)):
                        continue
                    if pd.notna(val):
                        result[col] = str(val) if not isinstance(val, (str, int, float, bool)) else val

            result['_text'] = extract_text_from_row(row)
            found_articles.append(result)

    if not found_articles:
        msg = f"❌ Aucun article trouvé pour le numéro '{numero_article}'"
        if code:
            msg += f" dans le code '{code}'"
        if date_vigueur:
            msg += f" en vigueur au {date_vigueur}"
        else:
            msg += " (status='VIGUEUR')"
        return msg

    # Retourner uniquement le premier article (devrait être unique avec recherche exacte)
    result = found_articles[0]

    # Formater la sortie pour un seul article
    output = ""

    if len(found_articles) > 1:
        output += f"⚠️  Note: {len(found_articles)} versions trouvées, affichage de la première\n"
        output += "\n"

    # Afficher l'article unique - format simplifié
    # Extraire titre - utiliser le numéro d'article
    article_num = result.get('number', '')
    base_title = result.get('title', 'Article')
    title = f"{base_title} - Article {article_num}" if article_num else base_title

    output += f"{title}\n"

    # Afficher le statut
    status = result.get('status', '')
    if status:
        output += f"📌 Statut: {status}\n"

    # Afficher les dates
    start_date = result.get('date_debut', result.get('start_date', ''))
    end_date = result.get('date_fin', result.get('end_date', ''))
    if start_date or end_date:
        date_info = f"📅 Période: {start_date or 'N/A'} → {end_date or 'en cours'}\n"
        output += date_info

    # Afficher le doc_id
    doc_id = result.get('doc_id', '')
    if doc_id:
        output += f"doc_id: {doc_id}\n"

    # Afficher le contenu (text sans le titre qui est déjà affiché)
    output += f"\n"
    if 'text' in result and result['text']:
        text_content = result['text']
        # Enlever le titre du text s'il commence par le titre
        if text_content.startswith(title):
            text_content = text_content[len(title):].strip()
        output += f"{text_content}\n"
    elif '_text' in result:
        text = result['_text']
        # Enlever le titre du _text s'il commence par le titre
        if text.startswith(title):
            text = text[len(title):].strip()
        output += f"{text}\n"
    else:
        output += "Aucun contenu disponible.\n"

    return output

@mcp.tool()
async def rechercher(
    query: str,
    dataset_filter: str,
    date_vigueur: str | None = None,
    utiliser_reranker: bool = True,
    code_filter: str | None = None
) -> str:
    """Recherche sémantique dans un dataset spécifique
    Args:
        query: Question ou mots-clés de recherche
        dataset_filter: Dataset dans lequel rechercher (OBLIGATOIRE)
                       Valeurs possibles: "service-public", "legi", "constit", "dole", "cnil"
        date_vigueur: Date au format YYYY-MM-DD pour filtrer les articles LEGI en vigueur à cette date (optionnel)
                     Si non spécifiée, retourne uniquement les articles actuellement en vigueur (status='VIGUEUR')
        utiliser_reranker: Utiliser le reranker BGE-M3 pour améliorer les résultats (défaut: True)
        code_filter: Nom du code LEGI dans lequel effectuer la recherche (ex: "code_civil", "code_commerce")
                    Uniquement pour dataset_filter="legi"

    Returns:
        Les 10 meilleurs résultats de recherche avec scores de pertinence et contenu des documents

    Exemples:
        - rechercher("révocation dirigeant", dataset_filter="legi", code_filter="code_commerce")
        - rechercher("aide au logement", dataset_filter="service-public")
        - rechercher("article 1er", dataset_filter="constit")
        - rechercher("RGPD", dataset_filter="cnil")
    """
    global USE_HF_API

    try:
        logger.info(f"Starting search for query: {query[:50]}...")

        # Charger les datasets
        logger.debug("Loading datasets...")
        datasets = await load_datasets()

        if not datasets:
            logger.warning("No datasets configured")
            return "❌ Aucun dataset configuré.\n" \
                   "💡 Utilisez configurer_datasets(action='query_mediatech') pour voir les datasets disponibles\n" \
                   "💡 Puis configurer_datasets(action='ajouter_datasets', datasets=['...']) pour les télécharger"
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}", exc_info=True)
        raise

    # Vérifier que le dataset_filter est valide et installé
    config = load_config()
    installed_datasets = config.get("selected_datasets", {})

    if dataset_filter not in installed_datasets:
        # Lister les datasets installés
        output = f"❌ Le dataset '{dataset_filter}' n'est pas installé localement.\n\n"
        output += "📂 DATASETS INSTALLÉS:\n"

        if not installed_datasets:
            output += "   Aucun dataset installé.\n\n"
        else:
            for ds_name in installed_datasets.keys():
                if ds_name == 'legi':
                    legi_config = installed_datasets['legi']
                    codes = [c['code'] for c in legi_config.get('codes', [])]
                    output += f"   • legi ({len(codes)} codes: {', '.join(codes[:3])}"
                    if len(codes) > 3:
                        output += f"... +{len(codes)-3} autres"
                    output += ")\n"
                else:
                    output += f"   • {ds_name}\n"
            output += "\n"

        output += "💡 Pour installer un dataset:\n"
        output += "   configurer_datasets(action='query_mediatech')  # Voir datasets disponibles\n"
        output += f"   configurer_datasets(action='ajouter_datasets', datasets=['{dataset_filter}'])\n"

        return output

    # Si le dataset est LEGI et qu'un code_filter est spécifié, vérifier qu'il existe
    if code_filter:
        config = load_config()
        legi_config = config.get("selected_datasets", {}).get("legi", {})
        available_codes = [c['code'] for c in legi_config.get('codes', [])]

        # Chercher un code qui contient le filtre
        matching_codes = [code for code in available_codes if code_filter in code]

        if not matching_codes:
            return f"❌ Le code '{code_filter}' n'est pas configuré.\n" \
                   f"Codes disponibles: {', '.join(available_codes)}\n" \
                   f"Utilisez configurer_datasets(action='ajouter_datasets', datasets=['legi'], codes_legi=['{code_filter}']) pour l'ajouter."

    # Charger le modèle BGE-M3 (ou vérifier l'API HF)
    # Note: Si USE_HF_API=True, model sera None (on n'en a pas besoin pour les embeddings)
    # Mais on en aura besoin pour le reranking plus tard
    model = get_embedding_model()
    logger.debug(f"Embedding setup complete, USE_HF_API={USE_HF_API}")

    # Générer l'embedding de la requête
    logger.debug("Generating query embedding...")
    if USE_HF_API:
        try:
            logger.debug("Using HuggingFace API for embeddings")
            query_embedding_list = get_hf_embeddings([query], TOKEN_HF)
            query_embedding = query_embedding_list[0] if isinstance(query_embedding_list[0], list) else query_embedding_list
            logger.debug("Query embedding generated via HF API")
        except Exception as e:
            logger.warning(f"HF API failed, falling back to local model: {e}")
            USE_HF_API = False
            if model is None:
                model = get_embedding_model()
            query_embedding = model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs'][0]
            logger.debug("Query embedding generated via local model")
    else:
        logger.debug("Using local BGE-M3 model for embeddings")
        query_embedding = model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs'][0]
        logger.debug("Query embedding generated via local model")

    # Rechercher uniquement dans le dataset spécifié
    all_results = []

    # Vérifier que le dataset existe
    if dataset_filter not in datasets:
        return f"❌ Le dataset '{dataset_filter}' n'est pas chargé en mémoire.\n" \
               f"Datasets chargés: {', '.join(datasets.keys())}"

    # Récupérer la configuration LEGI pour le filtrage par code
    legi_config = config.get("selected_datasets", {}).get("legi", {})
    available_codes = [c['code'] for c in legi_config.get('codes', [])]

    logger.debug(f"Searching in dataset: {dataset_filter}")
    logger.debug(f"Available LEGI codes: {available_codes}")

    # Ne traiter que le dataset spécifié
    dataframes = datasets[dataset_filter]
    logger.debug(f"Processing dataset: {dataset_filter} with {len(dataframes)} dataframe(s)")

    for df_idx, df in enumerate(dataframes):
        # Si code_filter est spécifié pour LEGI, vérifier qu'on est dans le bon code
        if code_filter and dataset_filter == 'legi':
            df_code = available_codes[df_idx] if df_idx < len(available_codes) else None
            # Vérifier si le code_filter est présent dans le nom du parquet
            if df_code and code_filter not in df_code:
                continue

        # OPTIMISATION: Filtrer AVANT le calcul de similarité pour réduire le volume
        # Pour les datasets LEGI, appliquer les filtres de vigueur en premier
        filtered_df = df
        if dataset_filter == 'legi' and 'status' in df.columns:
            if date_vigueur:
                # Filtrer les articles en vigueur à la date spécifiée
                # Un article est en vigueur si : start_date <= date_vigueur <= end_date
                try:
                    from datetime import datetime
                    target_date = datetime.strptime(date_vigueur, '%Y-%m-%d')

                    # Convertir les dates en datetime
                    df['_start_date_parsed'] = pd.to_datetime(df['start_date'], errors='coerce')
                    df['_end_date_parsed'] = pd.to_datetime(df['end_date'], errors='coerce')

                    # Filtrer : start_date <= target_date <= end_date (ou end_date est None/NaT)
                    mask = (df['_start_date_parsed'] <= target_date)
                    mask &= ((df['_end_date_parsed'] >= target_date) | df['_end_date_parsed'].isna())

                    filtered_df = df[mask].copy()

                    # Supprimer les colonnes temporaires
                    filtered_df = filtered_df.drop(columns=['_start_date_parsed', '_end_date_parsed'])
                except Exception as e:
                    # En cas d'erreur, filtrer par statut VIGUEUR par défaut
                    filtered_df = df[df['status'] == 'VIGUEUR'].copy()
            else:
                # Par défaut, retourner uniquement les articles en vigueur actuellement
                filtered_df = df[df['status'] == 'VIGUEUR'].copy()

        # Trouver la colonne d'embedding
        embedding_col = None
        for col in filtered_df.columns:
            if 'embedding' in col.lower():
                embedding_col = col
                break

        if not embedding_col:
            logger.warning(f"No embedding column found in {dataset_filter} df#{df_idx}")
            continue

        logger.debug(f"Dataset {dataset_filter} df#{df_idx}: {len(filtered_df)} rows after filtering, embedding column: {embedding_col}")

        # Calculer les similarités (uniquement sur les données filtrées)
        computed_count = 0
        error_count = 0
        # Convertir query_embedding en liste si c'est un numpy array
        query_emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding

        for idx, row in filtered_df.iterrows():
            try:
                doc_embedding = row[embedding_col]
                score = cosine_similarity(query_emb_list, doc_embedding)
                computed_count += 1

                # Créer le résultat
                result = {
                    '_score': score,
                    '_source': dataset_filter,
                    '_index': int(idx),
                    '_df_index': df_idx
                }

                # Ajouter le nom du code si c'est LEGI
                if dataset_filter == 'legi':
                    result['_code'] = available_codes[df_idx] if df_idx < len(available_codes) else 'unknown'

                # Ajouter tous les champs sauf embedding
                for col in df.columns:
                    if col != embedding_col:
                        val = row[col]
                        # Ignorer les numpy arrays et listes (embeddings, vecteurs, etc.)
                        if isinstance(val, (np.ndarray, list)):
                            continue
                        if pd.notna(val):
                            result[col] = str(val) if not isinstance(val, (str, int, float, bool)) else val

                # Extraire le texte pour le reranking
                result['_text'] = extract_text_from_row(row)

                all_results.append(result)

            except Exception as e:
                error_count += 1
                if error_count <= 3:  # Log seulement les 3 premières erreurs
                    logger.error(f"Error computing similarity for row {idx}: {e}")
                continue

        logger.debug(f"Dataset {dataset_filter} df#{df_idx}: computed {computed_count} similarities, {error_count} errors")
    
    logger.info(f"Total results found: {len(all_results)}")

    if not all_results:
        return "Aucun résultat trouvé pour cette recherche."

    # Trier par score de similarité
    all_results.sort(key=lambda x: x['_score'], reverse=True)

    # Dédupliquer par doc_id pour éviter d'avoir plusieurs chunks du même document
    # On garde le chunk avec le meilleur score pour chaque doc_id
    seen_docs = set()
    deduplicated_results = []
    for result in all_results:
        doc_id = result.get('doc_id', None)
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            deduplicated_results.append(result)
        elif not doc_id:
            # Si pas de doc_id, garder le résultat
            deduplicated_results.append(result)

    # Prendre les 100 meilleurs résultats après déduplication
    top_results = deduplicated_results[:min(len(deduplicated_results), 100)]

    # Reranking avec BGE-M3 si demandé
    # Note: Le reranking utilise TOUJOURS le modèle local car l'API HF Inference
    # ne supporte pas les tâches de reranking cross-encoder de manière native
    final_results = top_results[:10]

    if utiliser_reranker and len(top_results) > 1:
        try:
            logger.debug(f"Starting reranking for {len(top_results)} results...")
            # Extraire les textes pour le reranking
            texts = [r['_text'][:1000] for r in top_results]

            # IMPORTANT: Le reranking nécessite le modèle local BGE-M3
            # L'API HF Inference ne supporte pas les tâches de reranking cross-encoder
            # Charger le modèle local si nécessaire
            if model is None or USE_HF_API:
                logger.debug("Loading local BGE-M3 model for reranking (required for accurate results)")
                # Sauvegarder l'état de USE_HF_API pour les embeddings
                use_hf_for_embeddings = USE_HF_API
                # Forcer le chargement du modèle local pour le reranking
                old_use_hf = USE_HF_API
                USE_HF_API = False
                model = get_embedding_model()
                # Restaurer pour les futures opérations d'embedding
                USE_HF_API = old_use_hf

            logger.debug("Using local BGE-M3 model for reranking (most accurate method)")
            rerank_scores_raw = model.compute_score(
                [[query, text] for text in texts],
                weights_for_different_modes=[0.4, 0.2, 0.4]
            )

            # Extraire les scores combinés du dictionnaire retourné par compute_score
            # compute_score retourne un dict avec les clés: 'colbert', 'sparse', 'dense',
            # 'sparse+dense', 'colbert+sparse+dense'
            if isinstance(rerank_scores_raw, dict):
                # Utiliser le score combiné des 3 modes (dense + sparse + colbert)
                rerank_scores = rerank_scores_raw.get('colbert+sparse+dense',
                                                      rerank_scores_raw.get('dense', []))
                logger.debug(f"Extracted combined scores from dict (using 'colbert+sparse+dense')")
            else:
                # Si ce n'est pas un dict (comportement futur?), l'utiliser tel quel
                rerank_scores = rerank_scores_raw

            logger.debug("Reranking completed via local model")

            # Combiner les scores (70% rerank, 30% similarité originale)
            for i, result in enumerate(top_results):
                original_score = result['_score']
                rerank_score = float(rerank_scores[i])
                result['_rerank_score'] = 0.3 * original_score + 0.7 * rerank_score

            # Retrier par score de reranking
            top_results.sort(key=lambda x: x.get('_rerank_score', 0), reverse=True)
            final_results = top_results[:10]
            logger.debug("Reranking and scoring complete")

        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            final_results = top_results[:10]

    # Sauvegarder les résultats dans le cache
    global search_results_cache
    search_results_cache.clear()
    for i, result in enumerate(final_results, 1):
        search_results_cache[i] = result

    # Formater la sortie
    output = ""

    for i, result in enumerate(final_results, 1):
        score = result.get('_rerank_score', result.get('_score', 0))
        source = result.get('_source', 'unknown')
        code_name = result.get('_code', None)

        # Extraire titre - pour LEGI, utiliser le numéro d'article
        if source == 'legi' and 'number' in result:
            article_num = result.get('number', '')
            base_title = result.get('title', 'Article')
            title = f"{base_title} - Article {article_num}" if article_num else base_title
        else:
            title = result.get('title', result.get('theme', result.get('nature', f'Document #{i}')))

        if isinstance(title, str) and len(title) > 100:
            title = title[:100] + "..."

        output += f"#{i} [{score*100:.1f}%] {title}\n"
        if code_name:
            output += f" - Code: {code_name}"
        output += "\n"

        # Pour le premier résultat, afficher les métadonnées complètes
        if i == 1:
            # Métadonnées spécifiques selon la source
            if source == 'service-public':
                # Afficher les métadonnées spécifiques à service-public
                audience = result.get('audience', '')
                surtitle = result.get('surtitle', '')
                url = result.get('url', '')

                if audience:
                    output += f"👥 Public: {audience}\n"
                if surtitle:
                    output += f"📋 Type: {surtitle}\n"
                if url:
                    output += f"🔗 URL: {url}\n"

            elif source == 'legi':
                # Afficher les métadonnées spécifiques à LEGI
                status = result.get('status', '')
                if status:
                    output += f"📌 Statut: {status}\n"

                start_date = result.get('date_debut', result.get('start_date', ''))
                end_date = result.get('date_fin', result.get('end_date', ''))
                if start_date or end_date:
                    date_info = f"📅 Période: {start_date or 'N/A'} → {end_date or 'en cours'}\n"
                    output += date_info

            # Afficher le contenu complet pour TOUS les résultats #1 (quelle que soit la source)
            if 'text' in result and result['text']:
                output += f"\n📄 Texte complet:{result['text']}\n"
        else:
            # Pour les résultats 2 à 10, afficher les 200 premiers caractères (TOUTES sources)
            text_content = result.get('text', '')
            if text_content:
                # Limiter à 200 caractères
                text_preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
                output += f"\n💬 Aperçu: {text_preview}\n"

    output += f"\n💡 Utilisez lire_resultat(numero) pour lire le contenu complet d'un résultat (ex: lire_resultat(2))\n"

    return output

@mcp.tool()
async def lire_resultat(numero: int) -> str:
    """Lit le contenu textuel complet d'un résultat de recherche
    Args:
        numero: Numéro du résultat à lire (2, 3, 4, etc. - pas 1 car déjà affiché)
    """
    global search_results_cache

    if not search_results_cache:
        return "❌ Aucune recherche récente. Effectuez d'abord une recherche avec rechercher()"

    if numero == 1:
        return "❌ Le contenu du résultat #1 est déjà affiché dans les résultats de recherche. Utilisez lire_resultat() pour les résultats #2 et suivants."

    if numero not in search_results_cache:
        available = list(search_results_cache.keys())
        return f"❌ Résultat #{numero} introuvable. Numéros disponibles: {', '.join(map(str, available))}"

    result = search_results_cache[numero]
    source = result.get('_source', 'unknown')

    # Titre
    if source == 'legi' and 'number' in result:
        article_num = result.get('number', '')
        base_title = result.get('title', 'Article')
        title = f"{base_title} - Article {article_num}" if article_num else base_title
    else:
        title = result.get('title', result.get('theme', result.get('nature', 'Document')))

    # Formater la sortie
    output = f"📄 RÉSULTAT #{numero} - {title}\n"

    # Contenu textuel
    if 'text' in result and result['text']:
        output += result['text']
    elif '_text' in result:
        output += result['_text']
    else:
        output += "Aucun contenu textuel disponible."

    return output

def main():
    """Main entry point for the MCP server"""
    try:
        logger.info("Starting Legifrance Vector MCP server...")
        # Don't preload model - it causes MCP initialization timeout
        # Model will be loaded on first search request
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()