# MCP Legifrance Vector Search

Serveur MCP (Model Context Protocol) pour la recherche sémantique dans les datasets AgentPublic : **Service-Public** et **LEGI (Légifrance)**.

Utilise BGE-M3 pour l'embedding et le reranking sémantique des résultats.

## ✨ Fonctionnalités

- 🔍 **Recherche sémantique** dans les datasets Service-Public et LEGI
- 🎯 **Reranking intelligent** avec BGE-M3 (60 résultats → top 10)
- 📅 **Filtrage temporel** : recherche d'articles en vigueur à une date donnée
- 📚 **Filtrage par code** : recherche dans un code spécifique (Code civil, Code de commerce, etc.)
- ⚡ **API Hugging Face** : utilisation optionnelle de l'API HF pour les embeddings
- 💾 **Cache local** : stockage des datasets téléchargés pour accès rapide

## 🚀 Installation

### Option 1 : Utilisation avec Claude Desktop (depuis GitHub)

**Prérequis : Installer `uv`**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Ceci installe `uv` et `uvx` dans `~/.local/bin`. Redémarrez votre terminal ou Claude Desktop après l'installation.

**Configuration**

Ajoutez cette configuration dans votre fichier de configuration Claude Desktop (`claude_desktop_config.json`) :

```json
{
  "mcpServers": {
    "Legifrance Vector": {
      "command": "/Users/VOTRE_USERNAME/.local/bin/uvx",
      "args": [
        "--from",
        "git+https://github.com/PieceMaker-Legal/MCP-Legifrance-Vector.git",
        "mcp-semantic-local"
      ],
      "env": {
        "HF_TOKEN": "votre_token_huggingface",
        "SEMANTIC_CACHE_DIR": "/chemin/vers/votre/cache"
      }
    }
  }
}
```

**Note:** Remplacez `VOTRE_USERNAME` par votre nom d'utilisateur, ou utilisez le chemin complet vers `uvx` (trouvable avec `which uvx` dans votre terminal).

### Option 2 : Installation locale pour développement

```bash
# Cloner le repo
git clone https://github.com/PieceMaker-Legal/MCP-Legifrance-Vector.git
cd MCP-Legifrance-Vector

# Installer avec uv
uv pip install -e .

# Ou avec pip
pip install -e .
```

Configuration Claude Desktop pour usage local :

```json
{
  "mcpServers": {
    "Legifrance Vector": {
      "command": "/chemin/vers/.local/bin/uv",
      "args": [
        "--directory",
        "/chemin/vers/MCP-Legifrance-Vector",
        "run",
        "server.py"
      ]
    }
  }
}
```

## 🔑 Configuration

### Variables d'environnement

- `HF_TOKEN` : Token Hugging Face (optionnel, pour utiliser l'API Inference)
- `SEMANTIC_CACHE_DIR` : Chemin du dossier de cache (défaut : `semantic_cache`)

Si non spécifiées, le serveur utilisera les valeurs par défaut pour le développement local.

### Obtenir un token Hugging Face

1. Créez un compte sur [Hugging Face](https://huggingface.co)
2. Allez dans Settings → Access Tokens
3. Créez un nouveau token avec les permissions de lecture
4. Ajoutez-le dans votre config Claude Desktop

## 📖 Outils MCP disponibles

### `rechercher(query, dataset_filter, date_vigueur?, code_filter?)`

Recherche sémantique dans un dataset spécifique.

**IMPORTANT :** Vous devez obligatoirement spécifier le dataset dans lequel rechercher.

**Paramètres :**
- `query` (string, requis) : Question ou mots-clés de recherche
- `dataset_filter` (string, **REQUIS**) : Dataset dans lequel rechercher
  - Valeurs possibles : `"service-public"`, `"legi"`, `"constit"`, `"dole"`, `"cnil"`
- `date_vigueur` (string, optionnel) : Date au format YYYY-MM-DD pour filtrer les articles LEGI en vigueur
- `code_filter` (string, optionnel) : Filtrer par code LEGI (ex: "code_commerce", "code_civil")
  - Uniquement pour `dataset_filter="legi"`
- `utiliser_reranker` (bool, défaut: true) : Utiliser le reranking BGE-M3

**Comportement :**
- Récupère les 100 meilleurs résultats par similarité cosinus
- Applique un reranking BGE-M3 pour affiner
- Retourne les 10 meilleurs résultats
- Pour LEGI : retourne uniquement les articles EN VIGUEUR (sauf si `date_vigueur` spécifiée)
- Si le dataset n'est pas installé, renvoie la liste des datasets disponibles localement

**Exemples :**
```javascript
// Recherche dans LEGI
rechercher("révocation dirigeant", dataset_filter: "legi", code_filter: "code_commerce")

// Recherche dans Service-Public
rechercher("aide au logement", dataset_filter: "service-public")

// Recherche dans la Constitution
rechercher("article 1er", dataset_filter: "constit")

// Recherche dans les délibérations CNIL
rechercher("RGPD", dataset_filter: "cnil")

// Recherche dans DOLE (Doctrine en ligne)
rechercher("responsabilité civile", dataset_filter: "dole")

// Recherche LEGI à une date donnée
rechercher("conditions révocation", dataset_filter: "legi", date_vigueur: "2020-01-01")
```

### `lire_resultat(numero)`

Lit le contenu complet d'un résultat de recherche.

**Paramètres :**
- `numero` (int, requis) : Numéro du résultat (2-10, pas 1 car déjà affiché)

**Exemple :**
```javascript
lire_resultat(3)
```

### `rechercher_article_direct(numero_article, code?, date_vigueur?)`

Recherche directe d'un article par son numéro (sans recherche vectorielle).

**Paramètres :**
- `numero_article` (string, requis) : Numéro de l'article (ex: "1224", "L. 225-18")
- `code` (string, optionnel) : Code dans lequel chercher
- `date_vigueur` (string, optionnel) : Date de vigueur

**Exemple :**
```javascript
rechercher_article_direct("L. 225-18", code: "code_commerce")
```

### `configurer_datasets(action, datasets?, codes_legi?)`

Gestion des datasets pour la recherche sémantique.

**Paramètres :**
- `action` (string, requis) : Action à effectuer - "lister_datasets", "ajouter_datasets", ou "query_mediatech"
- `datasets` (array, optionnel) : Liste des datasets à ajouter (pour "ajouter_datasets")
  - Valeurs possibles : "service-public", "legi", "constit", "dole", "cnil"
- `codes_legi` (array, optionnel) : Liste des codes LEGI à télécharger

**Exemples :**
```javascript
// Lister les datasets téléchargés localement
configurer_datasets(action: "lister_datasets")

// Lister les datasets disponibles sur HuggingFace
configurer_datasets(action: "query_mediatech")

// Ajouter Service-Public
configurer_datasets(
  action: "ajouter_datasets",
  datasets: ["service-public"]
)

// Ajouter des codes LEGI spécifiques
configurer_datasets(
  action: "ajouter_datasets",
  datasets: ["legi"],
  codes_legi: ["code_civil", "code_penal"]
)

// Ajouter Constitution
configurer_datasets(
  action: "ajouter_datasets",
  datasets: ["constit"]
)

// Ajouter DOLE (Doctrine en ligne)
configurer_datasets(
  action: "ajouter_datasets",
  datasets: ["dole"]
)

// Ajouter CNIL (Délibérations)
configurer_datasets(
  action: "ajouter_datasets",
  datasets: ["cnil"]
)

// Ajouter plusieurs datasets à la fois
configurer_datasets(
  action: "ajouter_datasets",
  datasets: ["service-public", "constit", "dole", "cnil"]
)
```

## 🏗️ Architecture

```
MCP Client (Claude)
    ↓
MCP Server (ce projet)
    ↓
┌─────────────┬──────────────┐
│  Hugging    │   Modèle     │
│  Face API   │   Local      │
│  (BGE-M3)   │  (BGE-M3)    │
└─────────────┴──────────────┘
    ↓
Datasets AgentPublic
- Service-Public (HF)
- LEGI (HF)
```

## 📊 Datasets utilisés

- **Service-Public** : [`AgentPublic/service-public`](https://huggingface.co/datasets/AgentPublic/service-public) - Fiches pratiques service-public.fr
- **LEGI** : [`AgentPublic/legi`](https://huggingface.co/datasets/AgentPublic/legi) - Codes juridiques français (Code civil, Code de commerce, etc.)
- **Constitution** : [`AgentPublic/constit`](https://huggingface.co/datasets/AgentPublic/constit) - Constitution française
- **DOLE** : [`AgentPublic/dole`](https://huggingface.co/datasets/AgentPublic/dole) - Doctrine en ligne
- **CNIL** : [`AgentPublic/cnil`](https://huggingface.co/datasets/AgentPublic/cnil) - Délibérations de la CNIL

Les datasets sont téléchargés automatiquement depuis Hugging Face lors de la première utilisation.

## 🔧 Développement

### Prérequis

- Python ≥ 3.10
- `uv` (recommandé) ou `pip`

### Installation pour développement

```bash
git clone https://github.com/PieceMaker-Legal/MCP-Legifrance-Vector.git
cd MCP-Legifrance-Vector

# Avec uv (recommandé)
uv pip install -e ".[dev]"

# Ou avec pip
pip install -e ".[dev]"
```

### Structure du projet

```
.
├── server.py                # Serveur MCP principal
├── pyproject.toml           # Configuration du package
├── README.md                # Cette documentation
├── .gitignore              # Fichiers exclus du repo
└── semantic_cache/         # Cache local (non committé)
    ├── datasets_config.json
    └── *.parquet
```

## 📝 Licence

MIT

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 🔗 Liens utiles

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Claude Desktop](https://claude.ai/download)
- [AgentPublic Datasets](https://huggingface.co/AgentPublic)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)

---

Développé par [PieceMaker Legal](https://piecemaker.legal)
